//! Quartz-owned Wan 2.1 model-pack loading and validation.
//!
//! The current migration package contains GGUF weights for UMT5 and the DiT plus SafeTensors
//! weights for the VAE. Those formats are treated only as data containers: Quartz validates and
//! executes every graph itself and never invokes an external inference runtime.

use std::{
    collections::BTreeMap,
    fmt::Write as _,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::{
    gguf::GgufFile,
    safetensors::SafeTensorFile,
    t5_tokenizer::T5Tokenizer,
    tensor::Tensor,
    umt5::Umt5Encoder,
    wan_dit::{WanConfig, WanDit},
    wan_vae::WanVae,
};

const MANIFEST_NAME: &str = "pack.json";
const SUPPORTED_PACK_ID: &str = "wan2.1-t2v-1.3b-q4-v1";
const SUPPORTED_REFERENCE_COMMIT: &str = "e31a86ce9110b11a98bd5990c329093244c2d1e3";
const DIT_NAME: &str = "wan2.1_t2v_1.3B_Q4_K.gguf";
const TEXT_ENCODER_NAME: &str = "umt5-xxl-encoder-Q4_K_M.gguf";
const VAE_NAME: &str = "wan_2.1_vae.safetensors";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WanManifest {
    id: String,
    backend_commit: String,
    files: Vec<ManifestFile>,
    total_bytes: u64,
    profile: WanProfile,
}

#[derive(Debug, Deserialize)]
struct ManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WanProfile {
    pub width: usize,
    pub height: usize,
    pub fps: usize,
    pub minimum_frames: usize,
    pub maximum_frames: usize,
}

pub(crate) struct WanModelPack {
    root: PathBuf,
    manifest: WanManifest,
    dit_weights: GgufFile,
    text_encoder_weights: GgufFile,
    vae_weights: SafeTensorFile,
    tokenizer: T5Tokenizer,
}

impl WanModelPack {
    /// Open and fully integrity-check the migration pack.
    ///
    /// SHA-256 validation deliberately happens before model parsing. The pack is loaded once per
    /// process and a corrupt multi-gigabyte mapping must fail before any tensor can reach a kernel.
    pub(crate) fn open(root: impl AsRef<Path>) -> Result<Self> {
        let requested_root = root.as_ref();
        let root = requested_root.canonicalize().with_context(|| {
            format!(
                "cannot resolve Wan model directory {}",
                requested_root.display()
            )
        })?;
        if !root.is_dir() {
            bail!("Wan model root is not a directory: {}", root.display());
        }

        let manifest_path = required_pack_file(&root, MANIFEST_NAME)?;
        let manifest_bytes = std::fs::read(&manifest_path)
            .with_context(|| format!("cannot read {}", manifest_path.display()))?;
        let manifest: WanManifest = serde_json::from_slice(&manifest_bytes)
            .with_context(|| format!("invalid Wan manifest {}", manifest_path.display()))?;
        let files = validate_manifest(&manifest)?;

        let mut resolved = BTreeMap::new();
        for (name, entry) in files {
            let path = required_pack_file(&root, name)?;
            let actual_bytes = path
                .metadata()
                .with_context(|| format!("cannot stat Wan component {}", path.display()))?
                .len();
            if actual_bytes != entry.bytes {
                bail!(
                    "Wan component {name} has {actual_bytes} bytes; manifest requires {}",
                    entry.bytes
                );
            }
            let actual_sha256 = sha256_file(&path)?;
            if actual_sha256 != entry.sha256 {
                bail!(
                    "Wan component {name} checksum is {actual_sha256}; manifest requires {}",
                    entry.sha256
                );
            }
            resolved.insert(name, path);
        }

        let dit_weights = GgufFile::open(&resolved[DIT_NAME]).context("invalid Wan DiT GGUF")?;
        let text_encoder_weights =
            GgufFile::open(&resolved[TEXT_ENCODER_NAME]).context("invalid Wan UMT5 GGUF")?;
        let vae_weights =
            SafeTensorFile::open(&resolved[VAE_NAME]).context("invalid Wan VAE SafeTensors")?;
        let tokenizer = T5Tokenizer::from_gguf(&text_encoder_weights)
            .context("Wan UMT5 GGUF does not contain a valid Unigram tokenizer")?;

        // Constructing these borrowed graph descriptions validates every required GGUF tensor and
        // every fixed architecture dimension without retaining a second copy of the large files.
        WanDit::load(&dit_weights, WanConfig::default()).context("invalid Wan DiT graph")?;
        Umt5Encoder::load(&text_encoder_weights).context("invalid Wan UMT5 graph")?;
        validate_vae_container(&vae_weights)?;

        Ok(Self {
            root,
            manifest,
            dit_weights,
            text_encoder_weights,
            vae_weights,
            tokenizer,
        })
    }

    pub(crate) fn profile(&self) -> WanProfile {
        self.manifest.profile
    }

    pub(crate) fn print_summary(&self) {
        println!("Quartz Wan pack : {}", self.root.display());
        println!("package id      : {}", self.manifest.id);
        println!("reference commit: {}", self.manifest.backend_commit);
        println!("DiT tensors     : {}", self.dit_weights.tensors.len());
        println!(
            "UMT5 tensors    : {}",
            self.text_encoder_weights.tensors.len()
        );
        println!("VAE tensors     : {}", self.vae_weights.tensor_count());
        println!("weight bytes    : {}", self.manifest.total_bytes);
        println!(
            "profile         : {}x{}, {} fps, {}..={} frames",
            self.manifest.profile.width,
            self.manifest.profile.height,
            self.manifest.profile.fps,
            self.manifest.profile.minimum_frames,
            self.manifest.profile.maximum_frames,
        );
        println!("integrity       : SHA-256 passed");
        println!("execution       : Quartz-owned UMT5 / Wan DiT / Wan VAE");
    }

    /// Encode to the fixed `[512,4096]` context consumed by the verified DiT graph.
    pub(crate) fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let context = self.text_encoder_weights_context();
        let unpadded = self.tokenizer.encode(prompt);
        let valid_tokens = unpadded.len().min(context);
        let tokens = self.tokenizer.encode_padded(prompt, context);
        let encoder = Umt5Encoder::load(&self.text_encoder_weights)?;
        Tensor::new(
            vec![context, encoder.cfg.d_model],
            encoder.forward(&tokens, valid_tokens),
        )
    }

    pub(crate) fn load_dit(&self) -> Result<WanDit<'_>> {
        WanDit::load(&self.dit_weights, WanConfig::default())
    }

    pub(crate) fn load_vae(&self) -> Result<WanVae> {
        WanVae::load(&self.vae_weights)
    }

    fn text_encoder_weights_context(&self) -> usize {
        crate::umt5::Umt5Config::from_gguf(&self.text_encoder_weights).context
    }
}

fn validate_manifest(manifest: &WanManifest) -> Result<BTreeMap<&str, &ManifestFile>> {
    if manifest.id != SUPPORTED_PACK_ID {
        bail!(
            "unsupported Wan package id {:?}; expected {SUPPORTED_PACK_ID:?}",
            manifest.id
        );
    }
    if manifest.backend_commit != SUPPORTED_REFERENCE_COMMIT {
        bail!(
            "Wan reference commit {:?} is incompatible; expected {SUPPORTED_REFERENCE_COMMIT}",
            manifest.backend_commit
        );
    }
    if manifest.profile
        != (WanProfile {
            width: 416,
            height: 240,
            fps: 8,
            minimum_frames: 5,
            maximum_frames: 41,
        })
    {
        bail!(
            "Wan package profile is incompatible: {:?}",
            manifest.profile
        );
    }

    let expected = [DIT_NAME, TEXT_ENCODER_NAME, VAE_NAME];
    let mut files = BTreeMap::new();
    for file in &manifest.files {
        if !expected.contains(&file.path.as_str()) {
            bail!(
                "Wan manifest contains unsupported component {:?}",
                file.path
            );
        }
        if file.bytes == 0
            || file.sha256.len() != 64
            || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
            || file.sha256.bytes().any(|byte| byte.is_ascii_uppercase())
        {
            bail!(
                "Wan manifest component {} has invalid integrity metadata",
                file.path
            );
        }
        if files.insert(file.path.as_str(), file).is_some() {
            bail!("Wan manifest duplicates component {}", file.path);
        }
    }
    for name in expected {
        if !files.contains_key(name) {
            bail!("Wan manifest is missing component {name}");
        }
    }
    let total = files.values().try_fold(0u64, |total, file| {
        total
            .checked_add(file.bytes)
            .context("Wan manifest byte total overflow")
    })?;
    if total != manifest.total_bytes {
        bail!(
            "Wan manifest total is {}; component sizes sum to {total}",
            manifest.total_bytes
        );
    }
    Ok(files)
}

fn validate_vae_container(weights: &SafeTensorFile) -> Result<()> {
    for name in [
        "conv2.weight",
        "decoder.conv1.weight",
        "decoder.middle.1.to_qkv.weight",
        "decoder.head.0.gamma",
        "decoder.head.2.weight",
    ] {
        let info = weights
            .info(name)
            .with_context(|| format!("Wan VAE is missing required tensor {name}"))?;
        if info.element_count() == 0 {
            bail!("Wan VAE tensor {name} is empty");
        }
    }
    Ok(())
}

fn required_pack_file(root: &Path, name: &str) -> Result<PathBuf> {
    let requested = root.join(name);
    let resolved = requested
        .canonicalize()
        .with_context(|| format!("required Wan pack file is missing: {}", requested.display()))?;
    if !resolved.starts_with(root) {
        bail!("Wan pack file escapes the model directory: {name}");
    }
    if !resolved.is_file() {
        bail!(
            "Wan pack path is not a regular file: {}",
            resolved.display()
        );
    }
    Ok(resolved)
}

fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("cannot hash {}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("cannot hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> WanManifest {
        WanManifest {
            id: SUPPORTED_PACK_ID.to_owned(),
            backend_commit: SUPPORTED_REFERENCE_COMMIT.to_owned(),
            files: vec![
                ManifestFile {
                    path: DIT_NAME.to_owned(),
                    bytes: 11,
                    sha256: "0".repeat(64),
                },
                ManifestFile {
                    path: TEXT_ENCODER_NAME.to_owned(),
                    bytes: 13,
                    sha256: "1".repeat(64),
                },
                ManifestFile {
                    path: VAE_NAME.to_owned(),
                    bytes: 17,
                    sha256: "a".repeat(64),
                },
            ],
            total_bytes: 41,
            profile: WanProfile {
                width: 416,
                height: 240,
                fps: 8,
                minimum_frames: 5,
                maximum_frames: 41,
            },
        }
    }

    #[test]
    fn accepts_the_pinned_three_component_manifest() {
        let manifest = manifest();
        let files = validate_manifest(&manifest).unwrap();
        assert_eq!(
            files.keys().copied().collect::<Vec<_>>(),
            [TEXT_ENCODER_NAME, DIT_NAME, VAE_NAME]
        );
    }

    #[test]
    fn rejects_duplicate_components_and_incorrect_totals() {
        let mut duplicate = manifest();
        duplicate.files.push(ManifestFile {
            path: DIT_NAME.to_owned(),
            bytes: 1,
            sha256: "2".repeat(64),
        });
        assert!(
            validate_manifest(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicates")
        );

        let mut wrong_total = manifest();
        wrong_total.total_bytes += 1;
        assert!(
            validate_manifest(&wrong_total)
                .unwrap_err()
                .to_string()
                .contains("component sizes sum")
        );
    }

    #[test]
    fn sha256_streaming_matches_the_standard_abc_digest() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("quartz-wan-sha256-{nonce}"));
        std::fs::write(&path, b"abc").unwrap();
        let digest = sha256_file(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
