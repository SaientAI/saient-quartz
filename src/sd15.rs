//! Stable Diffusion 1.5 model-pack contract for Quartz.
//!
//! Quartz accepts the published Diffusers directory layout as input data, but
//! owns parsing, validation, tokenization, scheduling, and execution. No model
//! framework or inference runtime is loaded by this module.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::{
    clip_tokenizer::ClipTokenizer,
    safetensors::{DType, SafeTensorFile},
};

const TEXT_ENCODER_TENSORS: usize = 197;
const UNET_TENSORS: usize = 686;
const VAE_TENSORS: usize = 248;

pub struct Sd15Pack {
    root: PathBuf,
    text_encoder: SafeTensorFile,
    unet: SafeTensorFile,
    vae: SafeTensorFile,
    tokenizer: ClipTokenizer,
}

impl Sd15Pack {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let requested_root = root.as_ref();
        let root = requested_root.canonicalize().with_context(|| {
            format!(
                "cannot resolve model directory {}",
                requested_root.display()
            )
        })?;
        if !root.is_dir() {
            bail!("model root is not a directory: {}", root.display());
        }

        let model_index = read_json(&required_file(&root, "model_index.json")?)?;
        validate_model_index(&model_index)?;
        validate_text_config(&read_json(&required_file(
            &root,
            "text_encoder/config.json",
        )?)?)?;
        validate_unet_config(&read_json(&required_file(&root, "unet/config.json")?)?)?;
        validate_vae_config(&read_json(&required_file(&root, "vae/config.json")?)?)?;
        validate_scheduler_config(&read_json(&required_file(
            &root,
            "scheduler/scheduler_config.json",
        )?)?)?;

        let tokenizer = ClipTokenizer::from_files(
            required_file(&root, "tokenizer/vocab.json")?,
            required_file(&root, "tokenizer/merges.txt")?,
        )
        .context("invalid SD1.5 CLIP tokenizer")?;
        let empty_prompt = tokenizer.encode("");
        if empty_prompt[0] != 49_406 || empty_prompt[1] != 49_407 {
            bail!(
                "tokenizer special-token IDs are {}, {}; SD1.5 requires 49406, 49407",
                empty_prompt[0],
                empty_prompt[1]
            );
        }

        let text_encoder =
            SafeTensorFile::open(required_file(&root, "text_encoder/model.fp16.safetensors")?)
                .context("cannot load SD1.5 text encoder")?;
        let unet = SafeTensorFile::open(required_file(
            &root,
            "unet/diffusion_pytorch_model.fp16.safetensors",
        )?)
        .context("cannot load SD1.5 UNet")?;
        let vae = SafeTensorFile::open(required_file(
            &root,
            "vae/diffusion_pytorch_model.fp16.safetensors",
        )?)
        .context("cannot load SD1.5 VAE")?;

        validate_text_weights(&text_encoder)?;
        validate_unet_weights(&unet)?;
        validate_vae_weights(&vae)?;

        Ok(Self {
            root,
            text_encoder,
            unet,
            vae,
            tokenizer,
        })
    }

    pub fn print_summary(&self) {
        let total_bytes = self
            .text_encoder
            .mapped_len()
            .saturating_add(self.unet.mapped_len())
            .saturating_add(self.vae.mapped_len());
        println!("Saient SD1.5 pack: {}", self.root.display());
        println!(
            "text encoder : {:4} tensors  {:10} bytes",
            self.text_encoder.tensor_count(),
            self.text_encoder.mapped_len()
        );
        println!(
            "UNet         : {:4} tensors  {:10} bytes",
            self.unet.tensor_count(),
            self.unet.mapped_len()
        );
        println!(
            "VAE          : {:4} tensors  {:10} bytes",
            self.vae.tensor_count(),
            self.vae.mapped_len()
        );
        println!("weight total : {total_bytes} bytes");
        println!("contract     : SD1.5 / FP16 / 512x512 / CLIP-77");
        println!("validation   : passed");
    }

    pub fn print_tensors(&self) {
        print_component_tensors("text_encoder", &self.text_encoder);
        print_component_tensors("unet", &self.unet);
        print_component_tensors("vae", &self.vae);
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<crate::tensor::Tensor> {
        let tokens = self.tokenizer.encode(prompt);
        crate::sd_clip::encode(&self.text_encoder, &tokens)
    }

    pub fn decode_latents(&self, latents: &crate::tensor::Tensor) -> Result<crate::tensor::Tensor> {
        crate::sd_vae::decode(&self.vae, latents)
    }

    pub fn predict_noise(
        &self,
        sample: &crate::tensor::Tensor,
        timestep: f32,
        context: &crate::tensor::Tensor,
    ) -> Result<crate::tensor::Tensor> {
        crate::sd_unet::predict_noise(&self.unet, sample, timestep, context)
    }

    pub fn generate(
        &self,
        request: &crate::sd_pipeline::GenerationRequest,
    ) -> Result<crate::tensor::Tensor> {
        crate::sd_pipeline::generate(self, request)
    }

    pub fn decode_unscaled_latents(
        &self,
        latents: &crate::tensor::Tensor,
    ) -> Result<crate::tensor::Tensor> {
        crate::sd_vae::decode_unscaled(&self.vae, latents)
    }

    #[allow(dead_code)]
    pub fn tokenize(&self, prompt: &str) -> [u32; 77] {
        self.tokenizer.encode(prompt)
    }

    /// Opt this pack's UNet into staged Vulkan weight loading: the arena is
    /// capped to one block's worth of tensors (sized from this model's real
    /// tensor bytes) instead of the whole UNet file, evicting between blocks.
    /// Trades re-upload time across timesteps for a bounded peak weight
    /// footprint. No-op / returns an error if Vulkan support isn't compiled
    /// in; caller decides whether that's fatal.
    #[cfg(feature = "vulkan")]
    pub fn enable_staged_vulkan_loading(&self) -> Result<()> {
        let (budget_bytes, tensor_count_hint) = crate::sd_unet::staged_loading_budget(&self.unet)?;
        crate::vulkan::enable_staged_weight_loading(
            self.unet.mapping_key(),
            budget_bytes,
            tensor_count_hint,
        );
        Ok(())
    }
}

fn print_component_tensors(component: &str, file: &SafeTensorFile) {
    for name in file.tensor_names() {
        let info = file.info(name).expect("name came from the same tensor map");
        println!("{component:12} {name:88} {:?} {:?}", info.dtype, info.shape);
    }
}

pub(crate) fn required_file(root: &Path, relative: &str) -> Result<PathBuf> {
    let requested = root.join(relative);
    let resolved = requested
        .canonicalize()
        .with_context(|| format!("required model file is missing: {}", requested.display()))?;
    if !resolved.starts_with(root) {
        bail!("model file escapes the pack directory: {relative}");
    }
    if !resolved.is_file() {
        bail!(
            "required model path is not a regular file: {}",
            requested.display()
        );
    }
    Ok(resolved)
}

pub(crate) fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("invalid JSON in {}", path.display()))
}

pub(crate) fn expect_string(value: &Value, pointer: &str, expected: &str) -> Result<()> {
    let actual = value
        .pointer(pointer)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string config field {pointer}"))?;
    if actual != expected {
        bail!("config field {pointer} is {actual:?}; expected {expected:?}");
    }
    Ok(())
}

pub(crate) fn expect_u64(value: &Value, pointer: &str, expected: u64) -> Result<()> {
    let actual = value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing integer config field {pointer}"))?;
    if actual != expected {
        bail!("config field {pointer} is {actual}; expected {expected}");
    }
    Ok(())
}

pub(crate) fn expect_bool(value: &Value, pointer: &str, expected: bool) -> Result<()> {
    let actual = value
        .pointer(pointer)
        .and_then(Value::as_bool)
        .with_context(|| format!("missing boolean config field {pointer}"))?;
    if actual != expected {
        bail!("config field {pointer} is {actual}; expected {expected}");
    }
    Ok(())
}

pub(crate) fn expect_f64(value: &Value, pointer: &str, expected: f64) -> Result<()> {
    let actual = value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .with_context(|| format!("missing numeric config field {pointer}"))?;
    if (actual - expected).abs() > 1e-12 {
        bail!("config field {pointer} is {actual}; expected {expected}");
    }
    Ok(())
}

pub(crate) fn expect_u64_array(value: &Value, pointer: &str, expected: &[u64]) -> Result<()> {
    let array = value
        .pointer(pointer)
        .and_then(Value::as_array)
        .with_context(|| format!("missing array config field {pointer}"))?;
    let actual = array
        .iter()
        .map(|entry| entry.as_u64())
        .collect::<Option<Vec<_>>>()
        .with_context(|| format!("config field {pointer} must contain only integers"))?;
    if actual != expected {
        bail!("config field {pointer} is {actual:?}; expected {expected:?}");
    }
    Ok(())
}

fn validate_model_index(value: &Value) -> Result<()> {
    expect_string(value, "/_class_name", "StableDiffusionPipeline")?;
    expect_string(value, "/text_encoder/1", "CLIPTextModel")?;
    expect_string(value, "/tokenizer/1", "CLIPTokenizer")?;
    expect_string(value, "/unet/1", "UNet2DConditionModel")?;
    expect_string(value, "/vae/1", "AutoencoderKL")?;
    Ok(())
}

fn validate_text_config(value: &Value) -> Result<()> {
    expect_string(value, "/model_type", "clip_text_model")?;
    expect_string(value, "/hidden_act", "quick_gelu")?;
    expect_u64(value, "/hidden_size", 768)?;
    expect_u64(value, "/intermediate_size", 3_072)?;
    expect_u64(value, "/max_position_embeddings", 77)?;
    expect_u64(value, "/num_attention_heads", 12)?;
    expect_u64(value, "/num_hidden_layers", 12)?;
    expect_u64(value, "/vocab_size", 49_408)?;
    expect_f64(value, "/layer_norm_eps", 1e-5)?;
    Ok(())
}

fn validate_unet_config(value: &Value) -> Result<()> {
    expect_string(value, "/_class_name", "UNet2DConditionModel")?;
    expect_string(value, "/act_fn", "silu")?;
    expect_u64(value, "/attention_head_dim", 8)?;
    expect_u64_array(value, "/block_out_channels", &[320, 640, 1_280, 1_280])?;
    expect_u64(value, "/cross_attention_dim", 768)?;
    expect_bool(value, "/flip_sin_to_cos", true)?;
    expect_u64(value, "/in_channels", 4)?;
    expect_u64(value, "/layers_per_block", 2)?;
    expect_f64(value, "/norm_eps", 1e-5)?;
    expect_u64(value, "/norm_num_groups", 32)?;
    expect_u64(value, "/out_channels", 4)?;
    expect_u64(value, "/sample_size", 64)?;
    Ok(())
}

fn validate_vae_config(value: &Value) -> Result<()> {
    expect_string(value, "/_class_name", "AutoencoderKL")?;
    expect_string(value, "/act_fn", "silu")?;
    expect_u64_array(value, "/block_out_channels", &[128, 256, 512, 512])?;
    expect_u64(value, "/latent_channels", 4)?;
    expect_u64(value, "/layers_per_block", 2)?;
    expect_u64(value, "/norm_num_groups", 32)?;
    expect_u64(value, "/out_channels", 3)?;
    expect_u64(value, "/sample_size", 512)?;
    Ok(())
}

fn validate_scheduler_config(value: &Value) -> Result<()> {
    expect_string(value, "/_class_name", "PNDMScheduler")?;
    expect_string(value, "/beta_schedule", "scaled_linear")?;
    expect_f64(value, "/beta_start", 0.00085)?;
    expect_f64(value, "/beta_end", 0.012)?;
    expect_u64(value, "/num_train_timesteps", 1_000)?;
    expect_bool(value, "/set_alpha_to_one", false)?;
    expect_bool(value, "/skip_prk_steps", true)?;
    expect_u64(value, "/steps_offset", 1)?;
    Ok(())
}

fn require_tensor(file: &SafeTensorFile, name: &str, dtype: DType, shape: &[usize]) -> Result<()> {
    let info = file
        .info(name)
        .with_context(|| format!("required tensor is missing: {name}"))?;
    if info.dtype != dtype {
        bail!("tensor {name} is {:?}; expected {dtype:?}", info.dtype);
    }
    if info.shape != shape {
        bail!(
            "tensor {name} has shape {:?}; expected {shape:?}",
            info.shape
        );
    }
    Ok(())
}

fn require_count(file: &SafeTensorFile, component: &str, expected: usize) -> Result<()> {
    let actual = file.tensor_count();
    if actual != expected {
        bail!("{component} has {actual} tensors; SD1.5 FP16 requires {expected}");
    }
    Ok(())
}

fn require_all_f16(file: &SafeTensorFile, component: &str) -> Result<()> {
    for name in file.tensor_names() {
        let info = file.info(name).expect("name came from the same tensor map");
        if info.dtype != DType::F16 {
            bail!(
                "{component} tensor {name} is {:?}; the mobile pack requires F16",
                info.dtype
            );
        }
    }
    Ok(())
}

fn require_root_counts(
    file: &SafeTensorFile,
    component: &str,
    expected: &[(&str, usize)],
) -> Result<()> {
    let mut actual = BTreeMap::<&str, usize>::new();
    for name in file.tensor_names() {
        let root = name.split('.').next().expect("tensor names are nonempty");
        *actual.entry(root).or_default() += 1;
    }
    let expected = expected.iter().copied().collect::<BTreeMap<_, _>>();
    if actual != expected {
        bail!("{component} tensor tree is {actual:?}; expected {expected:?}");
    }
    Ok(())
}

fn validate_text_weights(file: &SafeTensorFile) -> Result<()> {
    require_count(file, "text encoder", TEXT_ENCODER_TENSORS)?;
    require_tensor(
        file,
        "text_model.embeddings.position_ids",
        DType::I64,
        &[1, 77],
    )?;
    require_tensor(
        file,
        "text_model.embeddings.position_embedding.weight",
        DType::F16,
        &[77, 768],
    )?;
    require_tensor(
        file,
        "text_model.embeddings.token_embedding.weight",
        DType::F16,
        &[49_408, 768],
    )?;

    for layer in 0..12 {
        let prefix = format!("text_model.encoder.layers.{layer}");
        for norm in ["layer_norm1", "layer_norm2"] {
            require_tensor(file, &format!("{prefix}.{norm}.weight"), DType::F16, &[768])?;
            require_tensor(file, &format!("{prefix}.{norm}.bias"), DType::F16, &[768])?;
        }
        require_tensor(
            file,
            &format!("{prefix}.mlp.fc1.weight"),
            DType::F16,
            &[3_072, 768],
        )?;
        require_tensor(
            file,
            &format!("{prefix}.mlp.fc1.bias"),
            DType::F16,
            &[3_072],
        )?;
        require_tensor(
            file,
            &format!("{prefix}.mlp.fc2.weight"),
            DType::F16,
            &[768, 3_072],
        )?;
        require_tensor(file, &format!("{prefix}.mlp.fc2.bias"), DType::F16, &[768])?;
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            require_tensor(
                file,
                &format!("{prefix}.self_attn.{projection}.weight"),
                DType::F16,
                &[768, 768],
            )?;
            require_tensor(
                file,
                &format!("{prefix}.self_attn.{projection}.bias"),
                DType::F16,
                &[768],
            )?;
        }
    }
    require_tensor(
        file,
        "text_model.final_layer_norm.weight",
        DType::F16,
        &[768],
    )?;
    require_tensor(file, "text_model.final_layer_norm.bias", DType::F16, &[768])?;
    Ok(())
}

fn validate_unet_weights(file: &SafeTensorFile) -> Result<()> {
    require_count(file, "UNet", UNET_TENSORS)?;
    require_all_f16(file, "UNet")?;
    require_root_counts(
        file,
        "UNet",
        &[
            ("conv_in", 2),
            ("conv_norm_out", 2),
            ("conv_out", 2),
            ("down_blocks", 246),
            ("mid_block", 46),
            ("time_embedding", 4),
            ("up_blocks", 384),
        ],
    )?;
    for (name, shape) in [
        ("conv_in.weight", &[320, 4, 3, 3][..]),
        ("conv_in.bias", &[320][..]),
        ("time_embedding.linear_1.weight", &[1_280, 320][..]),
        ("time_embedding.linear_2.weight", &[1_280, 1_280][..]),
        (
            "down_blocks.0.downsamplers.0.conv.weight",
            &[320, 320, 3, 3][..],
        ),
        (
            "down_blocks.1.resnets.0.conv1.weight",
            &[640, 320, 3, 3][..],
        ),
        (
            "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.weight",
            &[1_280, 768][..],
        ),
        (
            "mid_block.resnets.0.conv1.weight",
            &[1_280, 1_280, 3, 3][..],
        ),
        (
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.weight",
            &[1_280, 768][..],
        ),
        (
            "up_blocks.0.resnets.0.conv1.weight",
            &[1_280, 2_560, 3, 3][..],
        ),
        (
            "up_blocks.1.upsamplers.0.conv.weight",
            &[1_280, 1_280, 3, 3][..],
        ),
        ("up_blocks.3.resnets.2.conv2.weight", &[320, 320, 3, 3][..]),
        ("conv_out.weight", &[4, 320, 3, 3][..]),
        ("conv_out.bias", &[4][..]),
    ] {
        require_tensor(file, name, DType::F16, shape)?;
    }
    Ok(())
}

fn validate_vae_weights(file: &SafeTensorFile) -> Result<()> {
    require_count(file, "VAE", VAE_TENSORS)?;
    require_all_f16(file, "VAE")?;
    require_root_counts(
        file,
        "VAE",
        &[
            ("decoder", 138),
            ("encoder", 106),
            ("post_quant_conv", 2),
            ("quant_conv", 2),
        ],
    )?;
    for (name, shape) in [
        ("post_quant_conv.weight", &[4, 4, 1, 1][..]),
        ("decoder.conv_in.weight", &[512, 4, 3, 3][..]),
        (
            "decoder.mid_block.resnets.0.conv1.weight",
            &[512, 512, 3, 3][..],
        ),
        (
            "decoder.mid_block.attentions.0.to_q.weight",
            &[512, 512][..],
        ),
        (
            "decoder.up_blocks.0.resnets.0.conv1.weight",
            &[512, 512, 3, 3][..],
        ),
        (
            "decoder.up_blocks.0.upsamplers.0.conv.weight",
            &[512, 512, 3, 3][..],
        ),
        (
            "decoder.up_blocks.3.resnets.2.conv2.weight",
            &[128, 128, 3, 3][..],
        ),
        ("decoder.conv_norm_out.weight", &[128][..]),
        ("decoder.conv_out.weight", &[3, 128, 3, 3][..]),
        ("decoder.conv_out.bias", &[3][..]),
    ] {
        require_tensor(file, name, DType::F16, shape)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_non_sd15_text_configuration() {
        let config = serde_json::json!({
            "model_type": "clip_text_model",
            "hidden_act": "quick_gelu",
            "hidden_size": 1_024,
            "intermediate_size": 3_072,
            "max_position_embeddings": 77,
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "vocab_size": 49_408,
            "layer_norm_eps": 1e-5
        });
        let error = validate_text_config(&config).unwrap_err().to_string();
        assert!(error.contains("hidden_size"), "{error}");
    }

    #[test]
    fn validates_the_sd15_scheduler_contract() {
        let config = serde_json::json!({
            "_class_name": "PNDMScheduler",
            "beta_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "num_train_timesteps": 1000,
            "set_alpha_to_one": false,
            "skip_prk_steps": true,
            "steps_offset": 1
        });
        validate_scheduler_config(&config).unwrap();
    }
}
