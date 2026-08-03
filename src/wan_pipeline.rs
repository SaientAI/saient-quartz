//! Quartz-owned orchestration for the verified Wan 2.1 model stages.
//!
//! This module contains model semantics that sit between the DiT, flow scheduler, and VAE. The
//! operations remain expressed through `TensorBackend`, so the scalar and Vulkan paths execute the
//! same CFG ordering and latent normalization.

use anyhow::{Context, Result, bail};

use crate::{
    backend::{DeviceTensor, PreparedVectorHandle, TensorBackend},
    tensor::Tensor,
    wan::WanProfile,
    wan_scheduler,
};

#[cfg(feature = "vulkan")]
use crate::wan::WanModelPack;

#[cfg(all(test, feature = "vulkan"))]
fn read_test_reference_context(path: &std::path::Path) -> Result<Tensor> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("cannot read test reference context {}", path.display()))?;
    if bytes.len() < 8 || &bytes[..4] != b"SQD1" {
        bail!(
            "test reference context {} has an invalid SQD1 header",
            path.display()
        );
    }
    let dimensions = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let header = 8usize
        .checked_add(dimensions * 8)
        .context("test reference context header overflow")?;
    if bytes.len() < header || (bytes.len() - header) % 4 != 0 {
        bail!(
            "test reference context {} has an invalid length",
            path.display()
        );
    }
    let shape = (0..dimensions)
        .map(|index| {
            let offset = 8 + index * 8;
            i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
        })
        .collect::<Vec<_>>();
    if shape != [4096, 512, 1] {
        bail!(
            "test reference context {} has SQD1 shape {:?}; expected [4096, 512, 1]",
            path.display(),
            shape
        );
    }
    let values = bytes[header..]
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect::<Vec<_>>();
    Tensor::new(vec![512, 4096], values)
        .with_context(|| format!("construct test reference context {}", path.display()))
}

const LATENT_CHANNELS: usize = 16;
const PHILOX_MULTIPLIERS: [u32; 2] = [0xD251_1F53, 0xCD9E_8D57];
const PHILOX_WEYL: [u32; 2] = [0x9E37_79B9, 0xBB67_AE85];

#[derive(Clone, Debug)]
pub(crate) struct WanGenerationRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub steps: u32,
    pub guidance_scale: f32,
    pub flow_shift: f32,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WanGenerationPhase {
    Encoding,
    Denoising,
    Decoding,
}

impl WanGenerationPhase {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Encoding => "encoding",
            Self::Denoising => "denoising",
            Self::Decoding => "decoding",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WanGenerationProgress {
    pub phase: WanGenerationPhase,
    pub completed: usize,
    pub total: usize,
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct WanVulkanRunStats {
    pub runtime: std::time::Duration,
    pub peak_resident_bytes: u64,
    pub peak_device_local_bytes: u64,
    pub host_uploads: u64,
    pub weight_uploads: u64,
    pub downloads: u64,
    pub uploaded_bytes: u64,
    pub downloaded_bytes: u64,
    pub cache: crate::wan_vae::DeviceFeatureCacheStats,
}

#[cfg(feature = "vulkan")]
pub(crate) struct WanGenerationResult {
    pub video: Tensor,
    pub stats: WanVulkanRunStats,
}

/// Wan 2.1 VAE statistics from the pinned reference runner. Diffusion-space channel `c` is
/// converted to VAE space as `diffusion[c] * STD[c] + MEAN[c]`.
const LATENT_MEAN: [f32; LATENT_CHANNELS] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const LATENT_STD: [f32; LATENT_CHANNELS] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

/// Reference-compatible Philox 4x32 generator used to initialize Wan latents.
///
/// Each `randn` call advances the counter offset once, matching the pinned engine's `PhiloxRNG`.
/// The Box-Muller path intentionally converts the Philox words to `f32` before scaling because
/// that rounding is observable in the captured seed fixture.
pub(crate) struct WanPhiloxRng {
    seed: u64,
    offset: u32,
}

impl WanPhiloxRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { seed, offset: 0 }
    }

    pub(crate) fn randn(&mut self, count: usize) -> Result<Vec<f32>> {
        let count = u32::try_from(count).context("Wan Philox sample count exceeds u32")?;
        let key = [self.seed as u32, (self.seed >> 32) as u32];
        let mut output = Vec::with_capacity(count as usize);
        for index in 0..count {
            let mut counter = [self.offset, 0, index, 0];
            let mut round_key = key;
            for round in 0..10 {
                counter = philox_round(counter, round_key);
                if round != 9 {
                    round_key[0] = round_key[0].wrapping_add(PHILOX_WEYL[0]);
                    round_key[1] = round_key[1].wrapping_add(PHILOX_WEYL[1]);
                }
            }
            output.push(philox_box_muller(counter[0] as f32, counter[1] as f32));
        }
        self.offset = self.offset.wrapping_add(1);
        Ok(output)
    }
}

fn philox_round(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let product_0 = u64::from(counter[0]) * u64::from(PHILOX_MULTIPLIERS[0]);
    let product_1 = u64::from(counter[2]) * u64::from(PHILOX_MULTIPLIERS[1]);
    [
        (product_1 >> 32) as u32 ^ counter[1] ^ key[0],
        product_1 as u32,
        (product_0 >> 32) as u32 ^ counter[3] ^ key[1],
        product_0 as u32,
    ]
}

fn philox_box_muller(x: f32, y: f32) -> f32 {
    const TWO_POW_32_INVERSE: f32 = 2.328_306_4e-10;
    const TWO_POW_32_INVERSE_2_PI: f32 = TWO_POW_32_INVERSE * 6.283_185_5;
    let uniform_radius = x * TWO_POW_32_INVERSE + TWO_POW_32_INVERSE * 0.5;
    let uniform_angle = y * TWO_POW_32_INVERSE_2_PI + TWO_POW_32_INVERSE_2_PI * 0.5;
    (-2.0 * uniform_radius.ln()).sqrt() * uniform_angle.sin()
}

pub(crate) fn validate_generation_request(
    request: &WanGenerationRequest,
    profile: WanProfile,
) -> Result<[usize; 4]> {
    if request.width < 16
        || request.height < 16
        || request.width > profile.width
        || request.height > profile.height
        || request.width % 16 != 0
        || request.height % 16 != 0
    {
        bail!(
            "Wan dimensions must be multiples of 16 in the range 16x16..={}x{}, got {}x{}",
            profile.width,
            profile.height,
            request.width,
            request.height
        );
    }
    if request.frames < profile.minimum_frames
        || request.frames > profile.maximum_frames
        || (request.frames - 1) % 4 != 0
    {
        bail!(
            "Wan frame count must be 4k+1 in {}..={}, got {}",
            profile.minimum_frames,
            profile.maximum_frames,
            request.frames
        );
    }
    if request.steps == 0 || request.steps > 100 {
        bail!("Wan denoising steps must be in 1..=100");
    }
    if !request.guidance_scale.is_finite() || request.guidance_scale < 0.0 {
        bail!("Wan guidance scale must be finite and non-negative");
    }
    if !request.flow_shift.is_finite() || request.flow_shift <= 0.0 {
        bail!("Wan flow shift must be finite and positive");
    }

    let latent_time = (request.frames - 1) / 4 + 1;
    Ok([
        LATENT_CHANNELS,
        latent_time,
        request.height / 8,
        request.width / 8,
    ])
}

fn ensure_running(is_cancelled: &impl Fn() -> bool) -> Result<()> {
    if is_cancelled() {
        bail!("Wan generation cancelled");
    }
    Ok(())
}

/// Execute prompt encoding, classifier-free Wan denoising, and stateful VAE decoding through
/// Quartz-owned code. Model activations remain Vulkan-resident from the initial latent upload to
/// the final RGB download; progress callbacks and scalar schedule values are host-side control.
#[cfg(feature = "vulkan")]
pub(crate) fn generate_vulkan_with_control(
    pack: &WanModelPack,
    request: &WanGenerationRequest,
    mut on_progress: impl FnMut(WanGenerationProgress),
    is_cancelled: impl Fn() -> bool,
) -> Result<WanGenerationResult> {
    use crate::backend::VULKAN_BACKEND;

    struct ScratchGuard;
    impl Drop for ScratchGuard {
        fn drop(&mut self) {
            if let Err(error) = crate::vulkan::trim_sd_scratch() {
                eprintln!("Quartz Wan Vulkan scratch cleanup failed: {error:#}");
            }
        }
    }

    let latent_shape = validate_generation_request(request, pack.profile())?;
    let _scratch_guard = ScratchGuard;
    ensure_running(&is_cancelled)?;
    let before = crate::vulkan::persistence_stats().context("read initial Vulkan statistics")?;
    let started = std::time::Instant::now();

    on_progress(WanGenerationProgress {
        phase: WanGenerationPhase::Encoding,
        completed: 0,
        total: 2,
    });
    let unconditional = pack
        .encode_prompt(&request.negative_prompt)
        .context("encode Wan negative prompt")?;
    ensure_running(&is_cancelled)?;
    on_progress(WanGenerationProgress {
        phase: WanGenerationPhase::Encoding,
        completed: 1,
        total: 2,
    });
    let conditional = pack
        .encode_prompt(&request.prompt)
        .context("encode Wan prompt")?;
    ensure_running(&is_cancelled)?;
    on_progress(WanGenerationProgress {
        phase: WanGenerationPhase::Encoding,
        completed: 2,
        total: 2,
    });

    // Diagnostic isolation only: strict end-to-end tests can substitute the captured CUDA UMT5
    // outputs to determine whether a final-frame mismatch originates in prompt encoding or in a
    // downstream DiT/VAE operation. This branch is absent from non-test builds, so production
    // generation cannot accidentally depend on reference files or an external runtime.
    #[cfg(test)]
    let (unconditional, conditional) = if let Some(directory) =
        std::env::var_os("QUARTZ_TEST_WAN_REFERENCE_CONTEXT_DIR")
    {
        let directory = std::path::PathBuf::from(directory);
        let unconditional = read_test_reference_context(&directory.join("uncond_crossattn.bin"))?;
        let conditional = read_test_reference_context(&directory.join("cond_crossattn.bin"))?;
        eprintln!(
            "Wan strict diagnostic: substituted captured UMT5 contexts from {}",
            directory.display()
        );
        (unconditional, conditional)
    } else {
        (unconditional, conditional)
    };

    let backend: &dyn TensorBackend = &VULKAN_BACKEND;
    let conditional = backend
        .upload_tensor(&conditional)
        .context("upload Wan conditional context")?;
    let unconditional = backend
        .upload_tensor(&unconditional)
        .context("upload Wan unconditional context")?;
    let mut rng = WanPhiloxRng::new(request.seed);
    let latent_len = latent_shape.iter().product();
    let initial = Tensor::new(latent_shape.to_vec(), rng.randn(latent_len)?)?;
    let mut latent = backend
        .upload_tensor(&initial)
        .context("upload initial Wan latent")?;
    let sigmas = wan_scheduler::sigmas(request.steps, request.flow_shift);
    on_progress(WanGenerationProgress {
        phase: WanGenerationPhase::Denoising,
        completed: 0,
        total: request.steps as usize,
    });

    {
        let dit = pack.load_dit().context("load Wan DiT graph")?;
        let prepared = dit
            .prepare_wan_envelope(backend)
            .context("prepare Wan DiT envelope")?;
        let [_, time, height, width] = latent_shape;
        for (step_index, pair) in sigmas.windows(2).enumerate() {
            ensure_running(&is_cancelled)?;
            let sigma = pair[0];
            let next_sigma = pair[1];
            let timestep = sigma * wan_scheduler::TIMESTEPS as f32;
            let conditional_velocity = dit
                .forward_with_backend(
                    &latent,
                    time,
                    height,
                    width,
                    timestep,
                    &conditional,
                    conditional.shape()[0],
                    backend,
                    &prepared,
                )
                .with_context(|| format!("Wan conditional DiT step {step_index}"))?;
            ensure_running(&is_cancelled)?;
            let unconditional_velocity = dit
                .forward_with_backend(
                    &latent,
                    time,
                    height,
                    width,
                    timestep,
                    &unconditional,
                    unconditional.shape()[0],
                    backend,
                    &prepared,
                )
                .with_context(|| format!("Wan unconditional DiT step {step_index}"))?;
            latent = guided_flow_step_device(
                &latent,
                &conditional_velocity,
                &unconditional_velocity,
                request.guidance_scale,
                sigma,
                next_sigma,
                backend,
            )
            .with_context(|| format!("Wan flow scheduler step {step_index}"))?;
            on_progress(WanGenerationProgress {
                phase: WanGenerationPhase::Denoising,
                completed: step_index + 1,
                total: request.steps as usize,
            });
        }
    }

    ensure_running(&is_cancelled)?;
    on_progress(WanGenerationProgress {
        phase: WanGenerationPhase::Decoding,
        completed: 0,
        total: 1,
    });
    let transform = prepare_latent_transform(backend)?;
    let vae_latent = diffusion_to_vae_device(&latent, backend, &transform)?;
    let (video_device, cache) = {
        let vae = pack.load_vae().context("load Wan VAE graph")?;
        let prepared = vae
            .prepare_with_backend(backend)
            .context("prepare Wan VAE weights")?;
        vae.decode_device_with_backend(&vae_latent, backend, &prepared)
            .context("decode Wan video")?
    };
    let expected_shape = [1, 3, request.frames, request.height, request.width];
    if video_device.shape() != expected_shape {
        bail!(
            "Wan VAE produced {:?}; expected {expected_shape:?}",
            video_device.shape()
        );
    }
    let video = backend
        .download_tensor(&video_device)
        .context("download Wan RGB video")?;
    ensure_running(&is_cancelled)?;
    on_progress(WanGenerationProgress {
        phase: WanGenerationPhase::Decoding,
        completed: 1,
        total: 1,
    });
    let after = crate::vulkan::persistence_stats().context("read final Vulkan statistics")?;
    Ok(WanGenerationResult {
        video,
        stats: WanVulkanRunStats {
            runtime: started.elapsed(),
            peak_resident_bytes: after.peak_resident_allocated_bytes,
            peak_device_local_bytes: after.peak_resident_device_local_bytes,
            host_uploads: after.resident_tensor_uploads - before.resident_tensor_uploads,
            weight_uploads: after.resident_weight_uploads - before.resident_weight_uploads,
            downloads: after.resident_downloads - before.resident_downloads,
            uploaded_bytes: after.resident_uploaded_bytes - before.resident_uploaded_bytes,
            downloaded_bytes: after.resident_downloaded_bytes - before.resident_downloaded_bytes,
            cache,
        },
    })
}

/// Extract one `[1,3,H,W]` frame from the VAE's `[1,3,T,H,W]` `[0,1]` output. The returned image
/// uses the `[-1,1]` convention expected by Quartz's deterministic PNG encoder.
pub(crate) fn frame_for_png(video: &Tensor, frame: usize) -> Result<Tensor> {
    let [batch, channels, time, height, width]: [usize; 5] = video
        .shape()
        .try_into()
        .context("Wan video must be NCTHW")?;
    if batch != 1 || channels != 3 || time == 0 || frame >= time {
        bail!(
            "Wan video must have shape [1,3,T,H,W] and contain frame {frame}, got {:?}",
            video.shape()
        );
    }
    let plane = height * width;
    let channel_stride = time * plane;
    let mut values = Vec::with_capacity(3 * plane);
    for channel in 0..3 {
        let start = channel * channel_stride + frame * plane;
        values.extend(
            video.data()[start..start + plane]
                .iter()
                .map(|value| value * 2.0 - 1.0),
        );
    }
    Tensor::new(vec![1, 3, height, width], values)
}

pub(crate) struct PreparedWanLatentTransform {
    /// `[scale[16], bias[16]]`, consumed by the backend's NCTHW channel-affine operation.
    parameters: PreparedVectorHandle,
}

pub(crate) fn prepare_latent_transform(
    backend: &dyn TensorBackend,
) -> Result<PreparedWanLatentTransform> {
    let parameters = LATENT_STD
        .into_iter()
        .chain(LATENT_MEAN)
        .collect::<Vec<_>>();
    Ok(PreparedWanLatentTransform {
        parameters: backend.prepare_vector(&Tensor::new(vec![2 * LATENT_CHANNELS], parameters)?)?,
    })
}

/// Apply text classifier-free guidance without leaving backend-owned storage.
///
/// The ordering matches the pinned reference exactly:
/// `unconditional + guidance * (conditional - unconditional)`.
pub(crate) fn guided_velocity_device(
    conditional: &DeviceTensor,
    unconditional: &DeviceTensor,
    guidance: f32,
    backend: &dyn TensorBackend,
) -> Result<DeviceTensor> {
    if conditional.shape() != unconditional.shape() {
        bail!(
            "Wan CFG shape mismatch: conditional {:?}, unconditional {:?}",
            conditional.shape(),
            unconditional.shape()
        );
    }
    if !guidance.is_finite() {
        bail!("Wan CFG guidance must be finite");
    }
    let negative_unconditional = backend.scale_device(unconditional, -1.0)?;
    let delta = backend.add_device(conditional, &negative_unconditional)?;
    let guided_delta = backend.scale_device(&delta, guidance)?;
    backend.add_device(unconditional, &guided_delta)
}

/// Advance one resident Euler flow step after applying text CFG.
pub(crate) fn guided_flow_step_device(
    sample: &DeviceTensor,
    conditional: &DeviceTensor,
    unconditional: &DeviceTensor,
    guidance: f32,
    sigma: f32,
    next_sigma: f32,
    backend: &dyn TensorBackend,
) -> Result<DeviceTensor> {
    let velocity = guided_velocity_device(conditional, unconditional, guidance, backend)?;
    wan_scheduler::step_with_backend(sample, &velocity, sigma, next_sigma, backend)
}

/// Convert the DiT's `[C,T,H,W]` diffusion latent into the VAE's `[1,C,T,H,W]` layout and
/// per-channel value space. The reshape is metadata-only and the affine is device-resident.
pub(crate) fn diffusion_to_vae_device(
    diffusion: &DeviceTensor,
    backend: &dyn TensorBackend,
    prepared: &PreparedWanLatentTransform,
) -> Result<DeviceTensor> {
    let [channels, time, height, width]: [usize; 4] = diffusion
        .shape()
        .try_into()
        .context("Wan diffusion latent must be CTHW")?;
    if channels != LATENT_CHANNELS || time == 0 || height == 0 || width == 0 {
        bail!(
            "Wan diffusion latent must have shape [16,T,H,W] with non-zero axes, got {:?}",
            diffusion.shape()
        );
    }
    let ncthw = backend.reshape_device(diffusion, vec![1, channels, time, height, width])?;
    backend.ncthw_channel_affine_device(&ncthw, &prepared.parameters)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        backend::{SCALAR_BACKEND, TensorBackend},
        parity::{ParityTolerance, compare_tensors},
    };

    const REFERENCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/reference");

    fn read_dump(path: &Path) -> Result<(Vec<i64>, Vec<f32>)> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("cannot read captured tensor {}", path.display()))?;
        if bytes.len() < 8 || &bytes[..4] != b"SQD1" {
            bail!(
                "captured tensor {} has an invalid SQD1 header",
                path.display()
            );
        }
        let dimensions = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let header = 8usize
            .checked_add(dimensions * 8)
            .context("captured tensor header overflow")?;
        if bytes.len() < header || (bytes.len() - header) % 4 != 0 {
            bail!("captured tensor {} has an invalid length", path.display());
        }
        let shape = (0..dimensions)
            .map(|index| {
                let offset = 8 + index * 8;
                i64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
            })
            .collect::<Vec<_>>();
        let values = bytes[header..]
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>();
        Ok((shape, values))
    }

    fn captured_one_step_inputs() -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (sample_shape, sample) = read_dump(&Path::new(REFERENCE).join("dit/dit_in_0.bin"))?;
        let (second_sample_shape, second_sample) =
            read_dump(&Path::new(REFERENCE).join("dit/dit_in_1.bin"))?;
        let (conditional_shape, conditional) =
            read_dump(&Path::new(REFERENCE).join("dit/dit_out_0.bin"))?;
        let (unconditional_shape, unconditional) =
            read_dump(&Path::new(REFERENCE).join("dit/dit_out_1.bin"))?;
        let (vae_shape, vae) = read_dump(&Path::new(REFERENCE).join("vae/vae_in_full.bin"))?;
        let captured_shape = [52, 30, 2, 16, 1];
        if sample_shape != captured_shape
            || second_sample_shape != captured_shape
            || conditional_shape != captured_shape
            || unconditional_shape != captured_shape
            || vae_shape != captured_shape
        {
            bail!("captured one-step tensors do not share the expected full Wan shape");
        }
        if sample != second_sample {
            bail!("conditional and unconditional DiT evaluations used different latent inputs");
        }
        let cthw = vec![16, 2, 30, 52];
        Ok((
            Tensor::new(cthw.clone(), sample)?,
            Tensor::new(cthw.clone(), conditional)?,
            Tensor::new(cthw, unconditional)?,
            Tensor::new(vec![1, 16, 2, 30, 52], vae)?,
        ))
    }

    fn run_captured_one_step_seam(
        backend: &dyn TensorBackend,
    ) -> Result<(Tensor, crate::parity::ParityMetrics)> {
        let (sample, conditional, unconditional, expected) = captured_one_step_inputs()?;
        let sample = backend
            .upload_tensor(&sample)
            .context("upload captured Wan sample")?;
        let conditional = backend
            .upload_tensor(&conditional)
            .context("upload captured Wan conditional velocity")?;
        let unconditional = backend
            .upload_tensor(&unconditional)
            .context("upload captured Wan unconditional velocity")?;
        let transform =
            prepare_latent_transform(backend).context("prepare Wan latent transform")?;
        let diffusion = guided_flow_step_device(
            &sample,
            &conditional,
            &unconditional,
            6.0,
            1.0,
            0.0,
            backend,
        )
        .context("execute captured Wan CFG and flow step")?;
        let vae = diffusion_to_vae_device(&diffusion, backend, &transform)
            .context("execute captured Wan diffusion-to-VAE transform")?;
        let output = backend
            .download_tensor(&vae)
            .context("download captured Wan seam result")?;
        let metrics = compare_tensors(&output, &expected)?;
        Ok((output, metrics))
    }

    fn run_channel_affine_fixture(backend: &dyn TensorBackend) -> Result<Tensor> {
        const SHAPE: [usize; 5] = [2, 3, 2, 2, 3];
        let input = Tensor::new(
            SHAPE.to_vec(),
            (0..SHAPE.iter().product())
                .map(|index| index as f32 * 0.125 - 2.0)
                .collect(),
        )?;
        let parameters = Tensor::new(vec![6], vec![1.0, -2.0, 0.5, 0.25, 1.0, -3.0])?;
        let device_input = backend.upload_tensor(&input)?;
        let device_parameters = backend.prepare_vector(&parameters)?;
        let output = backend.ncthw_channel_affine_device(&device_input, &device_parameters)?;
        backend.download_tensor(&output)
    }

    #[test]
    fn channel_affine_uses_ncthw_channel_axis_for_multiple_batches() {
        const SHAPE: [usize; 5] = [2, 3, 2, 2, 3];
        let output = run_channel_affine_fixture(&SCALAR_BACKEND).unwrap();
        let plane = SHAPE[2] * SHAPE[3] * SHAPE[4];
        let scales = [1.0, -2.0, 0.5];
        let biases = [0.25, 1.0, -3.0];
        assert_eq!(output.shape(), SHAPE);
        for (index, actual) in output.data().iter().enumerate() {
            let channel = (index / plane) % SHAPE[1];
            let input = index as f32 * 0.125 - 2.0;
            assert_eq!(*actual, input * scales[channel] + biases[channel]);
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn resident_vulkan_channel_affine_matches_scalar_on_awkward_ncthw() {
        use crate::backend::VULKAN_BACKEND;

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let expected = run_channel_affine_fixture(&SCALAR_BACKEND).unwrap();
        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident NCTHW channel-affine parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident NCTHW channel-affine parity failed: {error:#}"),
        };
        let output = run_channel_affine_fixture(&VULKAN_BACKEND).unwrap();
        let after = crate::vulkan::persistence_stats().unwrap();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 1.0,
                maximum_absolute_error: 0.0,
                maximum_mean_absolute_error: 0.0,
            })
            .unwrap();
        assert_eq!(output.shape(), &[2, 3, 2, 2, 3]);
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "resident NCTHW channel affine: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
        );
    }

    #[test]
    fn captured_cfg_scheduler_and_vae_transform_match_reference() {
        let (output, metrics) = run_captured_one_step_seam(&SCALAR_BACKEND).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-5,
                maximum_mean_absolute_error: 2e-6,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, 16, 2, 30, 52]);
        println!(
            "captured scalar Wan one-step seam: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
        );
    }

    #[test]
    fn philox_seed_12345_matches_the_captured_initial_latent() {
        let (shape, expected) = read_dump(&Path::new(REFERENCE).join("dit/dit_in_0.bin")).unwrap();
        assert_eq!(shape, [52, 30, 2, 16, 1]);
        let mut rng = WanPhiloxRng::new(12_345);
        let actual = rng.randn(expected.len()).unwrap();
        let actual = Tensor::new(vec![16, 2, 30, 52], actual).unwrap();
        let expected = Tensor::new(vec![16, 2, 30, 52], expected).unwrap();
        let metrics = compare_tensors(&actual, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999_99,
                maximum_absolute_error: 1e-6,
                maximum_mean_absolute_error: 1e-7,
            })
            .unwrap();
        println!(
            "Philox seed 12345: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            actual.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
        );
    }

    #[test]
    fn generation_request_maps_video_dimensions_to_the_verified_latent_shape() {
        let request = WanGenerationRequest {
            prompt: "test".to_owned(),
            negative_prompt: String::new(),
            width: 416,
            height: 240,
            frames: 5,
            steps: 8,
            guidance_scale: 6.0,
            flow_shift: 8.0,
            seed: 12_345,
        };
        assert_eq!(
            validate_generation_request(
                &request,
                WanProfile {
                    width: 416,
                    height: 240,
                    fps: 8,
                    minimum_frames: 5,
                    maximum_frames: 41,
                }
            )
            .unwrap(),
            [16, 2, 30, 52]
        );
    }

    #[test]
    fn generation_request_rejects_unpatchable_dimensions_and_temporal_counts() {
        let profile = WanProfile {
            width: 416,
            height: 240,
            fps: 8,
            minimum_frames: 5,
            maximum_frames: 41,
        };
        let mut request = WanGenerationRequest {
            prompt: String::new(),
            negative_prompt: String::new(),
            width: 65,
            height: 64,
            frames: 5,
            steps: 1,
            guidance_scale: 6.0,
            flow_shift: 8.0,
            seed: 0,
        };
        assert!(validate_generation_request(&request, profile).is_err());
        request.width = 64;
        request.frames = 6;
        assert!(validate_generation_request(&request, profile).is_err());
    }

    #[test]
    fn frame_extraction_preserves_ncthw_channel_and_time_order() {
        let video = Tensor::new(
            vec![1, 3, 2, 1, 2],
            vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.25],
        )
        .unwrap();
        let frame = frame_for_png(&video, 1).unwrap();
        assert_eq!(frame.shape(), &[1, 3, 1, 2]);
        let expected = [-0.6, -0.4, 0.2, 0.4, 1.0, -0.5];
        for (actual, expected) in frame.data().iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads the complete Wan pack and runs native UMT5, two full DiT passes, and VAE"]
    fn full_native_vulkan_generation_matches_captured_reference() {
        const PACK: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack";

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let pack = WanModelPack::open(PACK).unwrap();
        let request = WanGenerationRequest {
            prompt: "a red fox".to_owned(),
            negative_prompt: String::new(),
            width: 416,
            height: 240,
            frames: 5,
            steps: 1,
            guidance_scale: 6.0,
            flow_shift: 8.0,
            seed: 12_345,
        };
        let result = generate_vulkan_with_control(&pack, &request, |_| {}, || false).unwrap();
        let (captured_shape, expected_values) =
            read_dump(&Path::new(REFERENCE).join("vae/vae_out_full.bin")).unwrap();
        assert_eq!(captured_shape, [416, 240, 5, 3, 1]);
        let expected = Tensor::new(vec![1, 3, 5, 240, 416], expected_values).unwrap();
        let metrics = compare_tensors(&result.video, &expected).unwrap();
        println!(
            "native full Wan generation: shape={:?} runtime_s={:.3} cosine={:.9} max_abs={:.9} mean_abs={:.9} peak_resident={} peak_device_local={} cache_peak={} host_uploads={} weight_uploads={} downloads={} uploaded_bytes={} downloaded_bytes={}",
            result.video.shape(),
            result.stats.runtime.as_secs_f64(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            result.stats.peak_resident_bytes,
            result.stats.peak_device_local_bytes,
            result.stats.cache.peak_bytes,
            result.stats.host_uploads,
            result.stats.weight_uploads,
            result.stats.downloads,
            result.stats.uploaded_bytes,
            result.stats.downloaded_bytes,
        );
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999,
                maximum_absolute_error: 0.03,
                maximum_mean_absolute_error: 0.005,
            })
            .unwrap();
        assert_eq!(result.video.shape(), &[1, 3, 5, 240, 416]);
        assert_eq!(result.stats.downloads, 1);
        assert!(result.stats.cache.all_slots_resident);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn resident_vulkan_cfg_scheduler_and_vae_transform_match_reference() {
        use crate::backend::VULKAN_BACKEND;

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping captured resident Wan seam parity: {error:#}");
                return;
            }
            Err(error) => panic!("required captured resident Wan seam parity failed: {error:#}"),
        };
        eprintln!(
            "captured resident Wan seam pre-state: resident_bytes={} device_local_bytes={} allocation_bytes={} scratch_bytes={} scratch_allocation_bytes={} cached_model_mappings={}",
            before.resident_allocated_bytes,
            before.resident_device_local_bytes,
            before.resident_device_local_allocation_bytes,
            before.scratch_buffer_bytes,
            before.scratch_buffer_allocation_bytes,
            before.cached_model_mappings,
        );
        let started = std::time::Instant::now();
        let (output, metrics) = run_captured_one_step_seam(&VULKAN_BACKEND).unwrap();
        let runtime = started.elapsed();
        let after = crate::vulkan::persistence_stats().unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-5,
                maximum_mean_absolute_error: 2e-6,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, 16, 2, 30, 52]);
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            3,
            "sample, conditional velocity, and unconditional velocity are uploaded once"
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            1,
            "Wan latent mean/std parameters are prepared once"
        );
        println!(
            "captured resident Wan one-step seam: shape={:?} runtime_ms={:.3} cosine={:.9} max_abs={:.9} mean_abs={:.9} current_vulkan_bytes={} peak_vulkan_bytes={} host_uploads={} weight_uploads={} downloads={}",
            output.shape(),
            runtime.as_secs_f64() * 1_000.0,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            after.peak_resident_allocated_bytes,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
        );
    }
}
