//! End-to-end Quartz SDXL base denoising pipeline.
//!
//! Structurally mirrors sd_pipeline.rs (SD1.5) with three real differences:
//! Euler scheduling (scale the model input every step, and the initial noise
//! by `init_noise_sigma`, neither of which SD1.5's PNDM scheduler needs),
//! 1024x1024-native [1,4,128,128] latents, and the extra pooled-embedding +
//! time_ids conditioning SDXL's UNet requires on every call.
//!
//! Deliberate scope simplification: `force_zeros_for_empty_prompt` (SDXL
//! zeroes conditioning for an empty negative prompt instead of encoding it)
//! is not implemented — an empty negative prompt is encoded normally, which
//! produces a near-equivalent low-signal embedding. Also: `time_ids` for both
//! branches assume no cropping and original_size == target_size == the
//! output resolution, which covers plain text-to-image generation.

use anyhow::{Result, bail};

use crate::{
    sd_pipeline,
    sd_scheduler::{classifier_free_guidance, gaussian_noise},
    sdxl::SdxlPack,
    sdxl_scheduler::SdxlScheduler,
    tensor::Tensor,
};

pub struct GenerationRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    /// Output side length in pixels; must be a multiple of 8. SDXL base was
    /// trained at 1024, but the UNet is fully convolutional.
    pub resolution: usize,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            steps: 20,
            guidance_scale: 5.0,
            seed: 0,
            resolution: 1024,
        }
    }
}

pub fn generate(pack: &SdxlPack, request: &GenerationRequest) -> Result<Tensor> {
    generate_with_control(pack, request, |_| {}, || false)
}

pub fn generate_with_control(
    pack: &SdxlPack,
    request: &GenerationRequest,
    mut on_progress: impl FnMut(sd_pipeline::GenerationProgress),
    is_cancelled: impl Fn() -> bool,
) -> Result<Tensor> {
    validate_request(request)?;
    #[cfg(feature = "vulkan")]
    let _scratch_guard = VulkanScratchGuard;
    ensure_running(&is_cancelled)?;

    on_progress(progress(sd_pipeline::GenerationPhase::Encoding, 0, 2));
    let (unconditional_context, unconditional_pooled) =
        pack.encode_prompt(&request.negative_prompt)?;
    ensure_running(&is_cancelled)?;
    on_progress(progress(sd_pipeline::GenerationPhase::Encoding, 1, 2));
    let (conditional_context, conditional_pooled) = pack.encode_prompt(&request.prompt)?;
    ensure_running(&is_cancelled)?;
    on_progress(progress(sd_pipeline::GenerationPhase::Encoding, 2, 2));

    let latent_side = request.resolution / 8;
    let latent_shape = vec![1, 4, latent_side, latent_side];
    let latent_len = 4 * latent_side * latent_side;
    let resolution = request.resolution as f32;
    let time_ids: [f32; 6] = [resolution, resolution, 0.0, 0.0, resolution, resolution];

    let mut scheduler = SdxlScheduler::new(request.steps)?;
    let timesteps = scheduler.timesteps().to_vec();
    let mut latents = Tensor::new(
        latent_shape.clone(),
        gaussian_noise(request.seed, latent_len)
            .into_iter()
            .map(|value| value * scheduler.init_noise_sigma())
            .collect(),
    )?;

    on_progress(progress(
        sd_pipeline::GenerationPhase::Denoising,
        0,
        timesteps.len(),
    ));
    for (step_index, timestep) in timesteps.iter().copied().enumerate() {
        ensure_running(&is_cancelled)?;
        let scaled_sample = Tensor::new(
            latent_shape.clone(),
            scheduler.scale_model_input(latents.data()),
        )?;

        let unconditional_noise = pack.predict_noise(
            &scaled_sample,
            timestep as f32,
            &unconditional_context,
            &unconditional_pooled,
            time_ids,
        )?;
        ensure_running(&is_cancelled)?;
        let conditional_noise = pack.predict_noise(
            &scaled_sample,
            timestep as f32,
            &conditional_context,
            &conditional_pooled,
            time_ids,
        )?;
        ensure_running(&is_cancelled)?;

        let guided = classifier_free_guidance(
            unconditional_noise.data(),
            conditional_noise.data(),
            request.guidance_scale,
        )?;
        let updated = scheduler.step(&guided, timestep, latents.data())?;
        latents = Tensor::new(latent_shape.clone(), updated)?;
        on_progress(progress(
            sd_pipeline::GenerationPhase::Denoising,
            step_index + 1,
            timesteps.len(),
        ));
    }

    on_progress(progress(sd_pipeline::GenerationPhase::Decoding, 0, 1));
    ensure_running(&is_cancelled)?;
    let image = pack.decode_latents(&latents)?;
    ensure_running(&is_cancelled)?;
    on_progress(progress(sd_pipeline::GenerationPhase::Decoding, 1, 1));
    Ok(image)
}

fn progress(
    phase: sd_pipeline::GenerationPhase,
    completed: usize,
    total: usize,
) -> sd_pipeline::GenerationProgress {
    sd_pipeline::GenerationProgress {
        phase,
        completed,
        total,
    }
}

fn ensure_running(is_cancelled: &impl Fn() -> bool) -> Result<()> {
    if is_cancelled() {
        bail!("generation cancelled");
    }
    Ok(())
}

fn validate_request(request: &GenerationRequest) -> Result<()> {
    if request.prompt.trim().is_empty() {
        bail!("generation prompt cannot be empty");
    }
    if request.steps == 0 || request.steps > 100 {
        bail!("generation steps must be between 1 and 100");
    }
    if !request.guidance_scale.is_finite() || !(0.0..=30.0).contains(&request.guidance_scale) {
        bail!("guidance scale must be finite and between 0 and 30");
    }
    if request.resolution == 0 || request.resolution % 8 != 0 {
        bail!("resolution must be a non-zero multiple of 8");
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
struct VulkanScratchGuard;

#[cfg(feature = "vulkan")]
impl Drop for VulkanScratchGuard {
    fn drop(&mut self) {
        if crate::vulkan::sd_acceleration_requested()
            && let Err(error) = crate::vulkan::trim_sd_scratch()
        {
            eprintln!("Quartz Vulkan scratch cleanup failed: {error:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_user_generation_limits() {
        let mut request = GenerationRequest::default();
        assert!(validate_request(&request).is_err());
        request.prompt = "lion".to_string();
        assert!(validate_request(&request).is_ok());
        request.resolution = 1023;
        assert!(validate_request(&request).is_err());
        request.resolution = 1024;
        request.steps = 101;
        assert!(validate_request(&request).is_err());
    }
}
