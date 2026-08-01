//! End-to-end Quartz SD1.5 denoising pipeline.

use anyhow::{Context, Result, bail};

use crate::{
    sd_scheduler::{Sd15Scheduler, classifier_free_guidance, gaussian_noise},
    sd15::Sd15Pack,
    tensor::Tensor,
};

pub struct GenerationRequest {
    pub prompt: String,
    pub negative_prompt: String,
    pub steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationPhase {
    Encoding,
    Denoising,
    Decoding,
}

impl GenerationPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Encoding => "encoding",
            Self::Denoising => "denoising",
            Self::Decoding => "decoding",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenerationProgress {
    pub phase: GenerationPhase,
    pub completed: usize,
    pub total: usize,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: String::new(),
            steps: 20,
            guidance_scale: 7.5,
            seed: 0,
        }
    }
}

pub fn generate(pack: &Sd15Pack, request: &GenerationRequest) -> Result<Tensor> {
    generate_with_control(pack, request, |_| {}, || false)
}

pub fn generate_with_control(
    pack: &Sd15Pack,
    request: &GenerationRequest,
    mut on_progress: impl FnMut(GenerationProgress),
    is_cancelled: impl Fn() -> bool,
) -> Result<Tensor> {
    validate_request(request)?;
    #[cfg(feature = "vulkan")]
    let _scratch_guard = VulkanScratchGuard;
    ensure_running(&is_cancelled)?;
    on_progress(GenerationProgress {
        phase: GenerationPhase::Encoding,
        completed: 0,
        total: 2,
    });
    let unconditional = pack.encode_prompt(&request.negative_prompt)?;
    ensure_running(&is_cancelled)?;
    on_progress(GenerationProgress {
        phase: GenerationPhase::Encoding,
        completed: 1,
        total: 2,
    });
    let conditional = pack.encode_prompt(&request.prompt)?;
    ensure_running(&is_cancelled)?;
    on_progress(GenerationProgress {
        phase: GenerationPhase::Encoding,
        completed: 2,
        total: 2,
    });

    let latent_shape = vec![1, 4, 64, 64];
    let mut latents = Tensor::new(
        latent_shape.clone(),
        gaussian_noise(request.seed, 4 * 64 * 64),
    )?;
    let mut scheduler = Sd15Scheduler::new(request.steps)?;
    let timesteps = scheduler.timesteps().to_vec();
    on_progress(GenerationProgress {
        phase: GenerationPhase::Denoising,
        completed: 0,
        total: timesteps.len(),
    });
    for (step_index, timestep) in timesteps.iter().copied().enumerate() {
        ensure_running(&is_cancelled)?;
        // The S24 executes two batch-one graphs faster than one batch-two graph,
        // and doing so halves peak activation pressure. Both predictions use the
        // same latents and weights, so classifier-free guidance is unchanged.
        let unconditional_noise = pack.predict_noise(&latents, timestep as f32, &unconditional)?;
        ensure_running(&is_cancelled)?;
        let conditional_noise = pack.predict_noise(&latents, timestep as f32, &conditional)?;
        ensure_running(&is_cancelled)?;
        let guided = classifier_free_guidance(
            unconditional_noise.data(),
            conditional_noise.data(),
            request.guidance_scale,
        )?;
        let updated = scheduler.step(&guided, timestep, latents.data())?;
        latents = Tensor::new(latent_shape.clone(), updated)?;
        on_progress(GenerationProgress {
            phase: GenerationPhase::Denoising,
            completed: step_index + 1,
            total: timesteps.len(),
        });
    }
    on_progress(GenerationProgress {
        phase: GenerationPhase::Decoding,
        completed: 0,
        total: 1,
    });
    ensure_running(&is_cancelled)?;
    let image = pack.decode_latents(&latents)?;
    ensure_running(&is_cancelled)?;
    on_progress(GenerationProgress {
        phase: GenerationPhase::Decoding,
        completed: 1,
        total: 1,
    });
    Ok(image)
}

fn ensure_running(is_cancelled: &impl Fn() -> bool) -> Result<()> {
    if is_cancelled() {
        bail!("generation cancelled");
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

pub fn rgb8(image: &Tensor) -> Result<Vec<u8>> {
    let [batch, channels, height, width]: [usize; 4] = image
        .shape()
        .try_into()
        .context("decoded image must be NCHW")?;
    if batch != 1 || channels != 3 {
        bail!("decoded image must have shape [1, 3, H, W]");
    }
    let plane = height * width;
    let mut output = vec![0; plane * 3];
    for pixel in 0..plane {
        for channel in 0..3 {
            let value = image.data()[channel * plane + pixel];
            output[pixel * 3 + channel] =
                ((value * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    Ok(output)
}

/// Encode a decoded RGB tensor as a standards-compliant PNG using stored
/// DEFLATE blocks. Image encoding stays small and deterministic without adding
/// an inference or codec runtime to the Android binary.
pub fn png(image: &Tensor) -> Result<Vec<u8>> {
    let [batch, channels, height, width]: [usize; 4] = image
        .shape()
        .try_into()
        .context("decoded image must be NCHW")?;
    if batch != 1 || channels != 3 || width == 0 || height == 0 {
        bail!("decoded image must have shape [1, 3, H, W]");
    }
    let width_u32 = u32::try_from(width).context("PNG width exceeds u32")?;
    let height_u32 = u32::try_from(height).context("PNG height exceeds u32")?;
    let rgb = rgb8(image)?;
    let row_bytes = width.checked_mul(3).context("PNG row size overflow")?;
    let raw_len = height
        .checked_mul(row_bytes.checked_add(1).context("PNG row size overflow")?)
        .context("PNG payload size overflow")?;
    let mut raw = Vec::with_capacity(raw_len);
    for row in rgb.chunks_exact(row_bytes) {
        raw.push(0); // PNG filter: None
        raw.extend_from_slice(row);
    }

    let mut zlib = Vec::with_capacity(raw.len() + raw.len() / 65_535 * 5 + 16);
    zlib.extend_from_slice(&[0x78, 0x01]); // deflate, 32 KiB window, fastest/no compression
    let mut offset = 0usize;
    while offset < raw.len() {
        let block_len = (raw.len() - offset).min(u16::MAX as usize);
        let final_block = offset + block_len == raw.len();
        zlib.push(u8::from(final_block)); // BFINAL plus stored-block BTYPE=00
        let length = block_len as u16;
        zlib.extend_from_slice(&length.to_le_bytes());
        zlib.extend_from_slice(&(!length).to_le_bytes());
        zlib.extend_from_slice(&raw[offset..offset + block_len]);
        offset += block_len;
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut output = Vec::with_capacity(zlib.len() + 64);
    output.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&width_u32.to_be_bytes());
    header.extend_from_slice(&height_u32.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB, deflate, filter, no interlace
    append_png_chunk(&mut output, b"IHDR", &header)?;
    append_png_chunk(&mut output, b"IDAT", &zlib)?;
    append_png_chunk(&mut output, b"IEND", &[])?;
    Ok(output)
}

fn append_png_chunk(output: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) -> Result<()> {
    let length = u32::try_from(data.len()).context("PNG chunk exceeds u32")?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(data);
    let mut checksum_input = Vec::with_capacity(kind.len() + data.len());
    checksum_input.extend_from_slice(kind);
    checksum_input.extend_from_slice(data);
    output.extend_from_slice(&crc32(&checksum_input).to_be_bytes());
    Ok(())
}

fn adler32(bytes: &[u8]) -> u32 {
    const MODULUS: u32 = 65_521;
    let mut first = 1u32;
    let mut second = 0u32;
    for &byte in bytes {
        first = (first + u32::from(byte)) % MODULUS;
        second = (second + first) % MODULUS;
    }
    second << 16 | first
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = crc >> 1 ^ (0xedb8_8320 & mask);
        }
    }
    !crc
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
    Ok(())
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
        request.steps = 101;
        assert!(validate_request(&request).is_err());
        request.steps = 20;
        request.guidance_scale = f32::NAN;
        assert!(validate_request(&request).is_err());
    }

    #[test]
    fn converts_nchw_decoder_range_to_interleaved_rgb() {
        let image = Tensor::new(vec![1, 3, 1, 2], vec![-1.0, 1.0, 0.0, 0.5, -0.5, 2.0]).unwrap();
        assert_eq!(rgb8(&image).unwrap(), vec![0, 128, 64, 255, 191, 255]);
    }

    #[test]
    fn writes_a_png_with_verified_checksums() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(adler32(b"Wikipedia"), 0x11e6_0398);
        let image = Tensor::new(vec![1, 3, 1, 1], vec![-1.0, 0.0, 1.0]).unwrap();
        let encoded = png(&image).unwrap();
        assert_eq!(&encoded[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(&encoded[12..16], b"IHDR");
        assert_eq!(&encoded[16..24], &[0, 0, 0, 1, 0, 0, 0, 1]);
        assert_eq!(&encoded[encoded.len() - 12..], b"\0\0\0\0IEND\xaeB`\x82");
    }

    #[test]
    fn cancellation_check_is_explicit() {
        assert!(ensure_running(&|| false).is_ok());
        assert!(ensure_running(&|| true).is_err());
    }
}
