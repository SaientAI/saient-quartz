//! Quartz-owned orchestration for the verified Wan 2.1 model stages.
//!
//! This module contains model semantics that sit between the DiT, flow scheduler, and VAE. The
//! operations remain expressed through `TensorBackend`, so the scalar and Vulkan paths execute the
//! same CFG ordering and latent normalization.

use anyhow::{Context, Result, bail};

use crate::{
    backend::{DeviceTensor, PreparedVectorHandle, TensorBackend},
    tensor::Tensor,
    wan_scheduler,
};

const LATENT_CHANNELS: usize = 16;

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
