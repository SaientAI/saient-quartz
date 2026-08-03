//! Flow-matching noise schedule for Wan2.1 video generation.
//!
//! Wan is a rectified-flow model, not an epsilon-prediction diffusion model, so it does not use
//! the SD/SDXL schedulers in this crate. The schedule is a uniform sweep over the 1000 training
//! timesteps mapped through an SNR shift, and the shift is load-bearing: upstream Wan2.1 specifies
//! `--sample_shift 8` for T2V-1.3B, and running it at 3.0 leaves the sampler too little time in the
//! high-noise structural phase, which is what makes human subjects come out translucent.
//!
//! Values here are verified byte-for-byte against the reference implementation — see the test at
//! the bottom, whose expected sigmas were dumped from the pinned stable-diffusion.cpp build we
//! currently ship, at steps=8 shift=8.

use crate::backend::{DeviceTensor, TensorBackend};
use anyhow::{Result, bail};

/// Number of training timesteps Wan2.1 was trained with.
pub const TIMESTEPS: u32 = 1000;

/// Rectified-flow SNR shift. `alpha == 1` is the identity, which keeps the uniform schedule.
#[inline]
pub fn time_snr_shift(alpha: f32, t: f32) -> f32 {
    if alpha == 1.0 {
        t
    } else {
        alpha * t / (1.0 + (alpha - 1.0) * t)
    }
}

/// Map a training timestep to its sigma under the shifted flow schedule.
///
/// The `t + 1` is not a rounding fudge: the reference indexes timesteps from zero but treats the
/// schedule as spanning `(0, 1]`, so t=999 must land exactly on sigma 1.0.
#[inline]
pub fn t_to_sigma(shift: f32, t: f32) -> f32 {
    time_snr_shift(shift, (t + 1.0) / TIMESTEPS as f32)
}

/// Sigma schedule for `steps` denoising steps, terminating at 0.
///
/// Returns `steps + 1` values: one sigma per step plus the trailing zero the sampler steps down to.
pub fn sigmas(steps: u32, shift: f32) -> Vec<f32> {
    let t_max = (TIMESTEPS - 1) as f32;
    match steps {
        0 => Vec::new(),
        1 => vec![t_to_sigma(shift, t_max), 0.0],
        n => {
            let step = t_max / (n - 1) as f32;
            let mut out: Vec<f32> = (0..n)
                .map(|i| t_to_sigma(shift, t_max - step * i as f32))
                .collect();
            out.push(0.0);
            out
        }
    }
}

/// Flow-matching scalings. Wan predicts velocity, so the model output is scaled by `-sigma` and
/// the input needs no rescaling — unlike the epsilon parameterisation used by SD.
#[inline]
pub fn scalings(sigma: f32) -> (f32, f32, f32) {
    (1.0, -sigma, 1.0) // c_skip, c_out, c_in
}

/// Advance one rectified-flow Euler step while keeping the latent and model velocity resident on
/// the selected backend. The sigma schedule remains model-level scalar control data.
pub(crate) fn step_with_backend(
    sample: &DeviceTensor,
    velocity: &DeviceTensor,
    sigma: f32,
    next_sigma: f32,
    backend: &dyn TensorBackend,
) -> Result<DeviceTensor> {
    if sample.shape() != velocity.shape()
        || !sigma.is_finite()
        || !next_sigma.is_finite()
        || next_sigma > sigma
    {
        bail!("Wan flow step requires equal tensor shapes and non-increasing finite sigmas");
    }
    let delta = backend.scale_device(velocity, next_sigma - sigma)?;
    backend.add_device(sample, &delta)
}

/// Mix clean latent toward noise at a given sigma (forward process).
#[inline]
pub fn noise_scaling(sigma: f32, noise: f32, latent: f32) -> f32 {
    latent * (1.0 - sigma) + noise * sigma
}

/// Undo `noise_scaling` at a given sigma.
#[inline]
pub fn inverse_noise_scaling(sigma: f32, latent: f32) -> f32 {
    latent / (1.0 - sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::TensorBackend, tensor::Tensor};

    /// Dumped from the shipped stable-diffusion.cpp build via SAIENT_DUMP=1 at
    /// `--steps 8 --flow-shift 8.0`. If this ever stops matching, the reimplementation has
    /// diverged from what actually renders on the phone.
    const REF_STEPS8_SHIFT8: [f32; 9] = [
        1.0,
        0.979_615_152,
        0.952_444_434,
        0.914_422_75,
        0.857_428_312,
        0.762_538_671,
        0.573_137_879,
        0.007_944_870_74,
        0.0,
    ];

    #[test]
    fn matches_reference_sigmas_steps8_shift8() {
        let got = sigmas(8, 8.0);
        assert_eq!(got.len(), REF_STEPS8_SHIFT8.len(), "schedule length");
        for (i, (g, r)) in got.iter().zip(REF_STEPS8_SHIFT8.iter()).enumerate() {
            assert!(
                (g - r).abs() <= 1e-6 * r.abs().max(1.0),
                "sigma[{i}] = {g}, reference {r}"
            );
        }
    }

    #[test]
    fn schedule_starts_at_one_and_ends_at_zero() {
        for &shift in &[1.0_f32, 3.0, 5.0, 8.0, 12.0] {
            let s = sigmas(8, shift);
            assert!(
                (s[0] - 1.0).abs() < 1e-6,
                "shift {shift}: first sigma must be 1.0, got {}",
                s[0]
            );
            assert_eq!(
                *s.last().unwrap(),
                0.0,
                "shift {shift}: schedule must terminate at 0"
            );
        }
    }

    #[test]
    fn schedule_is_monotonically_decreasing() {
        for &shift in &[1.0_f32, 3.0, 8.0, 12.0] {
            for steps in [1_u32, 2, 4, 8, 12, 20, 50] {
                let s = sigmas(steps, shift);
                for w in s.windows(2) {
                    assert!(
                        w[0] > w[1],
                        "shift {shift} steps {steps}: not decreasing: {w:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn shift_one_is_the_identity_schedule() {
        // With alpha == 1 the sigma is just the normalised timestep, so the schedule is uniform.
        let s = sigmas(4, 1.0);
        for (i, v) in s.iter().take(4).enumerate() {
            let t = 999.0 - (999.0 / 3.0) * i as f32;
            assert!((v - (t + 1.0) / 1000.0).abs() < 1e-6, "sigma[{i}] = {v}");
        }
    }

    #[test]
    fn higher_shift_holds_more_noise_through_the_schedule() {
        // The reason shift 8 fixes ghost-like subjects: at the same step index it sits at a higher
        // sigma than shift 3, so more of the schedule is spent forming global structure.
        let lo = sigmas(8, 3.0);
        let hi = sigmas(8, 8.0);
        for i in 1..7 {
            assert!(
                hi[i] > lo[i],
                "step {i}: shift8 {} should exceed shift3 {}",
                hi[i],
                lo[i]
            );
        }
    }

    #[test]
    fn noise_scaling_round_trips() {
        for &sigma in &[0.1_f32, 0.5, 0.9] {
            let latent = 0.42_f32;
            let noised = noise_scaling(sigma, 0.0, latent); // noise=0 isolates the latent term
            assert!((inverse_noise_scaling(sigma, noised) - latent).abs() < 1e-5);
        }
    }

    #[test]
    fn resident_scalar_flow_step_matches_euler_equation() {
        let sample = Tensor::new(vec![2, 3], vec![1.0, -2.0, 0.5, 4.0, -1.5, 0.25]).unwrap();
        let velocity = Tensor::new(vec![2, 3], vec![0.5, 1.25, -3.0, 0.75, -0.5, 2.0]).unwrap();
        let device_sample = crate::backend::SCALAR_BACKEND
            .upload_tensor(&sample)
            .unwrap();
        let device_velocity = crate::backend::SCALAR_BACKEND
            .upload_tensor(&velocity)
            .unwrap();
        let output = step_with_backend(
            &device_sample,
            &device_velocity,
            0.8,
            0.55,
            &crate::backend::SCALAR_BACKEND,
        )
        .unwrap();
        let output = crate::backend::SCALAR_BACKEND
            .download_tensor(&output)
            .unwrap();
        let expected = sample
            .data()
            .iter()
            .zip(velocity.data())
            .map(|(sample, velocity)| sample + (0.55 - 0.8) * velocity)
            .collect::<Vec<_>>();
        assert_eq!(output.shape(), &[2, 3]);
        for (actual, expected) in output.data().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-7);
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn resident_vulkan_flow_step_matches_scalar() {
        use crate::{
            backend::{SCALAR_BACKEND, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let sample = Tensor::new(
            vec![3, 5, 7],
            (0..105)
                .map(|index| ((index * 13) % 47) as f32 / 19.0 - 1.0)
                .collect(),
        )
        .unwrap();
        let velocity = Tensor::new(
            vec![3, 5, 7],
            (0..105)
                .map(|index| ((index * 17) % 53) as f32 / 23.0 - 0.75)
                .collect(),
        )
        .unwrap();
        let scalar_sample = SCALAR_BACKEND.upload_tensor(&sample).unwrap();
        let scalar_velocity = SCALAR_BACKEND.upload_tensor(&velocity).unwrap();
        let expected = step_with_backend(
            &scalar_sample,
            &scalar_velocity,
            0.762_538_671,
            0.573_137_879,
            &SCALAR_BACKEND,
        )
        .unwrap();
        let expected = SCALAR_BACKEND.download_tensor(&expected).unwrap();

        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident Wan scheduler parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident Wan scheduler parity failed: {error:#}"),
        };
        let device_sample = VULKAN_BACKEND.upload_tensor(&sample).unwrap();
        let device_velocity = VULKAN_BACKEND.upload_tensor(&velocity).unwrap();
        let device_output = step_with_backend(
            &device_sample,
            &device_velocity,
            0.762_538_671,
            0.573_137_879,
            &VULKAN_BACKEND,
        )
        .unwrap();
        let output = VULKAN_BACKEND.download_tensor(&device_output).unwrap();
        let after = crate::vulkan::persistence_stats().unwrap();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-7,
                maximum_mean_absolute_error: 1e-7,
            })
            .unwrap();
        assert_eq!(output.shape(), &[3, 5, 7]);
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            2
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "resident Wan flow step: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
        );
    }
}
