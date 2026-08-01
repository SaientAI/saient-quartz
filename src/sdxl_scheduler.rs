//! SDXL's EulerDiscreteScheduler: scaled-linear betas, "leading" timestep
//! spacing with offset 1, deterministic (non-ancestral) Euler ODE step,
//! epsilon prediction. Distinct algorithm from SD1.5's PNDM/PLMS scheduler in
//! sd_scheduler.rs — the two are not interchangeable.

use anyhow::{Context, Result, bail};

const TRAIN_TIMESTEPS: usize = 1_000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

pub struct SdxlScheduler {
    sigmas: Vec<f32>,
    timesteps: Vec<usize>,
    counter: usize,
}

impl SdxlScheduler {
    pub fn new(inference_steps: usize) -> Result<Self> {
        if inference_steps == 0 || inference_steps > TRAIN_TIMESTEPS {
            bail!("SDXL inference steps must be between 1 and {TRAIN_TIMESTEPS}");
        }
        let beta_start_sqrt = (BETA_START as f32).sqrt();
        let beta_end_sqrt = (BETA_END as f32).sqrt();
        let mut product = 1.0f32;
        let mut alphas_cumprod = Vec::with_capacity(TRAIN_TIMESTEPS);
        for index in 0..TRAIN_TIMESTEPS {
            let fraction = index as f32 / (TRAIN_TIMESTEPS - 1) as f32;
            let beta_sqrt = beta_start_sqrt + fraction * (beta_end_sqrt - beta_start_sqrt);
            let beta = beta_sqrt * beta_sqrt;
            product *= 1.0 - beta;
            alphas_cumprod.push(product);
        }
        // sigma(t) = sqrt((1 - alpha_cumprod[t]) / alpha_cumprod[t]), full training schedule.
        let full_sigmas: Vec<f32> = alphas_cumprod
            .iter()
            .map(|&alpha| ((1.0 - alpha) / alpha).sqrt())
            .collect();

        // timestep_spacing="leading", steps_offset=1: evenly spaced indices into the
        // training schedule, then +1, matching diffusers' leading-spacing branch.
        let step_ratio = TRAIN_TIMESTEPS / inference_steps;
        let timesteps: Vec<usize> = (0..inference_steps)
            .map(|index| index * step_ratio + 1)
            .rev()
            .collect();

        // Sigmas at those timesteps (linear interpolation into the continuous training
        // index, matching interpolation_type="linear"), descending, with a trailing 0.
        let mut sigmas: Vec<f32> = timesteps
            .iter()
            .map(|&timestep| interpolate_sigma(&full_sigmas, timestep))
            .collect();
        sigmas.push(0.0);

        Ok(Self {
            sigmas,
            timesteps,
            counter: 0,
        })
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
    }

    /// Sigma for the timestep about to be predicted (call before running the UNet).
    pub fn current_sigma(&self) -> f32 {
        self.sigmas[self.counter]
    }

    /// Karras/EDM input preconditioning: divide the sample by sqrt(sigma^2 + 1)
    /// before it enters the UNet. Required for every step, including the first.
    pub fn scale_model_input(&self, sample: &[f32]) -> Vec<f32> {
        let sigma = self.current_sigma();
        let denominator = (sigma * sigma + 1.0).sqrt();
        sample.iter().map(|value| value / denominator).collect()
    }

    /// Initial noise must be scaled by the first (largest) sigma, unlike SD1.5's
    /// scheduler which consumes raw unit-variance noise.
    pub fn init_noise_sigma(&self) -> f32 {
        self.sigmas.first().copied().unwrap_or(1.0)
    }

    pub fn step(
        &mut self,
        model_output: &[f32],
        timestep: usize,
        sample: &[f32],
    ) -> Result<Vec<f32>> {
        if model_output.len() != sample.len() {
            bail!(
                "scheduler sample/model length mismatch: {} vs {}",
                sample.len(),
                model_output.len()
            );
        }
        let expected_timestep = self
            .timesteps
            .get(self.counter)
            .context("scheduler received more steps than configured")?;
        if timestep != *expected_timestep {
            bail!(
                "scheduler expected timestep {expected_timestep} at step {}, got {timestep}",
                self.counter
            );
        }
        let sigma = self.sigmas[self.counter];
        let sigma_next = self.sigmas[self.counter + 1];
        // Epsilon prediction: pred_original_sample = sample - sigma * model_output, and the
        // ODE derivative (sample - pred_original_sample) / sigma reduces to model_output.
        let dt = sigma_next - sigma;
        let output = sample
            .iter()
            .zip(model_output)
            .map(|(sample, derivative)| sample + derivative * dt)
            .collect();
        self.counter += 1;
        Ok(output)
    }
}

/// Linear interpolation of the sigma schedule at a fractional training index,
/// matching diffusers' `np.interp(timesteps, np.arange(...), sigmas)`.
fn interpolate_sigma(full_sigmas: &[f32], timestep: usize) -> f32 {
    if timestep == 0 {
        return full_sigmas[0];
    }
    if timestep >= full_sigmas.len() {
        return *full_sigmas.last().expect("non-empty sigma schedule");
    }
    full_sigmas[timestep]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_leading_spacing_with_offset_one() {
        let scheduler = SdxlScheduler::new(4).unwrap();
        // step_ratio = 250; base indices [0,250,500,750] + 1, reversed.
        assert_eq!(scheduler.timesteps(), &[751, 501, 251, 1]);
    }

    #[test]
    fn sigmas_are_descending_with_a_trailing_zero() {
        let scheduler = SdxlScheduler::new(10).unwrap();
        assert_eq!(scheduler.sigmas.len(), 11);
        assert_eq!(*scheduler.sigmas.last().unwrap(), 0.0);
        assert!(scheduler.sigmas.windows(2).all(|pair| pair[0] >= pair[1]));
    }

    #[test]
    fn scale_model_input_shrinks_toward_zero_as_sigma_grows() {
        let scheduler = SdxlScheduler::new(4).unwrap();
        let scaled = scheduler.scale_model_input(&[2.0, -4.0]);
        let sigma = scheduler.current_sigma();
        let expected = 2.0 / (sigma * sigma + 1.0).sqrt();
        assert!((scaled[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn step_moves_sample_by_derivative_times_dt() {
        let mut scheduler = SdxlScheduler::new(2).unwrap();
        let timesteps = scheduler.timesteps().to_vec();
        let sigma0 = scheduler.sigmas[0];
        let sigma1 = scheduler.sigmas[1];
        let sample = [1.0f32, -1.0];
        let model_output = [0.5f32, 0.5];
        let output = scheduler
            .step(&model_output, timesteps[0], &sample)
            .unwrap();
        let dt = sigma1 - sigma0;
        assert!((output[0] - (1.0 + 0.5 * dt)).abs() < 1e-6);
        assert!((output[1] - (-1.0 + 0.5 * dt)).abs() < 1e-6);
    }

    #[test]
    fn rejects_out_of_order_steps() {
        let mut scheduler = SdxlScheduler::new(4).unwrap();
        let error = scheduler.step(&[0.0], 501, &[0.0]).unwrap_err();
        assert!(error.to_string().contains("expected timestep 751"));
    }
}
