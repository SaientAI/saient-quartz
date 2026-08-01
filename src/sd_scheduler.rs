//! Stable Diffusion 1.x PNDM/PLMS schedule and classifier-free guidance.
//!
//! Quartz owns the denoising loop. This implements the exact SD1.5 scheduler
//! contract: scaled-linear betas, leading spacing with offset 1, skipped PRK
//! warmup, and the fourth-order pseudo-linear multistep update.

use anyhow::{Context, Result, bail};
use rand::{Rng, SeedableRng, rngs::SmallRng};

const TRAIN_TIMESTEPS: usize = 1_000;
const BETA_START: f64 = 0.00085;
const BETA_END: f64 = 0.012;

pub struct Sd15Scheduler {
    alphas_cumprod: Vec<f32>,
    timesteps: Vec<usize>,
    step_ratio: usize,
    counter: usize,
    ets: Vec<Vec<f32>>,
    current_sample: Option<Vec<f32>>,
}

impl Sd15Scheduler {
    pub fn new(inference_steps: usize) -> Result<Self> {
        if inference_steps == 0 || inference_steps > TRAIN_TIMESTEPS {
            bail!("SD1 inference steps must be between 1 and {TRAIN_TIMESTEPS}");
        }
        let step_ratio = TRAIN_TIMESTEPS / inference_steps;
        let base = (0..inference_steps)
            .map(|index| index * step_ratio + 1)
            .collect::<Vec<_>>();
        // skip_prk_steps=true duplicates the penultimate timestep for the
        // improved-Euler bootstrap used by PLMS.
        let mut timesteps = Vec::with_capacity(inference_steps + 1);
        timesteps.push(*base.last().expect("inference_steps is nonzero"));
        if inference_steps > 1 {
            timesteps.push(base[inference_steps - 2]);
            timesteps.push(base[inference_steps - 2]);
            timesteps.extend(base[..inference_steps - 2].iter().rev().copied());
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
        Ok(Self {
            alphas_cumprod,
            timesteps,
            step_ratio,
            counter: 0,
            ets: Vec::with_capacity(4),
            current_sample: None,
        })
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps
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
        if timestep >= TRAIN_TIMESTEPS {
            bail!("scheduler timestep {timestep} is outside the SD1 training schedule");
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

        let mut effective_timestep = timestep;
        let mut previous_timestep = timestep as isize - self.step_ratio as isize;
        if self.counter != 1 {
            if self.ets.len() > 3 {
                self.ets.remove(0);
            }
            self.ets.push(model_output.to_vec());
        } else {
            previous_timestep = timestep as isize;
            effective_timestep = timestep
                .checked_add(self.step_ratio)
                .context("PNDM bootstrap timestep overflow")?;
        }

        let bootstrap_sample = if self.counter == 1 {
            Some(
                self.current_sample
                    .take()
                    .context("PNDM bootstrap sample is missing")?,
            )
        } else {
            None
        };
        let combined_output = match (self.ets.len(), self.counter) {
            (1, 0) => {
                self.current_sample = Some(sample.to_vec());
                model_output.to_vec()
            }
            (1, 1) => model_output
                .iter()
                .zip(&self.ets[0])
                .map(|(current, previous)| (current + previous) * 0.5)
                .collect(),
            (2, _) => combine_outputs(&self.ets, &[3.0 / 2.0, -1.0 / 2.0])?,
            (3, _) => combine_outputs(&self.ets, &[23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0])?,
            (4, _) => combine_outputs(
                &self.ets,
                &[55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0],
            )?,
            (count, _) => bail!("invalid PNDM history length {count}"),
        };
        let effective_sample = bootstrap_sample.as_deref().unwrap_or(sample);

        let output = previous_sample(
            &self.alphas_cumprod,
            effective_sample,
            effective_timestep,
            previous_timestep,
            &combined_output,
        )?;
        self.counter += 1;
        Ok(output)
    }
}

fn combine_outputs(history: &[Vec<f32>], coefficients: &[f32]) -> Result<Vec<f32>> {
    if history.len() != coefficients.len() {
        bail!("PNDM coefficient/history mismatch");
    }
    let width = history.first().context("PNDM history is empty")?.len();
    if history.iter().any(|entry| entry.len() != width) {
        bail!("PNDM history tensors have inconsistent lengths");
    }
    let mut output = vec![0.0; width];
    // Coefficients are newest-to-oldest, while history is oldest-to-newest.
    for (entry, coefficient) in history.iter().rev().zip(coefficients) {
        for (output, value) in output.iter_mut().zip(entry) {
            *output += coefficient * value;
        }
    }
    Ok(output)
}

fn previous_sample(
    alphas_cumprod: &[f32],
    sample: &[f32],
    timestep: usize,
    previous_timestep: isize,
    model_output: &[f32],
) -> Result<Vec<f32>> {
    if sample.len() != model_output.len() {
        bail!("PNDM sample/model length mismatch");
    }
    let alpha = *alphas_cumprod
        .get(timestep)
        .context("PNDM timestep is outside the alpha schedule")?;
    // set_alpha_to_one=false uses alpha[0] after the final step.
    let previous_alpha = if previous_timestep >= 0 {
        alphas_cumprod[previous_timestep as usize]
    } else {
        alphas_cumprod[0]
    };
    let beta = 1.0 - alpha;
    let previous_beta = 1.0 - previous_alpha;
    let sample_coefficient = (previous_alpha / alpha).sqrt();
    let model_denominator = alpha * previous_beta.sqrt() + (alpha * beta * previous_alpha).sqrt();
    Ok(sample
        .iter()
        .zip(model_output)
        .map(|(sample, model)| {
            sample_coefficient * sample - (previous_alpha - alpha) * model / model_denominator
        })
        .collect())
}

pub fn classifier_free_guidance(
    unconditional: &[f32],
    conditional: &[f32],
    scale: f32,
) -> Result<Vec<f32>> {
    if unconditional.len() != conditional.len() {
        bail!(
            "guidance input length mismatch: {} vs {}",
            unconditional.len(),
            conditional.len()
        );
    }
    if !scale.is_finite() || scale < 0.0 {
        bail!("guidance scale must be finite and non-negative");
    }
    Ok(unconditional
        .iter()
        .zip(conditional)
        .map(|(&unconditional, &conditional)| unconditional + scale * (conditional - unconditional))
        .collect())
}

/// Stable, seedable Gaussian noise using the Box-Muller transform.
pub fn gaussian_noise(seed: u64, count: usize) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut output = Vec::with_capacity(count);
    while output.len() < count {
        let first = (1.0f64 - rng.r#gen::<f64>()).max(f64::MIN_POSITIVE);
        let second = rng.r#gen::<f64>();
        let radius = (-2.0 * first.ln()).sqrt();
        let angle = std::f64::consts::TAU * second;
        output.push((radius * angle.cos()) as f32);
        if output.len() < count {
            output.push((radius * angle.sin()) as f32);
        }
    }
    output
}

/// Sinusoidal timestep embedding used by SD1.5's UNet.
pub fn timestep_embedding(timestep: f32, dimension: usize) -> Result<Vec<f32>> {
    if dimension == 0 || dimension % 2 != 0 {
        bail!("timestep embedding dimension must be a non-zero even number");
    }
    if !timestep.is_finite() {
        bail!("timestep must be finite");
    }
    let half = dimension / 2;
    let mut cosines = Vec::with_capacity(half);
    let mut sines = Vec::with_capacity(half);
    for index in 0..half {
        let exponent = -(10_000.0f32).ln() * index as f32 / half as f32;
        let value = timestep * exponent.exp();
        cosines.push(value.cos());
        sines.push(value.sin());
    }
    // SD1.5 config: flip_sin_to_cos=true.
    cosines.extend(sines);
    Ok(cosines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_the_sd15_leading_timestep_spacing() {
        let scheduler = Sd15Scheduler::new(20).unwrap();
        assert_eq!(scheduler.timesteps().len(), 21);
        assert_eq!(scheduler.timesteps()[0], 951);
        assert_eq!(&scheduler.timesteps()[1..3], &[901, 901]);
        assert_eq!(scheduler.timesteps()[20], 1);
        assert!(
            scheduler
                .alphas_cumprod
                .windows(2)
                .all(|pair| pair[1] < pair[0])
        );
    }

    #[test]
    fn pndm_matches_the_official_golden_sequence() {
        let mut scheduler = Sd15Scheduler::new(5).unwrap();
        assert_eq!(scheduler.timesteps(), &[801, 601, 601, 401, 201, 1]);
        let expected = [
            [0.40919268, -1.5115727],
            [0.35239375, -1.4831733],
            [0.28068942, -2.2663603],
            [0.16392766, -2.9173186],
            [-0.12385945, -3.2046015],
            [-0.13180457, -3.2020257],
        ];
        let timesteps = scheduler.timesteps().to_vec();
        let mut sample = vec![0.25f32, -0.75];
        for (index, timestep) in timesteps.into_iter().enumerate() {
            let model = [0.1 * (index + 1) as f32, -0.05 * (index + 1) as f32];
            sample = scheduler.step(&model, timestep, &sample).unwrap();
            for element in 0..2 {
                assert!(
                    (sample[element] - expected[index][element]).abs() < 2e-6,
                    "step {index} element {element}: {} != {}",
                    sample[element],
                    expected[index][element]
                );
            }
        }
    }

    #[test]
    fn rejects_out_of_order_steps() {
        let mut scheduler = Sd15Scheduler::new(5).unwrap();
        let error = scheduler.step(&[0.0], 601, &[0.0]).unwrap_err();
        assert!(error.to_string().contains("expected timestep 801"));
    }

    #[test]
    fn classifier_free_guidance_combines_predictions() {
        assert_eq!(
            classifier_free_guidance(&[1.0, -2.0], &[3.0, 2.0], 1.5).unwrap(),
            vec![4.0, 4.0]
        );
        assert!(classifier_free_guidance(&[1.0], &[1.0, 2.0], 7.5).is_err());
    }

    #[test]
    fn gaussian_noise_is_reproducible_and_finite() {
        let first = gaussian_noise(42, 17);
        let second = gaussian_noise(42, 17);
        assert_eq!(first, second);
        assert!(first.iter().all(|value| value.is_finite()));
        assert_ne!(first, gaussian_noise(43, 17));
    }

    #[test]
    fn zero_timestep_embedding_is_cosines_then_sines() {
        assert_eq!(
            timestep_embedding(0.0, 6).unwrap(),
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        );
        assert!(timestep_embedding(1.0, 7).is_err());
    }
}
