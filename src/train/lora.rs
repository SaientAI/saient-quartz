//! CPU reference LoRA math for a single frozen linear projection.
//! y = W*x + (alpha/r) * B * (A*x)
//! W lives in Quartz's existing quantized GGUF path and is NEVER updated here.

#[derive(Clone, Debug)]
pub struct LoraLinear {
    pub input: usize,
    pub output: usize,
    pub rank: usize,
    pub alpha: f32,
    /// Row-major [rank, input].
    pub a: Vec<f32>,
    /// Row-major [output, rank].
    pub b: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct LoraGrads {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl LoraLinear {
    pub fn new(input: usize, output: usize, rank: usize, alpha: f32, seed: u64) -> Self {
        assert!(input > 0 && output > 0 && rank > 0);
        assert!(alpha.is_finite() && alpha > 0.0);
        // Reproducible small nonzero A; exactly zero B, as in standard LoRA.
        let mut state = seed.max(1);
        let mut a = vec![0.0; rank * input];
        for value in &mut a {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            *value = (unit - 0.5) * 0.04;
        }
        Self {
            input,
            output,
            rank,
            alpha,
            a,
            b: vec![0.0; output * rank],
        }
    }

    pub fn zeros_like_grads(&self) -> LoraGrads {
        LoraGrads {
            a: vec![0.0; self.a.len()],
            b: vec![0.0; self.b.len()],
        }
    }

    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Add LoRA's contribution into an existing frozen-base output vector.
    pub fn forward_add(&self, x: &[f32], output: &mut [f32]) {
        assert_eq!(x.len(), self.input);
        assert_eq!(output.len(), self.output);
        let mut ax = vec![0.0; self.rank];
        for r in 0..self.rank {
            for i in 0..self.input {
                ax[r] += self.a[r * self.input + i] * x[i];
            }
        }
        let s = self.scale();
        for o in 0..self.output {
            for (r, ar) in ax.iter().enumerate() {
                output[o] += s * self.b[o * self.rank + r] * ar;
            }
        }
    }

    /// Accumulate adapter gradients and add adapter-only dX into the supplied
    /// buffer. Callers must separately compute and add frozen W^T * dY.
    /// This is deliberately NOT a complete transformer block backward.
    pub fn backward_add(&self, x: &[f32], dy: &[f32], dx: &mut [f32], grads: &mut LoraGrads) {
        assert_eq!(x.len(), self.input);
        assert_eq!(dy.len(), self.output);
        assert_eq!(dx.len(), self.input);
        assert_eq!(grads.a.len(), self.a.len());
        assert_eq!(grads.b.len(), self.b.len());
        let s = self.scale();
        let mut ax = vec![0.0; self.rank];
        let mut bt_dy = vec![0.0; self.rank];
        for r in 0..self.rank {
            for i in 0..self.input {
                ax[r] += self.a[r * self.input + i] * x[i];
            }
        }
        for o in 0..self.output {
            for r in 0..self.rank {
                grads.b[o * self.rank + r] += s * dy[o] * ax[r];
                bt_dy[r] += self.b[o * self.rank + r] * dy[o];
            }
        }
        for r in 0..self.rank {
            for i in 0..self.input {
                grads.a[r * self.input + i] += s * bt_dy[r] * x[i];
                dx[i] += s * self.a[r * self.input + i] * bt_dy[r];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(m: &LoraLinear, x: &[f32], dy: &[f32]) -> f32 {
        let mut y = vec![0.0; m.output];
        m.forward_add(x, &mut y);
        y.iter().zip(dy).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn zero_b_initialization_is_noop() {
        let model = LoraLinear::new(5, 3, 2, 4.0, 42);
        let mut y = vec![4.0, -2.0, 1.0];
        model.forward_add(&[1., 2., 3., 4., 5.], &mut y);
        assert_eq!(y, [4.0, -2.0, 1.0]);
    }

    #[test]
    fn adapter_gradient_and_input_gradient_match_finite_difference() {
        let mut model = LoraLinear::new(4, 3, 2, 4.0, 42);
        // Both A and B MUST be nonzero to catch dropped dX paths.
        model.a = vec![0.2, -0.5, 0.3, 0.7, 0.8, -0.2, 0.1, -0.3];
        model.b = vec![0.6, 0.4, -0.5, 0.7, 0.1, 0.2];
        let mut x = vec![0.4, -0.6, 0.9, 0.2];
        let dy = [0.3, -0.2, 0.5];
        let mut dx = vec![0.0; model.input];
        let mut grads = model.zeros_like_grads();
        model.backward_add(&x, &dy, &mut dx, &mut grads);
        let eps = 1e-3;
        for i in 0..model.a.len() {
            model.a[i] += eps;
            let hi = scalar(&model, &x, &dy);
            model.a[i] -= 2. * eps;
            let lo = scalar(&model, &x, &dy);
            model.a[i] += eps;
            let numeric = (hi - lo) / (2. * eps);
            assert!(
                (grads.a[i] - numeric).abs() < 3e-4,
                "dA[{i}]: {} vs {numeric}",
                grads.a[i]
            );
        }
        for i in 0..model.b.len() {
            model.b[i] += eps;
            let hi = scalar(&model, &x, &dy);
            model.b[i] -= 2. * eps;
            let lo = scalar(&model, &x, &dy);
            model.b[i] += eps;
            let numeric = (hi - lo) / (2. * eps);
            assert!(
                (grads.b[i] - numeric).abs() < 3e-4,
                "dB[{i}]: {} vs {numeric}",
                grads.b[i]
            );
        }
        for i in 0..x.len() {
            x[i] += eps;
            let hi = scalar(&model, &x, &dy);
            x[i] -= 2. * eps;
            let lo = scalar(&model, &x, &dy);
            x[i] += eps;
            let numeric = (hi - lo) / (2. * eps);
            assert!(
                (dx[i] - numeric).abs() < 3e-4,
                "dX[{i}]: {} vs {numeric}",
                dx[i]
            );
        }
    }

    #[test]
    fn gradients_accumulate_without_touching_frozen_base() {
        let mut lora = LoraLinear::new(2, 2, 1, 2.0, 77);
        lora.a = vec![0.5, -0.2];
        lora.b = vec![0.3, -0.1];
        let base = vec![0.9_f32, 0.2, -0.4, 0.8];
        let original = base.clone();
        let mut g = lora.zeros_like_grads();
        let mut dx = vec![0.0; 2];
        lora.backward_add(&[0.7, -0.5], &[0.2, 0.1], &mut dx, &mut g);
        lora.backward_add(&[0.7, -0.5], &[0.2, 0.1], &mut dx, &mut g);
        assert_eq!(base, original);
        assert!(g.a.iter().any(|v| v.abs() > 0.0));
        assert!(g.b.iter().any(|v| v.abs() > 0.0));
    }
}
