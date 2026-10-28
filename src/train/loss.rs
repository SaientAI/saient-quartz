//! Vocab-chunked two-pass cross-entropy for ONE target token.
//! A future Quartz GGUF projection adapter will supply the callbacks.
//! No full vocabulary-wide logits or gradient buffer is allocated.

/// project(start, hidden, out) computes logits for out.len() vocabulary rows.
/// transpose_accumulate(start, d_logits, d_hidden) adds W_chunk^T*d_logits.
pub fn chunked_cross_entropy<P, B>(
    hidden: &[f32],
    vocab: usize,
    target: usize,
    chunk: usize,
    mut project: P,
    mut transpose_accumulate: B,
) -> (f32, Vec<f32>)
where
    P: FnMut(usize, &[f32], &mut [f32]),
    B: FnMut(usize, &[f32], &mut [f32]),
{
    assert!(!hidden.is_empty());
    assert!(vocab > 0 && target < vocab && chunk > 0);
    let mut scores = vec![0.0_f32; chunk.min(vocab)];
    let mut running_max = f64::NEG_INFINITY;
    let mut exp_sum = 0.0_f64;
    let mut target_logit = None;

    // Pass 1: stable global logsumexp and target score.
    for start in (0..vocab).step_by(chunk) {
        let n = chunk.min(vocab - start);
        let out = &mut scores[..n];
        project(start, hidden, out);
        for (j, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "non-finite logit");
            let value = v as f64;
            if value > running_max {
                exp_sum = exp_sum * (running_max - value).exp() + 1.0;
                running_max = value;
            } else {
                exp_sum += (value - running_max).exp();
            }
            if start + j == target {
                target_logit = Some(value);
            }
        }
    }
    let log_z = running_max + exp_sum.ln();
    let loss = (log_z - target_logit.expect("target score not found")) as f32;
    let mut dh = vec![0.0_f32; hidden.len()];

    // Pass 2: recompute only the active vocab chunk.
    for start in (0..vocab).step_by(chunk) {
        let n = chunk.min(vocab - start);
        let out = &mut scores[..n];
        project(start, hidden, out);
        for (j, v) in out.iter_mut().enumerate() {
            let probability = ((*v as f64 - log_z).exp()) as f32;
            *v = probability - if start + j == target { 1.0 } else { 0.0 };
        }
        transpose_accumulate(start, out, &mut dh);
    }
    (loss, dh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        weights: &[f32],
        hidden: &[f32],
        vocab: usize,
        chunk: usize,
        target: usize,
    ) -> (f32, Vec<f32>) {
        let d = hidden.len();
        chunked_cross_entropy(
            hidden,
            vocab,
            target,
            chunk,
            |start, x, scores| {
                for (j, out) in scores.iter_mut().enumerate() {
                    *out = (0..d).map(|i| weights[(start + j) * d + i] * x[i]).sum();
                }
            },
            |start, g, dx| {
                for (j, &dj) in g.iter().enumerate() {
                    for i in 0..d {
                        dx[i] += weights[(start + j) * d + i] * dj;
                    }
                }
            },
        )
    }

    #[test]
    fn chunked_matches_dense_cross_entropy_and_gradient() {
        let vocab = 9;
        let d = 3;
        let w: Vec<f32> = (0..vocab * d)
            .map(|i| ((i as f32 * 1.31).sin() + (i as f32 * 0.28).cos()) * 0.7)
            .collect();
        let hidden = [0.25, -1.0, 0.4];
        let target = 6;
        let full_scores: Vec<f32> = (0..vocab)
            .map(|j| (0..d).map(|i| w[j * d + i] * hidden[i]).sum())
            .collect();
        let z: f32 = full_scores.iter().map(|s| s.exp()).sum();
        let expected_loss = z.ln() - full_scores[target];
        let mut expected_dx = vec![0.0; d];
        for (j, &score) in full_scores.iter().enumerate() {
            let p = score.exp() / z - if j == target { 1.0 } else { 0.0 };
            for i in 0..d {
                expected_dx[i] += p * w[j * d + i];
            }
        }
        for chunk in [1, 2, 4, 9, 20] {
            let (l, g) = run(&w, &hidden, vocab, chunk, target);
            assert!((l - expected_loss).abs() < 1e-5, "{chunk} loss");
            for i in 0..d {
                assert!((g[i] - expected_dx[i]).abs() < 1e-5, "{chunk} dH[{i}]");
            }
        }
    }

    #[test]
    fn stable_with_very_large_logits() {
        let w = [1000.0, -950.0, 1050.0, 20.0, -1000.0, 1030.0];
        let (loss, dh) = run(&w, &[1.0], 6, 2, 2);
        assert!(loss.is_finite() && loss > 0.0);
        assert!(dh.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn loss_gradient_matches_input_finite_difference() {
        let vocab = 6;
        let d = 4;
        let weights: Vec<f32> = (0..vocab * d).map(|k| (k as f32 * 0.17).sin()).collect();
        let mut h = [0.3_f32, -0.6, 0.4, 0.9];
        let (_, dh) = run(&weights, &h, vocab, 2, 3);
        let eps = 1e-3;
        for i in 0..d {
            h[i] += eps;
            let a = run(&weights, &h, vocab, 2, 3).0;
            h[i] -= 2. * eps;
            let b = run(&weights, &h, vocab, 2, 3).0;
            h[i] += eps;
            let numerical = (a - b) / (2. * eps);
            assert!(
                (dh[i] - numerical).abs() < 3e-4,
                "{i}: {} vs {numerical}",
                dh[i]
            );
        }
    }
}
