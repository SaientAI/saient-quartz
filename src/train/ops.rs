//! CPU reference backward operators for the last dense block.
//! Frozen quantized GGUF weights remain unchanged. Gradients use f32.

use crate::dequant;
use crate::gguf::{GgufFile, TensorInfo, ggml_type_size};

pub fn rms_bwd(x: &[f32], w: &[f32], dy: &[f32], eps: f32) -> Vec<f32> {
    assert!(!x.is_empty() && x.len() == w.len() && x.len() == dy.len());
    let n = x.len() as f32;
    let inv = (x.iter().map(|v| v * v).sum::<f32>() / n + eps)
        .sqrt()
        .recip();
    let dot = x
        .iter()
        .zip(w)
        .zip(dy)
        .map(|((&x, &w), &dy)| x * w * dy)
        .sum::<f32>();
    (0..x.len())
        .map(|i| w[i] * dy[i] * inv - x[i] * (inv * inv * inv / n) * dot)
        .collect()
}
pub fn qk_norm_bwd(
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    heads: usize,
    hd: usize,
    eps: f32,
) -> Vec<f32> {
    assert!(heads > 0 && hd > 0 && x.len() == heads * hd && dy.len() == x.len());
    if w.is_empty() {
        return dy.to_vec();
    }
    if w.len() == hd {
        let mut dx = vec![0.; x.len()];
        for h in 0..heads {
            let r = h * hd..(h + 1) * hd;
            dx[r.clone()].copy_from_slice(&rms_bwd(&x[r.clone()], w, &dy[r], eps));
        }
        dx
    } else {
        assert_eq!(w.len(), x.len());
        rms_bwd(x, w, dy, eps)
    }
}
/// Inverse transpose of the same rotation as the production rope_yarn forward.
pub fn rope_bwd(g: &mut [f32], pos: usize, dim: usize, theta: f32, ys: f32, yo: usize, neox: bool) {
    use std::f32::consts::PI;
    assert_eq!(g.len(), dim);
    assert_eq!(dim % 2, 0);
    let half = dim / 2;
    let hfw = yo as f32 / 32.0;
    let lfw = yo as f32;
    for i in 0..half {
        let inv = 1.0 / theta.powf(2.0 * i as f32 / dim as f32);
        let effective = if ys <= 1.0 || yo == 0 {
            inv
        } else {
            let wl = 2.0 * PI / inv;
            let ramp = ((lfw - wl) / (lfw - hfw)).clamp(0.0, 1.0);
            ramp * inv + (1.0 - ramp) * (inv / ys)
        };
        let (sin, cos) = (pos as f32 * effective).sin_cos();
        let (a, b) = if neox {
            (i, i + half)
        } else {
            (2 * i, 2 * i + 1)
        };
        let x = g[a];
        let y = g[b];
        g[a] = x * cos + y * sin;
        g[b] = -x * sin + y * cos;
    }
}
pub fn silu_derivative(x: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    sigmoid * (1.0 + x * (1.0 - sigmoid))
}
#[derive(Clone)]
pub struct AttentionResult {
    pub out: Vec<f32>,
    pub probs: Vec<Vec<f32>>,
}
pub fn attention_last(
    q: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    heads: usize,
    kvheads: usize,
    hd: usize,
    start: usize,
    scale: f32,
    sinks: &[f32],
) -> AttentionResult {
    assert!(!keys.is_empty() && start < keys.len() && keys.len() == values.len());
    assert!(heads >= kvheads && kvheads > 0 && hd > 0 && q.len() == heads * hd);
    for t in 0..keys.len() {
        assert_eq!(keys[t].len(), kvheads * hd);
        assert_eq!(values[t].len(), kvheads * hd);
    }
    let mut out = vec![0.; heads * hd];
    let mut probs = Vec::new();
    for h in 0..heads {
        let kv = h * kvheads / heads;
        let qh = &q[h * hd..(h + 1) * hd];
        let mut ss: Vec<f32> = (start..keys.len())
            .map(|t| {
                qh.iter()
                    .zip(&keys[t][kv * hd..(kv + 1) * hd])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale
            })
            .collect();
        let sink = sinks.get(h).copied().unwrap_or(f32::NEG_INFINITY);
        let m = ss
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .max(sink);
        let mut z = 0.;
        for s in &mut ss {
            *s = (*s - m).exp();
            z += *s;
        }
        if sink.is_finite() {
            z += (sink - m).exp();
        }
        for s in &mut ss {
            *s /= z;
        }
        for (j, &p) in ss.iter().enumerate() {
            for i in 0..hd {
                out[h * hd + i] += p * values[start + j][kv * hd + i];
            }
        }
        probs.push(ss);
    }
    AttentionResult { out, probs }
}
pub struct AttentionGradient {
    pub dq: Vec<f32>,
    pub dk: Vec<Vec<f32>>,
    pub dv: Vec<Vec<f32>>,
}
pub fn attention_last_bwd(
    q: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    cache: &AttentionResult,
    dy: &[f32],
    heads: usize,
    kvheads: usize,
    hd: usize,
    start: usize,
    scale: f32,
) -> AttentionGradient {
    assert_eq!(q.len(), dy.len());
    assert_eq!(cache.probs.len(), heads);
    let mut dq = vec![0.; heads * hd];
    let mut dk = vec![vec![0.; kvheads * hd]; keys.len()];
    let mut dv = vec![vec![0.; kvheads * hd]; keys.len()];
    for h in 0..heads {
        let kv = h * kvheads / heads;
        let qh = &q[h * hd..(h + 1) * hd];
        let gh = &dy[h * hd..(h + 1) * hd];
        let prob = &cache.probs[h];
        let mut dprob = vec![0.; prob.len()];
        let mut avg = 0.0;
        for (j, &p) in prob.iter().enumerate() {
            let t = start + j;
            let dp = gh
                .iter()
                .zip(&values[t][kv * hd..(kv + 1) * hd])
                .map(|(a, b)| a * b)
                .sum::<f32>();
            dprob[j] = dp;
            avg += p * dp;
            for i in 0..hd {
                dv[t][kv * hd + i] += p * gh[i];
            }
        }
        // A sink has no V contribution. It remains in the softmax
        // denominator and therefore contributes zero to the weighted mean.
        for (j, &p) in prob.iter().enumerate() {
            let t = start + j;
            let ds = p * (dprob[j] - avg) * scale;
            for i in 0..hd {
                dq[h * hd + i] += ds * keys[t][kv * hd + i];
                dk[t][kv * hd + i] += ds * qh[i];
            }
        }
    }
    AttentionGradient { dq, dk, dv }
}
/// Transpose GEMV dX=W^T dY using only 32 dequantized rows of frozen W at once.
pub fn gguf_transpose(
    gguf: &GgufFile,
    ti: &TensorInfo,
    dy: &[f32],
    input: usize,
    output: usize,
) -> Vec<f32> {
    assert_eq!(ti.dims.len(), 2);
    assert_eq!(ti.dims[0] as usize, input);
    assert_eq!(ti.dims[1] as usize, output);
    assert_eq!(dy.len(), output);
    const TILE: usize = 32;
    let rowbytes = ggml_type_size(ti.ggml_type, input);
    let data = gguf.tensor_data(ti);
    assert_eq!(data.len(), rowbytes * output);
    let mut dx = vec![0.; input];
    let mut scratch = Vec::new();
    for st in (0..output).step_by(TILE) {
        let en = (st + TILE).min(output);
        if dy[st..en].iter().all(|v| *v == 0.) {
            continue;
        }
        dequant::dequant_into(
            &data[st * rowbytes..en * rowbytes],
            ti.ggml_type,
            (en - st) * input,
            &mut scratch,
        );
        for j in st..en {
            if dy[j] == 0.0 {
                continue;
            }
            let row = &scratch[(j - st) * input..(j - st + 1) * input];
            for i in 0..input {
                dx[i] += dy[j] * row[i];
            }
        }
    }
    dx
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{apply_qk_norm, rms_norm, rope_yarn};
    fn fd<F: Fn(&[f32]) -> f32>(v: &mut [f32], i: usize, f: F) -> f32 {
        let e = 0.001;
        v[i] += e;
        let hi = f(v);
        v[i] -= 2. * e;
        let lo = f(v);
        v[i] += e;
        (hi - lo) / (2. * e)
    }
    fn check(a: f32, n: f32, tol: f32, label: &str) {
        assert!(
            (a - n).abs() < tol,
            "{label}: analytic={a:.7} numeric={n:.7}"
        );
    }
    #[test]
    fn rmsnorm_grad_numerical() {
        let mut x = vec![0.3, 0.7, -0.2, 1.1, -0.9, 0.4];
        let w = [0.7, 1.1, 0.9, 1.4, 0.3, 0.5];
        let dy = [0.25, -0.3, 0.6, 0.7, -0.2, 0.1];
        let g = rms_bwd(&x, &w, &dy, 1e-5);
        for i in 0..x.len() {
            let n = fd(&mut x, i, |v| {
                rms_norm(v, &w, 1e-5)
                    .iter()
                    .zip(dy)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            });
            check(g[i], n, 0.00025, "rms");
        }
    }
    #[test]
    fn qk_norm_grad_perhead_full_and_none() {
        let mut x = vec![0.7, -0.3, 0.4, 0.9, 0.2, -0.1, 1.1, -0.5];
        let dy = [0.5, 0.3, -0.4, 0.7, 0.1, -0.2, 0.3, 0.4];
        for w in [vec![1.1, 0.9, 1.2, 0.8], vec![0.9; 8], vec![]] {
            let g = qk_norm_bwd(&x, &w, &dy, 2, 4, 1e-5);
            for i in 0..x.len() {
                let n = fd(&mut x, i, |v| {
                    let mut z = v.to_vec();
                    apply_qk_norm(&mut z, &w, 2, 4, 1e-5);
                    z.iter().zip(&dy).map(|(a, b)| a * b).sum::<f32>()
                });
                check(g[i], n, 0.0003, "qk");
            }
        }
    }
    #[test]
    fn rope_grad_neox_pair_and_yarn() {
        let mut x = vec![0.2, -0.3, 0.7, 0.9, 0.1, -0.8, 0.5, 1.0];
        let dy = [0.2, -0.7, 0.9, 0.3, 0.5, 0.6, -0.2, -0.1];
        for neox in [false, true] {
            for (ys, yo) in [(1.0, 0), (8.0, 2048)] {
                let mut g = dy;
                rope_bwd(&mut g, 73, 8, 1e6, ys, yo, neox);
                for i in 0..x.len() {
                    let n = fd(&mut x, i, |v| {
                        let mut z = v.to_vec();
                        rope_yarn(&mut z, 73, 8, 1e6, ys, yo, neox);
                        z.iter().zip(dy).map(|(a, b)| a * b).sum::<f32>()
                    });
                    check(g[i], n, 0.0003, "rope");
                }
            }
        }
    }
    #[test]
    fn silu_grad_numerical() {
        let mut x = vec![-2., -0.3, 0.6, 2.];
        for i in 0..x.len() {
            let n = fd(&mut x, i, |v| v[i] / (1. + (-v[i]).exp()));
            check(silu_derivative(x[i]), n, 0.00015, "silu");
        }
    }
    fn loss(
        q: &[f32],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        dy: &[f32],
        start: usize,
        sinks: &[f32],
    ) -> f32 {
        attention_last(q, k, v, 4, 2, 2, start, 0.7, sinks)
            .out
            .iter()
            .zip(dy)
            .map(|(a, b)| a * b)
            .sum()
    }
    #[test]
    fn gqa_attention_grad_qkv_sinks_and_causal_window() {
        let mut q = vec![0.25, -0.3, 0.8, 0.6, -0.7, 0.4, 0.6, -0.2];
        let mut k = vec![
            vec![0.2, 0.6, -0.1, -0.3],
            vec![-0.6, 0.3, 0.2, 0.5],
            vec![0.1, -0.5, 0.4, 0.8],
        ];
        let mut v = vec![
            vec![0.4, 0.8, -0.7, 0.3],
            vec![-0.1, 0.5, 0.3, 0.6],
            vec![0.9, -0.2, 0.1, -0.3],
        ];
        let dy = [0.3, -0.6, 0.4, 0.7, -0.5, 0.2, 0.1, -0.3];
        for (start, sinks) in [(0, vec![]), (0, vec![-0.9, 0.7, 0.5, -0.3]), (1, vec![])] {
            let cache = attention_last(&q, &k, &v, 4, 2, 2, start, 0.7, &sinks);
            let g = attention_last_bwd(&q, &k, &v, &cache, &dy, 4, 2, 2, start, 0.7);
            for i in 0..q.len() {
                let n = fd(&mut q, i, |z| loss(z, &k, &v, &dy, start, &sinks));
                check(g.dq[i], n, 0.00025, "dQ");
            }
            for t in 0..k.len() {
                for i in 0..k[t].len() {
                    let old = k[t][i];
                    k[t][i] = old + 0.001;
                    let hi = loss(&q, &k, &v, &dy, start, &sinks);
                    k[t][i] = old - 0.001;
                    let lo = loss(&q, &k, &v, &dy, start, &sinks);
                    k[t][i] = old;
                    check(g.dk[t][i], (hi - lo) / 0.002, 0.00025, "dK");
                    let old = v[t][i];
                    v[t][i] = old + 0.001;
                    let hi = loss(&q, &k, &v, &dy, start, &sinks);
                    v[t][i] = old - 0.001;
                    let lo = loss(&q, &k, &v, &dy, start, &sinks);
                    v[t][i] = old;
                    check(g.dv[t][i], (hi - lo) / 0.002, 0.00025, "dV");
                }
            }
        }
    }
}
