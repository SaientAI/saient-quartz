//! Streaming W^T * dY for several tokens, dequantizing each GGUF row
//! once per batch. No mutable or trainable base-model tensors.

use crate::dequant;
use crate::gguf::{GgufFile, TensorInfo, ggml_type_size};

/// Matrix W has GGUF row-major dimensions [output, input].
/// dy[token][output] -> dx[token][input].
pub fn gguf_transpose_batch(
    gguf: &GgufFile,
    ti: &TensorInfo,
    dy: &[Vec<f32>],
    input: usize,
    output: usize,
) -> Vec<Vec<f32>> {
    assert!(!dy.is_empty());
    assert_eq!(ti.dims.len(), 2);
    assert_eq!(ti.dims[0] as usize, input);
    assert_eq!(ti.dims[1] as usize, output);
    for row in dy {
        assert_eq!(row.len(), output);
    }
    let data = gguf.tensor_data(ti);
    let bytes_per_row = ggml_type_size(ti.ggml_type, input);
    assert_eq!(data.len(), output * bytes_per_row);
    let mut dx = vec![vec![0.0f32; input]; dy.len()];
    let mut scratch = Vec::<f32>::new();
    const TILE: usize = 32;
    for first in (0..output).step_by(TILE) {
        let last = (first + TILE).min(output);
        if dy.iter().all(|g| g[first..last].iter().all(|x| *x == 0.0)) {
            continue;
        }
        dequant::dequant_into(
            &data[first * bytes_per_row..last * bytes_per_row],
            ti.ggml_type,
            (last - first) * input,
            &mut scratch,
        );
        for j in first..last {
            let wrow = &scratch[(j - first) * input..(j - first + 1) * input];
            for (t, gy) in dy.iter().enumerate() {
                let d = gy[j];
                if d == 0.0 {
                    continue;
                }
                for i in 0..input {
                    dx[t][i] += d * wrow[i];
                }
            }
        }
    }
    dx
}

pub fn dense_transpose_batch(
    w: &[f32],
    dy: &[Vec<f32>],
    input: usize,
    output: usize,
) -> Vec<Vec<f32>> {
    assert_eq!(w.len(), input * output);
    let mut dx = vec![vec![0.0; input]; dy.len()];
    for (t, v) in dy.iter().enumerate() {
        assert_eq!(v.len(), output);
        for j in 0..output {
            for i in 0..input {
                dx[t][i] += w[j * input + i] * v[j];
            }
        }
    }
    dx
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_batch_transpose_numeric_gradient() {
        let input = 5;
        let output = 4;
        let w: Vec<f32> = (0..input * output)
            .map(|i| ((i as f32) * 0.41).sin())
            .collect();
        let dy = vec![vec![0.2, -0.3, 0.5, 1.1], vec![0.4, 0.5, -1.0, 0.3]];
        let g = dense_transpose_batch(&w, &dy, input, output);
        let mut x = vec![0.1, 0.3, -0.2, 0.9, -0.4];
        for t in 0..dy.len() {
            for i in 0..input {
                let f = |x: &[f32]| -> f32 {
                    let mut acc = 0.0;
                    for j in 0..output {
                        let y = (0..input).map(|k| w[j * input + k] * x[k]).sum::<f32>();
                        acc += y * dy[t][j];
                    }
                    acc
                };
                x[i] += 1e-3;
                let hi = f(&x);
                x[i] -= 2e-3;
                let lo = f(&x);
                x[i] += 1e-3;
                assert!((g[t][i] - (hi - lo) / 2e-3).abs() < 1e-4);
            }
        }
    }
}
