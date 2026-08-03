//! 3D rotary position embedding for the Wan2.1 diffusion transformer.
//!
//! Wan's latents are a video grid, so position is three-dimensional. The head dimension is split
//! across the axes — for the 1.3B model `head_dim = 128` splits as `44 + 42 + 42` over
//! (temporal, height, width) — and each axis gets its own independent rotary embedding over its
//! slice. Sum the splits and you get the head dim back; get the split order wrong and the model
//! still runs, it just places every patch in the wrong place.
//!
//! Output layout matches the reference exactly: for each of `pos_len` positions, 64 pairs, each
//! stored as the four entries of a 2x2 rotation `[cos, -sin, sin, cos]`.

/// Rotation entries per frequency pair: a 2x2 matrix.
pub const PAIR_STRIDE: usize = 4;

/// Positions of every patch in the latent video grid, in the order the DiT flattens them.
///
/// `t/h/w` are latent dimensions and `pt/ph/pw` the patch size; the `+ p/2` matches the
/// reference's rounding so an odd dimension pads rather than truncates.
pub fn video_ids(t: usize, h: usize, w: usize, pt: usize, ph: usize, pw: usize) -> Vec<[f32; 3]> {
    let (t_len, h_len, w_len) = ((t + pt / 2) / pt, (h + ph / 2) / ph, (w + pw / 2) / pw);
    let mut ids = Vec::with_capacity(t_len * h_len * w_len);
    for i in 0..t_len {
        for j in 0..h_len {
            for k in 0..w_len {
                ids.push([i as f32, j as f32, k as f32]);
            }
        }
    }
    ids
}

/// Rotary embedding for one axis: `[pos_len, (dim/2) * 4]`.
fn rope_axis(pos: &[f32], dim: usize, theta: f32) -> Vec<f32> {
    debug_assert_eq!(dim % 2, 0, "rope dim must be even");
    let half = dim / 2;
    let mut out = vec![0.0f32; pos.len() * half * PAIR_STRIDE];
    for (i, &p) in pos.iter().enumerate() {
        for j in 0..half {
            // scale = linspace(0, (dim-2)/dim, half) evaluates to exactly 2j/dim.
            let omega = theta.powf(-(2.0 * j as f32) / dim as f32);
            let (sin, cos) = (p * omega).sin_cos();
            let o = (i * half + j) * PAIR_STRIDE;
            out[o] = cos;
            out[o + 1] = -sin;
            out[o + 2] = sin;
            out[o + 3] = cos;
        }
    }
    out
}

/// Concatenate per-axis rotary embeddings into the DiT's `pe` tensor.
///
/// Returns `[pos_len * (sum(axes_dim)/2) * 4]`, axes laid out back to back per position.
pub fn embed_nd(ids: &[[f32; 3]], axes_dim: &[usize; 3], theta: f32) -> Vec<f32> {
    let emb_pairs: usize = axes_dim.iter().map(|d| d / 2).sum();
    let per_pos = emb_pairs * PAIR_STRIDE;
    let mut out = vec![0.0f32; ids.len() * per_pos];

    let mut offset = 0usize;
    for (axis, &dim) in axes_dim.iter().enumerate() {
        let pos: Vec<f32> = ids.iter().map(|p| p[axis]).collect();
        let emb = rope_axis(&pos, dim, theta);
        let axis_span = (dim / 2) * PAIR_STRIDE;
        for i in 0..ids.len() {
            out[i * per_pos + offset..i * per_pos + offset + axis_span]
                .copy_from_slice(&emb[i * axis_span..(i + 1) * axis_span]);
        }
        offset += axis_span;
    }
    out
}

/// Full `pe` for a Wan latent grid.
pub fn wan_pe(
    t: usize, h: usize, w: usize,
    pt: usize, ph: usize, pw: usize,
    axes_dim: &[usize; 3],
    theta: f32,
) -> Vec<f32> {
    embed_nd(&video_ids(t, h, w, pt, ph, pw), axes_dim, theta)
}

/// Apply the rotation to one head's vector in place.
///
/// `pe_pos` is this position's slice of the `pe` tensor: `head_dim/2` consecutive 2x2 matrices.
pub fn apply_rope(vec_head: &mut [f32], pe_pos: &[f32]) {
    let pairs = vec_head.len() / 2;
    debug_assert!(pe_pos.len() >= pairs * PAIR_STRIDE);
    for j in 0..pairs {
        let (a, b) = (vec_head[2 * j], vec_head[2 * j + 1]);
        let m = &pe_pos[j * PAIR_STRIDE..j * PAIR_STRIDE + PAIR_STRIDE];
        vec_head[2 * j] = m[0] * a + m[1] * b;
        vec_head[2 * j + 1] = m[2] * a + m[3] * b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const AXES: [usize; 3] = [44, 42, 42];

    #[test]
    fn grid_size_matches_the_reference_run() {
        // 5 frames at 416x240 -> latent 2x30x52, patched by (1,2,2) -> 2*15*26 = 780 tokens.
        let ids = video_ids(2, 30, 52, 1, 2, 2);
        assert_eq!(ids.len(), 780, "DiT token count");
        assert_eq!(ids[0], [0.0, 0.0, 0.0]);
        assert_eq!(ids[1], [0.0, 0.0, 1.0], "width varies fastest");
        assert_eq!(ids[26], [0.0, 1.0, 0.0], "then height");
        assert_eq!(ids[390], [1.0, 0.0, 0.0], "then time");
    }

    #[test]
    fn axes_split_sums_to_head_dim() {
        assert_eq!(AXES.iter().sum::<usize>(), 128, "44+42+42 must equal head_dim");
    }

    #[test]
    fn each_entry_is_a_rotation_matrix() {
        let pe = wan_pe(2, 30, 52, 1, 2, 2, &AXES, 10000.0);
        let pairs: usize = AXES.iter().map(|d| d / 2).sum();
        assert_eq!(pe.len(), 780 * pairs * PAIR_STRIDE);
        for p in [0usize, 1, 400, 779] {
            for j in 0..pairs {
                let m = &pe[(p * pairs + j) * PAIR_STRIDE..(p * pairs + j) * PAIR_STRIDE + 4];
                assert!((m[0] - m[3]).abs() < 1e-6, "cos entries must match");
                assert!((m[1] + m[2]).abs() < 1e-6, "sin entries must be negatives");
                assert!((m[0] * m[0] + m[2] * m[2] - 1.0).abs() < 1e-5, "must be unit");
            }
        }
    }

    #[test]
    fn position_zero_is_the_identity() {
        // pos 0 gives angle 0 for every frequency, so the rotation is the identity.
        let pe = wan_pe(2, 30, 52, 1, 2, 2, &AXES, 10000.0);
        let pairs: usize = AXES.iter().map(|d| d / 2).sum();
        for j in 0..pairs {
            let m = &pe[j * PAIR_STRIDE..j * PAIR_STRIDE + 4];
            assert!((m[0] - 1.0).abs() < 1e-6 && m[1].abs() < 1e-6, "pair {j} not identity: {m:?}");
        }
    }

    #[test]
    fn apply_rope_preserves_length() {
        let pe = wan_pe(2, 30, 52, 1, 2, 2, &AXES, 10000.0);
        let pairs: usize = AXES.iter().map(|d| d / 2).sum();
        let mut v: Vec<f32> = (0..128).map(|i| (i as f32 * 0.017).sin()).collect();
        let before: f32 = v.iter().map(|x| x * x).sum();
        let pos = 137;
        apply_rope(&mut v, &pe[pos * pairs * PAIR_STRIDE..(pos + 1) * pairs * PAIR_STRIDE]);
        let after: f32 = v.iter().map(|x| x * x).sum();
        assert!((before - after).abs() < 1e-3, "rotation must preserve norm: {before} vs {after}");
    }

    #[test]
    fn matches_reference_pe_dump() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/dit/wan_pe.bin");
        let b = std::fs::read(&p).unwrap();
        assert_eq!(&b[..4], b"SQD1");
        let nd = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        let dims: Vec<i64> = (0..nd)
            .map(|i| i64::from_le_bytes(b[8 + i * 8..16 + i * 8].try_into().unwrap()))
            .collect();
        let want: Vec<f32> = b[8 + nd * 8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        eprintln!("reference pe dims={dims:?} numel={}", want.len());

        let got = wan_pe(2, 30, 52, 1, 2, 2, &AXES, 10000.0);
        assert_eq!(got.len(), want.len(), "pe element count");
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("pe max_abs_diff = {max_abs:.9}");
        assert!(max_abs < 1e-5, "pe does not match reference (max_abs {max_abs})");
    }
}
