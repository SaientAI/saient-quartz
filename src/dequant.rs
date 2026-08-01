use rayon::prelude::*;

// Dequantize GGML tensor data to f32.
//
// Supported types:
//   0  = F32
//   1  = F16
//   6  = Q5_0   (32 values / block, 22 bytes)
//   7  = Q5_1   (32 values / block, 24 bytes)
//   8  = Q8_0   (32 values / block, 34 bytes)
//   12 = Q4_K   (256 values / block, 144 bytes)
//   14 = Q6_K   (256 values / block, 210 bytes)
//   30 = BF16   (2 bytes each)

/// Fused dequant + matrix-vector multiply: y = W * x, where W is stored in `data`.
/// W is [out_dim × in_dim] row-major in the given quantized format.
/// Avoids materializing the full f32 weight matrix — keeps hot data in L1/L2 cache.
pub fn gemv(x: &[f32], data: &[u8], ggml_type: u32, in_dim: usize, out_dim: usize) -> Vec<f32> {
    match ggml_type {
        7 => gemv_q5_1(x, data, in_dim, out_dim),
        12 => gemv_q4_k(x, data, in_dim, out_dim),
        14 => gemv_q6_k(x, data, in_dim, out_dim),
        20 => gemv_iq4nl(x, data, in_dim, out_dim),
        30 => gemv_bf16(x, data, in_dim, out_dim),
        0 => gemv_f32_data(x, data, in_dim, out_dim),
        _ => {
            let w = dequant(data, ggml_type, in_dim * out_dim);
            crate::matmul(x, &w, in_dim, out_dim)
        }
    }
}

// One Q4_K output row dotted against ALL n tokens at once. Each 256-weight block
// is decoded once into a tiny L1-resident buffer, then dotted against every
// token's matching slice before moving on — so weight decode is amortized over n
// AND the decoded block stays hot. acc[t] accumulates token t's dot (len n).
#[inline(always)]
fn q4k_row_batched(x: &[f32], in_dim: usize, n: usize, row: &[u8], acc: &mut [f32]) {
    const BYTES: usize = 144;
    let bpr = in_dim / 256;
    for b in 0..bpr {
        let blk = &row[b * BYTES..(b + 1) * BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let mut wv = [0.0f32; 256];
        let (mut q_off, mut is, mut o) = (0usize, 0usize, 0usize);
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            for l in 0..32 {
                wv[o + l] = d1 * (qs[q_off + l] & 0x0F) as f32 - m1;
            }
            o += 32;
            for l in 0..32 {
                wv[o + l] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
            o += 32;
            q_off += 32;
            is += 2;
        }
        let base = b * 256;
        for t in 0..n {
            let xt = &x[t * in_dim + base..t * in_dim + base + 256];
            let mut s = 0.0f32;
            for i in 0..256 {
                s += xt[i] * wv[i];
            }
            acc[t] += s;
        }
    }
}

// Same idea for Q6_K.
#[inline(always)]
fn q6k_row_batched(x: &[f32], in_dim: usize, n: usize, row: &[u8], acc: &mut [f32]) {
    const BYTES: usize = 210;
    let bpr = in_dim / 256;
    for b in 0..bpr {
        let blk = &row[b * BYTES..(b + 1) * BYTES];
        let ql_base = &blk[0..128];
        let qh_base = &blk[128..192];
        let sc_base = &blk[192..208];
        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let mut wv = [0.0f32; 256];
        let (mut ql_off, mut qh_off, mut sc_off, mut y_off) = (0usize, 0usize, 0usize, 0usize);
        for _ in 0..2 {
            let ql = &ql_base[ql_off..ql_off + 64];
            let qh = &qh_base[qh_off..qh_off + 32];
            let sc = &sc_base[sc_off..sc_off + 8];
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                wv[y_off + l + 0] = d * sc[is + 0] as i8 as f32 * q1 as f32;
                wv[y_off + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                wv[y_off + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                wv[y_off + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            y_off += 128;
        }
        let base = b * 256;
        for t in 0..n {
            let xt = &x[t * in_dim + base..t * in_dim + base + 256];
            let mut s = 0.0f32;
            for i in 0..256 {
                s += xt[i] * wv[i];
            }
            acc[t] += s;
        }
    }
}

fn q4k_batched_scalar(x: &[f32], in_dim: usize, n: usize, row: &[u8], acc: &mut [f32]) {
    q4k_row_batched(x, in_dim, n, row, acc)
}
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
unsafe fn q4k_batched_avx2(x: &[f32], in_dim: usize, n: usize, row: &[u8], acc: &mut [f32]) {
    q4k_row_batched(x, in_dim, n, row, acc)
}
fn q6k_batched_scalar(x: &[f32], in_dim: usize, n: usize, row: &[u8], acc: &mut [f32]) {
    q6k_row_batched(x, in_dim, n, row, acc)
}
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
unsafe fn q6k_batched_avx2(x: &[f32], in_dim: usize, n: usize, row: &[u8], acc: &mut [f32]) {
    q6k_row_batched(x, in_dim, n, row, acc)
}

// Batched matrix-vector: X [n, in_dim] times W^T -> [n, out_dim] (token-major).
// Block-fused: each weight block is decoded once and dotted against all n tokens,
// so decode is amortized over the prompt while staying cache-resident. Parallel
// over output rows. Falls back to gemv for n == 1.
pub fn gemm(
    x: &[f32],
    data: &[u8],
    ggml_type: u32,
    in_dim: usize,
    out_dim: usize,
    n: usize,
) -> Vec<f32> {
    if n == 1 {
        return gemv(x, data, ggml_type, in_dim, out_dim);
    }
    let row_bytes = crate::gguf::ggml_type_size(ggml_type, in_dim);
    let avx = cpu_has_avx2();
    let cols: Vec<Vec<f32>> = (0..out_dim)
        .into_par_iter()
        .map(|j| {
            let row = &data[j * row_bytes..(j + 1) * row_bytes];
            let mut acc = vec![0.0f32; n];
            match ggml_type {
                12 => {
                    if avx {
                        unsafe { q4k_batched_avx2(x, in_dim, n, row, &mut acc) }
                    } else {
                        q4k_batched_scalar(x, in_dim, n, row, &mut acc)
                    }
                }
                14 => {
                    if avx {
                        unsafe { q6k_batched_avx2(x, in_dim, n, row, &mut acc) }
                    } else {
                        q6k_batched_scalar(x, in_dim, n, row, &mut acc)
                    }
                }
                _ => {
                    // Generic fallback: decode the row once, dot against each token.
                    let w = dequant(row, ggml_type, in_dim);
                    for t in 0..n {
                        let xt = &x[t * in_dim..(t + 1) * in_dim];
                        acc[t] = if avx {
                            unsafe { dot_avx2(xt, &w) }
                        } else {
                            dot_scalar(xt, &w)
                        };
                    }
                }
            }
            acc
        })
        .collect();
    let mut out = vec![0.0f32; n * out_dim];
    for j in 0..out_dim {
        let col = &cols[j];
        for t in 0..n {
            out[t * out_dim + j] = col[t];
        }
    }
    out
}

// ── Fused Q4_K / Q6_K matrix-vector products ────────────────────────────────────
//
// Both dequantize each 256-element block on the fly straight into the dot product
// (one output row per rayon task), avoiding the old path that materialized the
// entire dequantized weight matrix (tens of MB) on EVERY call -- that was ~100x
// too slow. The per-row math lives in `#[inline(always)]` helpers so it can be
// compiled twice: a baseline scalar version that runs on any x86-64 CPU, and an
// AVX2+FMA version. `gemv_*` picks the AVX2 path only when the CPU reports those
// features at runtime, so a single binary runs everywhere and crashes nowhere.

// One Q4_K output row. value = d1*(nibble) - m1  =>  x·value = d1*Σ(x*q) - m1*Σx.
#[inline(always)]
fn q4k_row(x: &[f32], row: &[u8]) -> f32 {
    const BYTES: usize = 144;
    let bpr = row.len() / BYTES;
    let mut acc = 0.0f32;
    for b in 0..bpr {
        let blk = &row[b * BYTES..(b + 1) * BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let xb = &x[b * 256..b * 256 + 256];
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut o = 0usize;
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            let mut sq1 = 0.0f32;
            let mut sx1 = 0.0f32;
            for l in 0..32 {
                let xv = xb[o + l];
                sq1 += xv * (qs[q_off + l] & 0x0F) as f32;
                sx1 += xv;
            }
            acc += d1 * sq1 - m1 * sx1;
            o += 32;

            let mut sq2 = 0.0f32;
            let mut sx2 = 0.0f32;
            for l in 0..32 {
                let xv = xb[o + l];
                sq2 += xv * (qs[q_off + l] >> 4) as f32;
                sx2 += xv;
            }
            acc += d2 * sq2 - m2 * sx2;
            o += 32;

            q_off += 32;
            is += 2;
        }
    }
    acc
}

// One Q6_K output row. Decodes each block into a small stack buffer then dots.
#[inline(always)]
fn q6k_row(x: &[f32], row: &[u8]) -> f32 {
    const BYTES: usize = 210;
    let bpr = row.len() / BYTES;
    let mut acc = 0.0f32;
    for b in 0..bpr {
        let blk = &row[b * BYTES..(b + 1) * BYTES];
        let ql_base = &blk[0..128];
        let qh_base = &blk[128..192];
        let sc_base = &blk[192..208];
        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let xb = &x[b * 256..b * 256 + 256];

        let mut y = [0.0f32; 256];
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut y_off = 0usize;
        for _ in 0..2 {
            let ql = &ql_base[ql_off..ql_off + 64];
            let qh = &qh_base[qh_off..qh_off + 32];
            let sc = &sc_base[sc_off..sc_off + 8];
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                y[y_off + l + 0] = d * sc[is + 0] as i8 as f32 * q1 as f32;
                y[y_off + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                y[y_off + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                y[y_off + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            y_off += 128;
        }
        for i in 0..256 {
            acc += xb[i] * y[i];
        }
    }
    acc
}

fn q4k_row_scalar(x: &[f32], row: &[u8]) -> f32 {
    q4k_row(x, row)
}
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
unsafe fn q4k_row_avx2(x: &[f32], row: &[u8]) -> f32 {
    q4k_row(x, row)
}

// ── ARM NEON Q4_K row dot ─────────────────────────────────────────────────────
// The scalar float reductions don't auto-vectorise without fast-math, so on
// aarch64 we hand-vectorise: unpack 4-bit nibbles → f32 lanes and FMA against x.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn q4_dot32_neon(x: *const f32, qs: *const u8, hi: bool) -> (f32, f32) {
    use core::arch::aarch64::*;
    // Rust 2024 requires unsafe operations to stay inside an explicit boundary
    // even when their containing helper is itself unsafe.
    unsafe {
        let q0 = vld1q_u8(qs);
        let q1 = vld1q_u8(qs.add(16));
        let (n0, n1) = if hi {
            (vshrq_n_u8::<4>(q0), vshrq_n_u8::<4>(q1))
        } else {
            let m = vdupq_n_u8(0x0F);
            (vandq_u8(q0, m), vandq_u8(q1, m))
        };
        let mut accsq = vdupq_n_f32(0.0);
        let mut accsx = vdupq_n_f32(0.0);
        for (ni, base) in [(n0, 0usize), (n1, 16usize)] {
            let lo16 = vmovl_u8(vget_low_u8(ni));
            let hi16 = vmovl_u8(vget_high_u8(ni));
            let chunks = [
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))),
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))),
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))),
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))),
            ];
            let mut k = 0usize;
            while k < 4 {
                let xv = vld1q_f32(x.add(base + k * 4));
                accsq = vfmaq_f32(accsq, xv, chunks[k]);
                accsx = vaddq_f32(accsx, xv);
                k += 1;
            }
        }
        (vaddvq_f32(accsq), vaddvq_f32(accsx))
    }
}

#[cfg(target_arch = "aarch64")]
fn q4k_row_neon(x: &[f32], row: &[u8]) -> f32 {
    const BYTES: usize = 144;
    let bpr = row.len() / BYTES;
    let mut acc = 0.0f32;
    for b in 0..bpr {
        let blk = &row[b * BYTES..(b + 1) * BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let xb = &x[b * 256..b * 256 + 256];
        let (mut q_off, mut is, mut o) = (0usize, 0usize, 0usize);
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;
            unsafe {
                let (sq1, sx1) = q4_dot32_neon(xb.as_ptr().add(o), qs.as_ptr().add(q_off), false);
                acc += d1 * sq1 - m1 * sx1;
                let (sq2, sx2) =
                    q4_dot32_neon(xb.as_ptr().add(o + 32), qs.as_ptr().add(q_off), true);
                acc += d2 * sq2 - m2 * sx2;
            }
            o += 64;
            q_off += 32;
            is += 2;
        }
    }
    acc
}

fn q6k_row_scalar(x: &[f32], row: &[u8]) -> f32 {
    q6k_row(x, row)
}
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
unsafe fn q6k_row_avx2(x: &[f32], row: &[u8]) -> f32 {
    q6k_row(x, row)
}

#[inline]
fn cpu_has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for k in 0..a.len() {
        s += a[k] * b[k];
    }
    s
}
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b)
}

fn gemv_q4_k(x: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    debug_assert_eq!(in_dim % 256, 0, "in_dim must be a multiple of 256 for Q4_K");
    let row_bytes = (in_dim / 256) * 144;
    #[cfg(not(target_arch = "aarch64"))]
    let avx = cpu_has_avx2();
    (0..out_dim)
        .into_par_iter()
        .map(|j| {
            let row = &data[j * row_bytes..(j + 1) * row_bytes];
            #[cfg(target_arch = "aarch64")]
            {
                q4k_row_neon(x, row)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                if avx {
                    unsafe { q4k_row_avx2(x, row) }
                } else {
                    q4k_row_scalar(x, row)
                }
            }
        })
        .collect()
}

fn gemv_q6_k(x: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    debug_assert_eq!(in_dim % 256, 0, "in_dim must be a multiple of 256 for Q6_K");
    let row_bytes = (in_dim / 256) * 210;
    let avx = cpu_has_avx2();
    (0..out_dim)
        .into_par_iter()
        .map(|j| {
            let row = &data[j * row_bytes..(j + 1) * row_bytes];
            if avx {
                unsafe { q6k_row_avx2(x, row) }
            } else {
                q6k_row_scalar(x, row)
            }
        })
        .collect()
}

fn gemv_q5_1(x: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    debug_assert_eq!(in_dim % 32, 0, "in_dim must be multiple of 32 for Q5_1");
    const BSIZ: usize = 24; // bytes per Q5_1 block of 32 elements
    let bpr = in_dim / 32; // blocks per row
    let row_bytes = bpr * BSIZ;
    (0..out_dim)
        .into_par_iter()
        .map(|j| {
            let row = &data[j * row_bytes..(j + 1) * row_bytes];
            let mut acc = 0.0f32;
            for b in 0..bpr {
                let blk = &row[b * BSIZ..(b + 1) * BSIZ];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                let qs = &blk[8..24];
                let xi = &x[b * 32..];
                for k in 0..16usize {
                    let xh0 = ((qh >> k) << 4) & 0x10;
                    let xh1 = (qh >> (k + 12)) & 0x10;
                    let q0 = ((qs[k] as u32 & 0x0F) | xh0) as f32;
                    let q1 = ((qs[k] as u32 >> 4) | xh1) as f32;
                    acc += xi[k] * (q0 * d + m) + xi[k + 16] * (q1 * d + m);
                }
            }
            acc
        })
        .collect()
}

fn gemv_bf16(x: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let row_bytes = in_dim * 2;
    (0..out_dim)
        .into_par_iter()
        .map(|j| {
            let row = &data[j * row_bytes..(j + 1) * row_bytes];
            (0..in_dim)
                .map(|k| {
                    let bits = u16::from_le_bytes([row[k * 2], row[k * 2 + 1]]);
                    x[k] * f32::from_bits((bits as u32) << 16)
                })
                .sum::<f32>()
        })
        .collect()
}

fn gemv_f32_data(x: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let row_bytes = in_dim * 4;
    (0..out_dim)
        .into_par_iter()
        .map(|j| {
            let row = &data[j * row_bytes..(j + 1) * row_bytes];
            (0..in_dim)
                .map(|k| {
                    let w = f32::from_le_bytes([
                        row[k * 4],
                        row[k * 4 + 1],
                        row[k * 4 + 2],
                        row[k * 4 + 3],
                    ]);
                    x[k] * w
                })
                .sum::<f32>()
        })
        .collect()
}

// ── IQ2 (XXS / XS / S) — 2-bit grid-codebook quants ────────────────────────────
// The grid is ggml's hardcoded iq2*_grid table: each u64 packs the 8 int8 codebook
// values for that entry (e.g. iq2s_grid[0] = 0x08…08 = [8,8,8,8,8,8,8,8]).
fn build_iq2_grid(table: &[u64]) -> Vec<[i8; 8]> {
    table
        .iter()
        .map(|&g| {
            let b = g.to_le_bytes();
            [
                b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8, b[4] as i8, b[5] as i8, b[6] as i8,
                b[7] as i8,
            ]
        })
        .collect()
}
fn iq2_grid(grid_size: usize) -> &'static Vec<[i8; 8]> {
    use crate::iq2_tables::*;
    use std::sync::OnceLock;
    static G256: OnceLock<Vec<[i8; 8]>> = OnceLock::new();
    static G512: OnceLock<Vec<[i8; 8]>> = OnceLock::new();
    static G1024: OnceLock<Vec<[i8; 8]>> = OnceLock::new();
    match grid_size {
        256 => G256.get_or_init(|| build_iq2_grid(&IQ2XXS_GRID)),
        512 => G512.get_or_init(|| build_iq2_grid(&IQ2XS_GRID)),
        _ => G1024.get_or_init(|| build_iq2_grid(&IQ2S_GRID)),
    }
}

fn dequant_iq2_xxs(data: &[u8], n_elems: usize) -> Vec<f32> {
    use crate::iq2_tables::{KMASK_IQ2XS, KSIGNS_IQ2XS};
    let grid = iq2_grid(256);
    let nb = n_elems / 256;
    const BS: usize = 66;
    let mut out = Vec::with_capacity(n_elems);
    for ib in 0..nb {
        let blk = &data[ib * BS..ib * BS + BS];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..]; // 64 bytes = u16[32]
        for ib32 in 0..8 {
            let b = 8 * ib32;
            let a0 = [qs[b], qs[b + 1], qs[b + 2], qs[b + 3]]; // 4 grid indices
            let a1 = u32::from_le_bytes([qs[b + 4], qs[b + 5], qs[b + 6], qs[b + 7]]); // signs + scale
            let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let g = &grid[a0[l] as usize];
                let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(db * g[j] as f32 * s);
                }
            }
        }
    }
    out
}

fn dequant_iq2_xs(data: &[u8], n_elems: usize) -> Vec<f32> {
    use crate::iq2_tables::{KMASK_IQ2XS, KSIGNS_IQ2XS};
    let grid = iq2_grid(512);
    let nb = n_elems / 256;
    const BS: usize = 74;
    let mut out = Vec::with_capacity(n_elems);
    for ib in 0..nb {
        let blk = &data[ib * BS..ib * BS + BS];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66]; // u16[32]
        let scales = &blk[66..74]; // u8[8]
        for ib32 in 0..8 {
            let db0 = d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let qi = 2 * (4 * ib32 + l);
                let q = u16::from_le_bytes([qs[qi], qs[qi + 1]]);
                let g = &grid[(q & 511) as usize];
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                let db = if l < 2 { db0 } else { db1 };
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(db * g[j] as f32 * s);
                }
            }
        }
    }
    out
}

fn dequant_iq2_s(data: &[u8], n_elems: usize) -> Vec<f32> {
    use crate::iq2_tables::KMASK_IQ2XS;
    let grid = iq2_grid(1024);
    let nb = n_elems / 256;
    const BS: usize = 82;
    let mut out = Vec::with_capacity(n_elems);
    for ib in 0..nb {
        let blk = &data[ib * BS..ib * BS + BS];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66]; // u8[64]: [0..32]=idx, [32..64]=signs
        let qh = &blk[66..74]; // u8[8]
        let scales = &blk[74..82]; // u8[8]
        for ib32 in 0..8 {
            let db0 = d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
            for l in 0..4 {
                let db = if l < 2 { db0 } else { db1 };
                let idx =
                    (qs[4 * ib32 + l] as usize) | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                let g = &grid[idx];
                let signs = qs[32 + 4 * ib32 + l];
                for j in 0..8 {
                    let s = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(db * g[j] as f32 * s);
                }
            }
        }
    }
    out
}

// ── Q5_K — standard 5-bit K-quant (256/block, 176 bytes) ───────────────────────
fn dequant_q2_k(data: &[u8], n_elems: usize) -> Vec<f32> {
    let nb = n_elems / 256;
    const BS: usize = 84;
    let mut out = Vec::with_capacity(n_elems);
    for ib in 0..nb {
        let blk = &data[ib * BS..ib * BS + BS];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
        let mut is = 0usize;
        for n in (0..256).step_by(128) {
            let q = &qs[(n / 128) * 32..(n / 128) * 32 + 32];
            for j in 0..4 {
                let shift = 2 * j;
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0xf) as f32, dmin * (sc >> 4) as f32);
                for l in 0..16 {
                    out.push(dl * (((q[l] >> shift) & 3) as f32) - ml);
                }
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0xf) as f32, dmin * (sc >> 4) as f32);
                for l in 0..16 {
                    out.push(dl * (((q[l + 16] >> shift) & 3) as f32) - ml);
                }
            }
        }
    }
    out
}

fn dequant_q5_k(data: &[u8], n_elems: usize) -> Vec<f32> {
    let nb = n_elems / 256;
    const BS: usize = 176;
    let mut out = Vec::with_capacity(n_elems);
    for ib in 0..nb {
        let blk = &data[ib * BS..ib * BS + BS];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qh = &blk[16..48]; // QK_K/8 = 32 bytes (high bits)
        let qs = &blk[48..176]; // QK_K/2 = 128 bytes (low 4 bits)
        let (mut is, mut u1, mut u2) = (0usize, 1u8, 2u8);
        for jb in 0..4 {
            // QK_K in steps of 64
            let ql = &qs[jb * 32..jb * 32 + 32];
            let (sc1, mm1) = get_scale_min_k4(is, scales);
            let (d1, m1) = (d * sc1 as f32, dmin * mm1 as f32);
            let (sc2, mm2) = get_scale_min_k4(is + 1, scales);
            let (d2, m2) = (d * sc2 as f32, dmin * mm2 as f32);
            for l in 0..32 {
                out.push(
                    d1 * (((ql[l] & 0xF) as f32) + if qh[l] & u1 != 0 { 16.0 } else { 0.0 }) - m1,
                );
            }
            for l in 0..32 {
                out.push(
                    d2 * (((ql[l] >> 4) as f32) + if qh[l] & u2 != 0 { 16.0 } else { 0.0 }) - m2,
                );
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out
}

// ── IQ3_S — 3-bit grid-codebook quant (256/block, 110 bytes) ───────────────────
fn dequant_iq3_s(data: &[u8], n_elems: usize) -> Vec<f32> {
    use crate::iq2_tables::KMASK_IQ2XS;
    use crate::iq3_tables::IQ3S_GRID;
    #[inline]
    fn g(idx: usize) -> [i8; 4] {
        let b = IQ3S_GRID[idx].to_le_bytes();
        [b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8]
    }
    let nb = n_elems / 256;
    const BS: usize = 110;
    let mut out = Vec::with_capacity(n_elems);
    for ib in 0..nb {
        let blk = &data[ib * BS..ib * BS + BS];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66]; // 64 bytes
        let qh = &blk[66..74]; // 8 bytes
        let sg = &blk[74..106]; // 32 bytes (signs)
        let scales = &blk[106..110]; // 4 bytes
        let (mut qo, mut so, mut ho) = (0usize, 0usize, 0usize);
        for ib32 in (0..8).step_by(2) {
            let db1 = d * (1.0 + 2.0 * (scales[ib32 / 2] & 0xf) as f32);
            let db2 = d * (1.0 + 2.0 * (scales[ib32 / 2] >> 4) as f32);
            for (half, db, qhb) in [(0usize, db1, qh[ho]), (1usize, db2, qh[ho + 1])] {
                let qbase = qo + half * 8;
                let sbase = so + half * 4;
                for l in 0..4 {
                    let i0 = qs[qbase + 2 * l] as usize | (((qhb as usize) << (8 - 2 * l)) & 256);
                    let i1 =
                        qs[qbase + 2 * l + 1] as usize | (((qhb as usize) << (7 - 2 * l)) & 256);
                    let (g1, g2, s) = (g(i0), g(i1), sg[sbase + l]);
                    for j in 0..4 {
                        out.push(
                            db * g1[j] as f32 * if s & KMASK_IQ2XS[j] != 0 { -1.0 } else { 1.0 },
                        );
                    }
                    for j in 0..4 {
                        out.push(
                            db * g2[j] as f32
                                * if s & KMASK_IQ2XS[j + 4] != 0 {
                                    -1.0
                                } else {
                                    1.0
                                },
                        );
                    }
                }
            }
            qo += 16;
            so += 8;
            ho += 2;
        }
    }
    out
}

pub fn dequant(data: &[u8], ggml_type: u32, n_elems: usize) -> Vec<f32> {
    match ggml_type {
        0 => dequant_f32(data, n_elems),
        1 => dequant_f16(data, n_elems),
        6 => dequant_q5_0(data, n_elems),
        7 => dequant_q5_1(data, n_elems),
        8 => dequant_q8_0(data, n_elems),
        10 => dequant_q2_k(data, n_elems),
        12 => dequant_q4_k(data, n_elems),
        13 => dequant_q5_k(data, n_elems),
        14 => dequant_q6_k(data, n_elems),
        16 => dequant_iq2_xxs(data, n_elems),
        17 => dequant_iq2_xs(data, n_elems),
        21 => dequant_iq3_s(data, n_elems),
        22 => dequant_iq2_s(data, n_elems),
        20 => dequant_iq4nl(data, n_elems),
        30 => dequant_bf16(data, n_elems),
        t => panic!("dequant: unsupported ggml_type {}", t),
    }
}

/// Dequant into a pre-allocated buffer — avoids large Vec allocation on each call.
/// `buf` is resized on first call; subsequent calls with the same n_elems reuse it.
pub fn dequant_into(data: &[u8], ggml_type: u32, n_elems: usize, buf: &mut Vec<f32>) {
    // Safety: we will overwrite every element below, so skip zero-init on resize.
    if buf.len() != n_elems {
        buf.clear();
        buf.reserve(n_elems);
        // SAFETY: we will write every element in the match below before reading.
        unsafe {
            buf.set_len(n_elems);
        }
    }
    match ggml_type {
        0 => dequant_f32_into(data, n_elems, buf),
        7 => dequant_q5_1_into(data, n_elems, buf),
        30 => dequant_bf16_into(data, n_elems, buf),
        _ => {
            let tmp = dequant(data, ggml_type, n_elems);
            buf.copy_from_slice(&tmp);
        }
    }
}

fn dequant_f32_into(data: &[u8], n: usize, out: &mut Vec<f32>) {
    for i in 0..n {
        out[i] = f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
    }
}

fn dequant_bf16_into(data: &[u8], n: usize, out: &mut Vec<f32>) {
    for i in 0..n {
        let bits = u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap());
        out[i] = f32::from_bits((bits as u32) << 16);
    }
}

fn dequant_q5_1_into(data: &[u8], n: usize, out: &mut Vec<f32>) {
    const BLOCK: usize = 32;
    const BYTES: usize = 24;
    let n_blocks = n / BLOCK;
    out.par_chunks_mut(BLOCK)
        .zip(data.par_chunks(BYTES))
        .for_each(|(chunk, blk)| {
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
            let qs = &blk[8..24];
            for j in 0..16usize {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((qs[j] as u32 & 0x0F) | xh_0) as f32;
                let x1 = ((qs[j] as u32 >> 4) | xh_1) as f32;
                chunk[j] = x0 * d + m;
                chunk[j + 16] = x1 * d + m;
            }
        });
    let _ = n_blocks; // used implicitly via n_blocks * BLOCK
}

// ── F32 ───────────────────────────────────────────────────────────────────────

fn dequant_f32(data: &[u8], n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect()
}

// ── F16 ───────────────────────────────────────────────────────────────────────

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut e = 0u32;
        let mut m = mant;
        while m & 0x400 == 0 {
            m <<= 1;
            e += 1;
        }
        return f32::from_bits((sign << 31) | ((127 - 15 - e + 1) << 23) | ((m & 0x3FF) << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | 0x7F800000 | (mant << 13));
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13))
}

fn dequant_f16(data: &[u8], n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            f16_to_f32(u16::from_le_bytes(
                data[i * 2..i * 2 + 2].try_into().unwrap(),
            ))
        })
        .collect()
}

// ── BF16 ──────────────────────────────────────────────────────────────────────
// BF16 is just the top 16 bits of a float32 — shift left by 16 to recover f32.

fn dequant_bf16(data: &[u8], n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let bits = u16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap());
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

// ── Q5_0 ──────────────────────────────────────────────────────────────────────
//
// Block: [d: f16][qh: 4 bytes][qs: 16 bytes] = 22 bytes, 32 values
// Each value is 5-bit signed (-16..15): 4 bits from qs nibble + 1 bit from qh.
// Output positions: j and j+16 (not j*2 and j*2+1).
//
// Reference: dequantize_row_q5_0 in llama.cpp ggml-quants.c

fn dequant_q5_0(data: &[u8], n: usize) -> Vec<f32> {
    const BLOCK: usize = 32;
    const BYTES: usize = 22;
    let n_blocks = n / BLOCK;
    let mut out = vec![0.0f32; n_blocks * BLOCK];

    for b in 0..n_blocks {
        let blk = &data[b * BYTES..(b + 1) * BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        let base = b * BLOCK;

        for j in 0..16usize {
            // high bit for y[j]: bit j of qh, shifted to position 4
            let xh_0 = ((qh >> j) << 4) & 0x10;
            // high bit for y[j+16]: bit j+16 of qh
            let xh_1 = (qh >> (j + 12)) & 0x10;

            let x0 = ((qs[j] as u32 & 0x0F) | xh_0) as i32 - 16;
            let x1 = ((qs[j] as u32 >> 4) | xh_1) as i32 - 16;

            out[base + j] = x0 as f32 * d;
            out[base + j + 16] = x1 as f32 * d;
        }
    }

    out
}

// ── Q5_1 ──────────────────────────────────────────────────────────────────────
//
// Block: [d: f16][m: f16][qh: 4 bytes][qs: 16 bytes] = 24 bytes, 32 values
// 5-bit unsigned: low 4 bits from qs nibble, high bit from qh.
// Output: q5 * d + m   (unsigned, no bias subtraction)
//
// Reference: dequantize_row_q5_1 in ggml-quants.c

fn dequant_q5_1(data: &[u8], n: usize) -> Vec<f32> {
    const BLOCK: usize = 32;
    const BYTES: usize = 24;
    let n_blocks = n / BLOCK;
    let mut out = vec![0.0f32; n_blocks * BLOCK];

    out.par_chunks_mut(BLOCK)
        .zip(data.par_chunks(BYTES))
        .for_each(|(chunk, blk)| {
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let m = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
            let qs = &blk[8..24];
            for j in 0..16usize {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((qs[j] as u32 & 0x0F) | xh_0) as f32;
                let x1 = ((qs[j] as u32 >> 4) | xh_1) as f32;
                chunk[j] = x0 * d + m;
                chunk[j + 16] = x1 * d + m;
            }
        });

    out
}

// ── Q8_0 ──────────────────────────────────────────────────────────────────────
//
// Block: [d: f16][qs: 32 × i8] = 34 bytes, 32 values
// value = d * qs[i]

fn dequant_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    const BLOCK: usize = 32;
    const BYTES: usize = 34;
    let n_blocks = n / BLOCK;
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let blk = &data[b * BYTES..(b + 1) * BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for i in 0..BLOCK {
            out.push(d * blk[2 + i] as i8 as f32);
        }
    }
    out
}

// ── Q4_K ──────────────────────────────────────────────────────────────────────
//
// Block: [d: f16][dmin: f16][scales: 12 bytes][qs: 128 bytes] = 144 bytes, 256 values
//
// Reference: dequantize_row_q4_K in llama.cpp ggml-quants.c

fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(data: &[u8], n: usize) -> Vec<f32> {
    const BLOCK: usize = 256;
    const BYTES: usize = 144;
    let n_blocks = n / BLOCK;
    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let blk = &data[b * BYTES..(b + 1) * BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];

        let mut q_off = 0usize;
        let mut is = 0usize;

        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let m1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let m2 = dmin * m2 as f32;

            for l in 0..32 {
                out.push(d1 * (qs[q_off + l] & 0x0F) as f32 - m1);
            }
            for l in 0..32 {
                out.push(d2 * (qs[q_off + l] >> 4) as f32 - m2);
            }

            q_off += 32;
            is += 2;
        }
    }

    out
}

// ── IQ4_NL ────────────────────────────────────────────────────────────────────
//
// Block: [d: f16][qs: 16 bytes] = 18 bytes, 32 values
// 4-bit indices into non-linear lookup table; two per byte (low nibble = k, high = k+16).
// value = d * table[idx]

const IQ4NL_TABLE: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

fn gemv_iq4nl(x: &[f32], data: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    debug_assert_eq!(in_dim % 32, 0);
    const BSIZ: usize = 18;
    let bpr = in_dim / 32;
    let row_bytes = bpr * BSIZ;
    (0..out_dim)
        .into_par_iter()
        .map(|j| {
            let row = &data[j * row_bytes..(j + 1) * row_bytes];
            let mut acc = 0.0f32;
            for b in 0..bpr {
                let blk = &row[b * BSIZ..(b + 1) * BSIZ];
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qs = &blk[2..18];
                let xi = &x[b * 32..];
                for k in 0..16 {
                    acc += xi[k] * d * IQ4NL_TABLE[(qs[k] & 0x0F) as usize];
                    acc += xi[k + 16] * d * IQ4NL_TABLE[(qs[k] >> 4) as usize];
                }
            }
            acc
        })
        .collect()
}

fn dequant_iq4nl(data: &[u8], n: usize) -> Vec<f32> {
    const BLOCK: usize = 32;
    const BYTES: usize = 18;
    let n_blocks = n / BLOCK;
    let mut out = Vec::with_capacity(n);
    for b in 0..n_blocks {
        let blk = &data[b * BYTES..(b + 1) * BYTES];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..18];
        let mut block = [0.0f32; 32];
        for k in 0..16 {
            block[k] = d * IQ4NL_TABLE[(qs[k] & 0x0F) as usize];
            block[k + 16] = d * IQ4NL_TABLE[(qs[k] >> 4) as usize];
        }
        out.extend_from_slice(&block);
    }
    out
}

// ── Q6_K ──────────────────────────────────────────────────────────────────────
//
// Block: [ql: 128 bytes][qh: 64 bytes][scales: 16 × i8][d: f16] = 210 bytes, 256 values
// Each value is 6-bit signed (-32..31): 4 bits from ql + 2 bits from qh, then - 32.
//
// Reference: dequantize_row_q6_K in llama.cpp ggml-quants.c

fn dequant_q6_k(data: &[u8], n: usize) -> Vec<f32> {
    const BLOCK: usize = 256;
    const BYTES: usize = 210;
    let n_blocks = n / BLOCK;
    let mut out = Vec::with_capacity(n);

    for b in 0..n_blocks {
        let blk = &data[b * BYTES..(b + 1) * BYTES];
        // layout: ql[128] qh[64] scales[16] d[2]
        let ql_base = &blk[0..128];
        let qh_base = &blk[128..192];
        let sc_base = &blk[192..208];
        let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));

        // two chunks of 128 values each
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut y = vec![0.0f32; BLOCK];
        let mut y_off = 0usize;

        for _ in 0..2 {
            let ql = &ql_base[ql_off..ql_off + 64];
            let qh = &qh_base[qh_off..qh_off + 32];
            let sc = &sc_base[sc_off..sc_off + 8];

            for l in 0..32usize {
                let is = l / 16; // 0 for l<16, 1 for l>=16
                let q1 = ((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                y[y_off + l + 0] = d * sc[is + 0] as i8 as f32 * q1 as f32;
                y[y_off + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                y[y_off + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                y[y_off + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
            }

            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            y_off += 128;
        }

        out.extend_from_slice(&y);
    }

    out
}
