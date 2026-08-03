//! Wan2.1 diffusion transformer — the model that actually denoises the video latent.
//!
//! Thirty blocks over a patchified 3D latent. Each block is: AdaLN-modulated self-attention with
//! 3D rotary embedding and QK-RMSNorm, then cross-attention into the UMT5 text embedding, then an
//! MLP — with the timestep supplying six modulation vectors per block (shift/scale/gate for the
//! attention and for the MLP).
//!
//! Points that are easy to get wrong and produce a model that runs but denoises to nothing:
//!
//! - `norm1` and `norm2` are LayerNorm **without** affine parameters; the modulation supplies the
//!   scale and shift instead. `norm3` (before cross-attention) *does* have weight and bias.
//! - QK-RMSNorm is applied over the whole `dim` **before** splitting into heads, not per head.
//! - The head consumes `e` (the timestep embedding) while the blocks consume `e0`
//!   (`time_projection(silu(e))`). They are different tensors.
//! - Patch embedding is a Conv3d whose kernel equals its stride, so it is a gather of each
//!   1x2x2 patch into 64 values followed by a linear — no sliding window.

use crate::dequant;
use crate::gguf::{GgufFile, TensorInfo};
use crate::wan_rope;
use anyhow::{Result, anyhow};

pub struct WanConfig {
    pub dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub text_dim: usize,
    pub freq_dim: usize,
    pub patch: (usize, usize, usize),
    pub eps: f32,
    pub axes_dim: [usize; 3],
    pub theta: f32,
}

impl Default for WanConfig {
    /// Wan2.1 T2V-1.3B. Verified against the shipped engine's own debug output rather than
    /// assumed: dim 1536, 12 heads, ffn 8960, 30 layers.
    fn default() -> Self {
        Self {
            dim: 1536,
            ffn_dim: 8960,
            num_heads: 12,
            head_dim: 128,
            num_layers: 30,
            in_dim: 16,
            out_dim: 16,
            text_dim: 4096,
            freq_dim: 256,
            patch: (1, 2, 2),
            eps: 1e-6,
            axes_dim: [44, 42, 42],
            theta: 10000.0,
        }
    }
}

// ── small numeric helpers ─────────────────────────────────────────────────────

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// tanh-approximation GELU, which is what Wan's MLP and text embedding use.
#[inline]
fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_56;
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// LayerNorm without affine parameters.
fn layer_norm(row: &mut [f32], eps: f32) {
    let n = row.len() as f32;
    let mean = row.iter().sum::<f32>() / n;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for v in row.iter_mut() {
        *v = (*v - mean) * inv;
    }
}

/// LayerNorm with weight and bias.
fn layer_norm_affine(row: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    layer_norm(row, eps);
    for i in 0..row.len() {
        row[i] = row[i] * w[i] + b[i];
    }
}

fn rms_norm(row: &mut [f32], w: &[f32], eps: f32) {
    let rms = (row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32 + eps).sqrt();
    for i in 0..row.len() {
        row[i] = row[i] / rms * w[i];
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Sinusoidal timestep embedding. `cos` occupies the first half and `sin` the second, which is
/// the `flip_sin_to_cos` convention the reference uses.
pub fn timestep_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let arg = t * freq;
        out[i] = arg.cos();
        out[i + half] = arg.sin();
    }
    out
}

// ── weights ───────────────────────────────────────────────────────────────────

struct Linear<'a> {
    w: &'a TensorInfo,
    b: Option<Vec<f32>>,
    in_dim: usize,
    out_dim: usize,
}

struct Attn<'a> {
    q: Linear<'a>,
    k: Linear<'a>,
    v: Linear<'a>,
    o: Linear<'a>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
}

struct DitBlock<'a> {
    modulation: Vec<f32>, // [6 * dim]
    self_attn: Attn<'a>,
    norm3_w: Vec<f32>,
    norm3_b: Vec<f32>,
    cross_attn: Attn<'a>,
    ffn0: Linear<'a>,
    ffn2: Linear<'a>,
}

pub struct WanDit<'a> {
    gguf: &'a GgufFile,
    pub cfg: WanConfig,
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    text0: Linear<'a>,
    text2: Linear<'a>,
    time0: Linear<'a>,
    time2: Linear<'a>,
    time_proj: Linear<'a>,
    blocks: Vec<DitBlock<'a>>,
    head_mod: Vec<f32>, // [2 * dim]
    head: Linear<'a>,
}

impl<'a> WanDit<'a> {
    pub fn load(gguf: &'a GgufFile, cfg: WanConfig) -> Result<Self> {
        let map = gguf.tensor_map();
        let p = "model.diffusion_model";
        let info = |n: String| -> Result<&'a TensorInfo> {
            map.get(n.as_str()).copied().ok_or_else(|| anyhow!("missing tensor {n}"))
        };
        let vals = |t: &TensorInfo| dequant::dequant(gguf.tensor_data(t), t.ggml_type, t.n_elems());
        let lin = |base: String, in_dim: usize, out_dim: usize| -> Result<Linear<'a>> {
            let w = info(format!("{base}.weight"))?;
            let b = map.get(format!("{base}.bias").as_str()).map(|t| vals(t));
            Ok(Linear { w, b, in_dim, out_dim })
        };

        let d = cfg.dim;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let bp = format!("{p}.blocks.{i}");
            let attn = |kind: &str| -> Result<Attn<'a>> {
                Ok(Attn {
                    q: lin(format!("{bp}.{kind}.q"), d, d)?,
                    k: lin(format!("{bp}.{kind}.k"), d, d)?,
                    v: lin(format!("{bp}.{kind}.v"), d, d)?,
                    o: lin(format!("{bp}.{kind}.o"), d, d)?,
                    norm_q: vals(info(format!("{bp}.{kind}.norm_q.weight"))?),
                    norm_k: vals(info(format!("{bp}.{kind}.norm_k.weight"))?),
                })
            };
            blocks.push(DitBlock {
                modulation: vals(info(format!("{bp}.modulation"))?),
                self_attn: attn("self_attn")?,
                norm3_w: vals(info(format!("{bp}.norm3.weight"))?),
                norm3_b: vals(info(format!("{bp}.norm3.bias"))?),
                cross_attn: attn("cross_attn")?,
                ffn0: lin(format!("{bp}.ffn.0"), d, cfg.ffn_dim)?,
                ffn2: lin(format!("{bp}.ffn.2"), cfg.ffn_dim, d)?,
            });
        }

        let patch_t = info(format!("{p}.patch_embedding.weight"))?;
        Ok(Self {
            patch_w: vals(patch_t),
            patch_b: vals(info(format!("{p}.patch_embedding.bias"))?),
            text0: lin(format!("{p}.text_embedding.0"), cfg.text_dim, d)?,
            text2: lin(format!("{p}.text_embedding.2"), d, d)?,
            time0: lin(format!("{p}.time_embedding.0"), cfg.freq_dim, d)?,
            time2: lin(format!("{p}.time_embedding.2"), d, d)?,
            time_proj: lin(format!("{p}.time_projection.1"), d, d * 6)?,
            head_mod: vals(info(format!("{p}.head.modulation"))?),
            head: lin(format!("{p}.head.head"), d, cfg.out_dim * cfg.patch.0 * cfg.patch.1 * cfg.patch.2)?,
            blocks,
            gguf,
            cfg,
        })
    }

    fn apply(&self, l: &Linear<'a>, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = dequant::gemm(x, self.gguf.tensor_data(l.w), l.w.ggml_type, l.in_dim, l.out_dim, n);
        if let Some(b) = &l.b {
            for t in 0..n {
                for j in 0..l.out_dim {
                    y[t * l.out_dim + j] += b[j];
                }
            }
        }
        y
    }

    /// Gather each `1x2x2` patch of the latent into 64 values and project to `dim`.
    ///
    /// The Conv3d kernel equals its stride, so this is a patchify rather than a convolution. The
    /// weight is `[KW, KH, KD, in*out]` with the flattened index `out * in_channels + in`, which
    /// is PyTorch's `[out, in, kd, kh, kw]` ordering seen through ggml's reversed dims.
    fn patchify(&self, latent: &[f32], t: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize, usize) {
        let (pt, ph, pw) = self.cfg.patch;
        let (c, d) = (self.cfg.in_dim, self.cfg.dim);
        let (tl, hl, wl) = (t / pt, h / ph, w / pw);
        let kvol = pt * ph * pw;
        let mut out = vec![0.0f32; tl * hl * wl * d];

        for ti in 0..tl {
            for hj in 0..hl {
                for wk in 0..wl {
                    let tok = (ti * hl + hj) * wl + wk;
                    let o = &mut out[tok * d..(tok + 1) * d];
                    o.copy_from_slice(&self.patch_b);
                    for ci in 0..c {
                        for kd in 0..pt {
                            for kh in 0..ph {
                                for kw in 0..pw {
                                    // latent is [c][t][h][w], row-major with w fastest
                                    let li = ((ci * t + ti * pt + kd) * h + hj * ph + kh) * w + wk * pw + kw;
                                    let xv = latent[li];
                                    let koff = (kd * ph + kh) * pw + kw;
                                    for oc in 0..d {
                                        o[oc] += xv * self.patch_w[(oc * c + ci) * kvol + koff];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        (out, tl, hl, wl)
    }

    /// Attention over `n_q` queries and `n_kv` keys for one projection set.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        attn: &Attn<'a>,
        xq: &[f32],
        xkv: &[f32],
        n_q: usize,
        n_kv: usize,
        pe: Option<&[f32]>,
    ) -> Vec<f32> {
        let (d, h, hd) = (self.cfg.dim, self.cfg.num_heads, self.cfg.head_dim);
        let mut q = self.apply(&attn.q, xq, n_q);
        let mut k = self.apply(&attn.k, xkv, n_kv);
        let v = self.apply(&attn.v, xkv, n_kv);

        // QK-RMSNorm runs over the full dim before the head split.
        for i in 0..n_q {
            rms_norm(&mut q[i * d..(i + 1) * d], &attn.norm_q, self.cfg.eps);
        }
        for i in 0..n_kv {
            rms_norm(&mut k[i * d..(i + 1) * d], &attn.norm_k, self.cfg.eps);
        }

        if let Some(pe) = pe {
            let pairs = hd / 2;
            let stride = pairs * wan_rope::PAIR_STRIDE;
            for i in 0..n_q {
                for head in 0..h {
                    let o = i * d + head * hd;
                    wan_rope::apply_rope(&mut q[o..o + hd], &pe[i * stride..(i + 1) * stride]);
                }
            }
            for i in 0..n_kv {
                for head in 0..h {
                    let o = i * d + head * hd;
                    wan_rope::apply_rope(&mut k[o..o + hd], &pe[i * stride..(i + 1) * stride]);
                }
            }
        }

        let scale = 1.0 / (hd as f32).sqrt();
        let mut ctx = vec![0.0f32; n_q * d];
        let mut scores = vec![0.0f32; n_kv];
        for head in 0..h {
            let off = head * hd;
            for qi in 0..n_q {
                for kj in 0..n_kv {
                    let mut dot = 0.0;
                    for t in 0..hd {
                        dot += q[qi * d + off + t] * k[kj * d + off + t];
                    }
                    scores[kj] = dot * scale;
                }
                softmax(&mut scores);
                for kj in 0..n_kv {
                    let sw = scores[kj];
                    for t in 0..hd {
                        ctx[qi * d + off + t] += sw * v[kj * d + off + t];
                    }
                }
            }
        }
        self.apply(&attn.o, &ctx, n_q)
    }

    /// Denoise one latent.
    ///
    /// `latent` is `[in_dim, t, h, w]` row-major, `context` is `[n_ctx, text_dim]` from UMT5.
    /// Returns the predicted velocity in the same shape as `latent`.
    pub fn forward(
        &self,
        latent: &[f32],
        t: usize,
        h: usize,
        w: usize,
        timestep: f32,
        context: &[f32],
        n_ctx: usize,
    ) -> Vec<f32> {
        let (d, cfg) = (self.cfg.dim, &self.cfg);

        // patch embedding
        let (mut x, tl, hl, wl) = self.patchify(latent, t, h, w);
        let n = tl * hl * wl;

        // timestep -> e -> e0 (six modulation vectors shared by every block)
        let te = timestep_embedding(timestep, cfg.freq_dim);
        let mut e = self.apply(&self.time0, &te, 1);
        for v in e.iter_mut() {
            *v = silu(*v);
        }
        let e = self.apply(&self.time2, &e, 1);
        let e_silu: Vec<f32> = e.iter().map(|v| silu(*v)).collect();
        let e0 = self.apply(&self.time_proj, &e_silu, 1); // [6 * dim]

        // text embedding: 4096 -> dim
        let mut ctx = self.apply(&self.text0, context, n_ctx);
        for v in ctx.iter_mut() {
            *v = gelu(*v);
        }
        let ctx = self.apply(&self.text2, &ctx, n_ctx);

        let pe = wan_rope::wan_pe(t, h, w, cfg.patch.0, cfg.patch.1, cfg.patch.2, &cfg.axes_dim, cfg.theta);

        let mut y = vec![0.0f32; n * d];
        for blk in &self.blocks {
            // Six modulation vectors: block bias plus the shared timestep projection.
            let m: Vec<f32> = (0..6 * d).map(|i| e0[i] + blk.modulation[i]).collect();
            let chunk = |i: usize| &m[i * d..(i + 1) * d];

            // self-attention
            y.copy_from_slice(&x);
            for i in 0..n {
                let row = &mut y[i * d..(i + 1) * d];
                layer_norm(row, cfg.eps);
                for j in 0..d {
                    row[j] = row[j] * (1.0 + chunk(1)[j]) + chunk(0)[j];
                }
            }
            let a = self.attention(&blk.self_attn, &y, &y, n, n, Some(&pe));
            for i in 0..n {
                for j in 0..d {
                    x[i * d + j] += a[i * d + j] * chunk(2)[j];
                }
            }

            // cross-attention into the text embedding
            y.copy_from_slice(&x);
            for i in 0..n {
                layer_norm_affine(&mut y[i * d..(i + 1) * d], &blk.norm3_w, &blk.norm3_b, cfg.eps);
            }
            let c = self.attention(&blk.cross_attn, &y, &ctx, n, n_ctx, None);
            for i in 0..n * d {
                x[i] += c[i];
            }

            // feed-forward
            y.copy_from_slice(&x);
            for i in 0..n {
                let row = &mut y[i * d..(i + 1) * d];
                layer_norm(row, cfg.eps);
                for j in 0..d {
                    row[j] = row[j] * (1.0 + chunk(4)[j]) + chunk(3)[j];
                }
            }
            let mut f = self.apply(&blk.ffn0, &y, n);
            for v in f.iter_mut() {
                *v = gelu(*v);
            }
            let f = self.apply(&blk.ffn2, &f, n);
            for i in 0..n {
                for j in 0..d {
                    x[i * d + j] += f[i * d + j] * chunk(5)[j];
                }
            }
        }

        // head — note this consumes `e`, not `e0`
        let mut hx = x.clone();
        for i in 0..n {
            let row = &mut hx[i * d..(i + 1) * d];
            layer_norm(row, cfg.eps);
            for j in 0..d {
                row[j] = row[j] * (1.0 + e[j] + self.head_mod[d + j]) + (e[j] + self.head_mod[j]);
            }
        }
        let out = self.apply(&self.head, &hx, n);
        self.unpatchify(&out, tl, hl, wl, t, h, w)
    }

    /// Scatter `[n_tokens, out_dim * patch_volume]` back to `[out_dim, t, h, w]`.
    fn unpatchify(&self, tok: &[f32], tl: usize, hl: usize, wl: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
        let (pt, ph, pw) = self.cfg.patch;
        let oc = self.cfg.out_dim;
        let kvol = pt * ph * pw;
        let mut out = vec![0.0f32; oc * t * h * w];
        for ti in 0..tl {
            for hj in 0..hl {
                for wk in 0..wl {
                    let token = (ti * hl + hj) * wl + wk;
                    for c in 0..oc {
                        for kd in 0..pt {
                            for kh in 0..ph {
                                for kw in 0..pw {
                                    // Channel is the *fastest* index inside a token's outputs:
                                    // the reference reshapes as [C, pw*ph*pt, tokens], so a
                                    // token's 64 values are patch-major, channel-minor. Getting
                                    // this the other way round scrambles the frame into noise
                                    // while every individual value still looks plausible.
                                    let src = token * oc * kvol + ((kd * ph + kh) * pw + kw) * oc + c;
                                    let dst = ((c * t + ti * pt + kd) * h + hj * ph + kh) * w + wk * pw + kw;
                                    out[dst] = tok[src];
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_shape_and_bounds() {
        let e = timestep_embedding(1000.0, 256);
        assert_eq!(e.len(), 256);
        assert!(e.iter().all(|v| v.abs() <= 1.0 + 1e-6), "sin/cos must be bounded");
        // freq 0 gives cos(t)=cos(1000), sin(t)=sin(1000)
        assert!((e[0] - 1000f32.cos()).abs() < 1e-4);
        assert!((e[128] - 1000f32.sin()).abs() < 1e-4);
    }

    #[test]
    fn timestep_zero_is_cos_one_sin_zero() {
        let e = timestep_embedding(0.0, 256);
        assert!(e[..128].iter().all(|v| (v - 1.0).abs() < 1e-6), "cos half must be 1");
        assert!(e[128..].iter().all(|v| v.abs() < 1e-6), "sin half must be 0");
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let mut r: Vec<f32> = (0..64).map(|i| i as f32 * 0.3 - 4.0).collect();
        layer_norm(&mut r, 1e-6);
        let mean = r.iter().sum::<f32>() / 64.0;
        let var = r.iter().map(|v| v * v).sum::<f32>() / 64.0;
        assert!(mean.abs() < 1e-4, "mean {mean}");
        assert!((var - 1.0).abs() < 1e-3, "var {var}");
    }

    #[test]
    fn activations_match_known_values() {
        assert!((silu(0.0)).abs() < 1e-6);
        assert!((silu(1.0) - 0.731_058_6).abs() < 1e-5, "silu(1)={}", silu(1.0));
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-4);
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    use crate::gguf::GgufFile;
    use std::path::Path;

    const PACK: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack";
    const REF: &str = "/home/tiny/projects/tinyq4/reference/dit";
    /// The 8MB UMT5 output is not committed; regenerate with SAIENT_DUMP=1 if absent.
    const CTX: &str = "/tmp/saient_ref/cond_crossattn.bin";

    fn read_dump(p: &Path) -> Option<(Vec<i64>, Vec<f32>)> {
        let b = std::fs::read(p).ok()?;
        if &b[..4] != b"SQD1" { return None; }
        let nd = u32::from_le_bytes(b[4..8].try_into().ok()?) as usize;
        let dims: Vec<i64> = (0..nd)
            .map(|i| i64::from_le_bytes(b[8 + i * 8..16 + i * 8].try_into().unwrap()))
            .collect();
        let vals = b[8 + nd * 8..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Some((dims, vals))
    }

    #[test]
    #[ignore = "loads the 816MB DiT and runs 30 blocks over 780 tokens; run explicitly"]
    fn matches_reference_velocity() {
        let gp = Path::new(PACK).join("wan2.1_t2v_1.3B_Q4_K.gguf");
        let (ip, op, tp, cp) = (
            Path::new(REF).join("dit_in_0.bin"),
            Path::new(REF).join("dit_out_0.bin"),
            Path::new(REF).join("dit_t_0.bin"),
            Path::new(CTX),
        );
        for p in [&gp, &ip, &op, &tp] {
            if !p.exists() { eprintln!("skipping: {p:?} missing"); return; }
        }
        if !cp.exists() {
            eprintln!("skipping: {cp:?} missing (regenerate the UMT5 dump with SAIENT_DUMP=1)");
            return;
        }

        let (in_dims, latent) = read_dump(&ip).unwrap();
        let (_, want) = read_dump(&op).unwrap();
        let (_, tvec) = read_dump(&tp).unwrap();
        let (ctx_dims, context) = read_dump(cp).unwrap();
        // ggml order: ne0 fastest. [w, h, t, c] on disk is [c][t][h][w] row-major.
        let (w, h, t, _c) = (
            in_dims[0] as usize, in_dims[1] as usize,
            in_dims[2] as usize, in_dims[3] as usize,
        );
        let n_ctx = ctx_dims[1] as usize;
        eprintln!("latent [c,t,h,w]=[{_c},{t},{h},{w}] timestep={} n_ctx={n_ctx}", tvec[0]);

        let gguf = GgufFile::open(&gp).expect("dit gguf");
        let dit = WanDit::load(&gguf, WanConfig::default()).expect("dit load");
        let start = std::time::Instant::now();
        let got = dit.forward(&latent, t, h, w, tvec[0], &context, n_ctx);
        eprintln!("forward took {:?}", start.elapsed());

        assert_eq!(got.len(), want.len(), "output element count");
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        let mut max_abs = 0.0f32;
        for (g, r) in got.iter().zip(&want) {
            max_abs = max_abs.max((g - r).abs());
            dot += *g as f64 * *r as f64;
            na += *g as f64 * *g as f64;
            nb += *r as f64 * *r as f64;
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        eprintln!("cosine={cos:.8} max_abs={max_abs:.6}");

        // Save our output so the failure can be dissected without paying for another 7-minute run.
        {
            let mut b = Vec::with_capacity(24 + got.len() * 4);
            b.extend_from_slice(b"SQD1");
            b.extend_from_slice(&4u32.to_le_bytes());
            for d in [w as i64, h as i64, t as i64, _c as i64] {
                b.extend_from_slice(&d.to_le_bytes());
            }
            for v in &got { b.extend_from_slice(&v.to_le_bytes()); }
            let _ = std::fs::write("/tmp/saient_ref/dit_out_ours.bin", b);
            eprintln!("  wrote /tmp/saient_ref/dit_out_ours.bin");
        }

        // Where does it agree? A per-channel or per-frame split localises a layout fault; a
        // uniform smear says the maths is wrong rather than the indexing.
        let cosf = |a: &[f32], b: &[f32]| -> f64 {
            let (mut d, mut x, mut y) = (0.0f64, 0.0f64, 0.0f64);
            for (p, q) in a.iter().zip(b) { d += *p as f64 * *q as f64; x += *p as f64 * *p as f64; y += *q as f64 * *q as f64; }
            if x == 0.0 || y == 0.0 { return 0.0; }
            d / (x.sqrt() * y.sqrt())
        };
        let plane = h * w;
        for c in 0..4usize {
            let a = c * t * plane;
            eprintln!("  channel {c}: cosine={:.6}", cosf(&got[a..a + t * plane], &want[a..a + t * plane]));
        }
        for ti in 0..t {
            let a = ti * plane;
            eprintln!("  frame {ti} (ch0): cosine={:.6}", cosf(&got[a..a + plane], &want[a..a + plane]));
        }
        // First latent row of channel 0 — if only the very start matches, ordering is suspect.
        eprintln!("  ch0 row0 (w=52): cosine={:.6}", cosf(&got[..w], &want[..w]));
        eprintln!("  ch0 first 4 patches: got={:?}", &got[..4]);
        eprintln!("  got[..4]  = {:?}", &got[..4]);
        eprintln!("  want[..4] = {:?}", &want[..4]);
        assert!(cos > 0.99, "DiT velocity does not match reference (cosine {cos:.6})");
    }
}
