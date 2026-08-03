//! UMT5-XXL text encoder — the conditioning half of Wan2.1 video generation.
//!
//! Three things here differ from the transformers already in this crate, and getting any of them
//! wrong produces plausible-looking embeddings that quietly steer generation off:
//!
//! 1. **Per-block relative-position bias.** This is UMT5, not T5: every one of the 24 blocks has
//!    its own `attn_rel_b` table. Standard T5 computes the bias once in block 0 and shares it.
//! 2. **No 1/sqrt(head_dim) attention scaling.** T5 folds that into initialisation, so applying
//!    the usual scaling makes every attention distribution too flat.
//! 3. **Gated FFN with the tanh GELU approximation**, not SwiGLU and not plain GELU.
//!
//! Memory: `token_embd` is 256384 x 4096. Dequantising it whole is 4.2 GB in f32, which is the
//! allocation that OOM-killed the Android app. Rows are read out of the quantised data one token
//! at a time instead — 4096 elements is exactly 16 Q6_K blocks, so a row is always block-aligned.

use crate::dequant;
use crate::gguf::{GgufFile, TensorInfo, ggml_type_size};
use anyhow::{Result, anyhow};

pub struct Umt5Config {
    pub n_layers: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub n_buckets: usize,
    pub max_distance: usize,
    pub context: usize,
}

impl Umt5Config {
    pub fn from_gguf(g: &GgufFile) -> Self {
        let u = |k: &str, d: u32| g.meta_u32(k).unwrap_or(d) as usize;
        Self {
            n_layers: u("t5encoder.block_count", 24),
            d_model: u("t5encoder.embedding_length", 4096),
            d_ff: u("t5encoder.feed_forward_length", 10240),
            n_heads: u("t5encoder.attention.head_count", 64),
            head_dim: u("t5encoder.attention.key_length", 64),
            eps: g
                .metadata
                .get("t5encoder.attention.layer_norm_rms_epsilon")
                .and_then(|v| if let crate::gguf::GgufValue::Float32(f) = v { Some(*f) } else { None })
                .unwrap_or(1e-6),
            n_buckets: u("t5encoder.attention.relative_buckets_count", 32),
            max_distance: 128,
            context: u("t5encoder.context_length", 512),
        }
    }
}

/// T5 bidirectional relative-position bucketing.
///
/// Nearby positions get their own bucket each; distant ones are binned logarithmically. Sign is
/// encoded by offsetting into the upper half of the table, which is why the effective bucket count
/// per direction is `n_buckets / 2`.
pub fn relative_bucket(rel: i32, n_buckets: usize, max_distance: usize) -> usize {
    let half = n_buckets / 2;
    let mut ret = if rel > 0 { half } else { 0 };
    let n = rel.unsigned_abs() as usize;
    let max_exact = half / 2;
    if n < max_exact {
        ret + n
    } else {
        let ln_ratio = (n as f32 / max_exact as f32).ln();
        let ln_span = (max_distance as f32 / max_exact as f32).ln();
        let large = max_exact + (ln_ratio / ln_span * (half - max_exact) as f32) as usize;
        ret += large.min(half - 1);
        ret
    }
}

/// T5 / GPT-2 "new" GELU — the tanh approximation, not the erf form.
#[inline]
fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_56; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn rms_norm_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] / rms * w[i];
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

struct Block<'a> {
    attn_norm: Vec<f32>,
    q: &'a TensorInfo,
    k: &'a TensorInfo,
    v: &'a TensorInfo,
    o: &'a TensorInfo,
    /// `[n_buckets][n_heads]`, i.e. indexed `bucket * n_heads + head`. Upstream builds it as
    /// `Embedding(num_buckets, num_heads)` and the GGUF dims are `[64, 32]` with ne0 fastest,
    /// so it is 32 rows of 64 — transposing this produces an encoder that runs happily and
    /// correlates with nothing.
    rel_b: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate: &'a TensorInfo,
    up: &'a TensorInfo,
    down: &'a TensorInfo,
}

pub struct Umt5Encoder<'a> {
    gguf: &'a GgufFile,
    pub cfg: Umt5Config,
    embd: &'a TensorInfo,
    blocks: Vec<Block<'a>>,
    out_norm: Vec<f32>,
}

impl<'a> Umt5Encoder<'a> {
    pub fn load(gguf: &'a GgufFile) -> Result<Self> {
        let cfg = Umt5Config::from_gguf(gguf);
        let map = gguf.tensor_map();
        let get = |n: &str| -> Result<&'a TensorInfo> {
            map.get(n).copied().ok_or_else(|| anyhow!("missing tensor {n}"))
        };
        let f32_of = |t: &TensorInfo| -> Vec<f32> {
            dequant::dequant(gguf.tensor_data(t), t.ggml_type, t.n_elems())
        };

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("enc.blk.{i}");
            blocks.push(Block {
                attn_norm: f32_of(get(&format!("{p}.attn_norm.weight"))?),
                q: get(&format!("{p}.attn_q.weight"))?,
                k: get(&format!("{p}.attn_k.weight"))?,
                v: get(&format!("{p}.attn_v.weight"))?,
                o: get(&format!("{p}.attn_o.weight"))?,
                rel_b: f32_of(get(&format!("{p}.attn_rel_b.weight"))?),
                ffn_norm: f32_of(get(&format!("{p}.ffn_norm.weight"))?),
                gate: get(&format!("{p}.ffn_gate.weight"))?,
                up: get(&format!("{p}.ffn_up.weight"))?,
                down: get(&format!("{p}.ffn_down.weight"))?,
            });
        }

        Ok(Self {
            gguf,
            embd: get("token_embd.weight")?,
            out_norm: f32_of(get("enc.output_norm.weight")?),
            blocks,
            cfg,
        })
    }

    /// Look one embedding row out of the quantised weights.
    ///
    /// Deliberately never dequantises the whole table — see the note at the top of this file.
    fn embed_row(&self, id: u32, out: &mut [f32]) {
        let d = self.cfg.d_model;
        let t = self.embd;
        let data = self.gguf.tensor_data(t);
        // Row-aligned addressing is only valid if a row is a whole number of quant blocks.
        // d_model is 4096 and K-quants are 256 per block, so this holds — assert rather than
        // assume, because a silent misalignment here reads garbage embeddings.
        let row_bytes = ggml_type_size(t.ggml_type, d);
        debug_assert_eq!(
            ggml_type_size(t.ggml_type, d * 2), row_bytes * 2,
            "embedding row is not block-aligned for ggml_type {}", t.ggml_type
        );
        let start = id as usize * row_bytes;
        let row = &data[start..start + row_bytes];
        let vals = dequant::dequant(row, t.ggml_type, d);
        out.copy_from_slice(&vals);
    }

    /// Encode a token sequence to `[seq, d_model]`, row-major.
    ///
    /// `n_valid` is the number of real tokens; everything from there to `ids.len()` is padding.
    /// Padding is masked out of attention *and* zeroed in the output, which is what the reference
    /// does — a padded position there is exactly 0.0, not "whatever the encoder happened to
    /// produce". Skipping the zeroing leaves 510 of 512 positions disagreeing on a 512-token
    /// context and makes the whole comparison meaningless.
    pub fn forward(&self, ids: &[u32], n_valid: usize) -> Vec<f32> {
        let (d, h, hd) = (self.cfg.d_model, self.cfg.n_heads, self.cfg.head_dim);
        let total = ids.len();
        // Padding is masked out of attention and zeroed in the output, so a valid position's
        // result depends only on the other valid positions. Running the stack over just the real
        // tokens is therefore exact, not an approximation — and it takes a 512-token context down
        // to the handful actually used, which is the difference between minutes and milliseconds.
        let n = n_valid.min(total);
        let ids = &ids[..n];

        // Token embeddings, one quantised row at a time.
        let mut x = vec![0.0f32; n * d];
        for (i, &id) in ids.iter().enumerate() {
            self.embed_row(id, &mut x[i * d..(i + 1) * d]);
        }

        // Bucket indices depend only on positions, so they are computed once and reused by every
        // block — it is the bias *values* that are per-block, not the bucketing.
        let mut buckets = vec![0usize; n * n];
        for qi in 0..n {
            for kj in 0..n {
                buckets[qi * n + kj] =
                    relative_bucket(kj as i32 - qi as i32, self.cfg.n_buckets, self.cfg.max_distance);
            }
        }

        let mut normed = vec![0.0f32; n * d];
        let mut scores = vec![0.0f32; n];
        let mut ctx = vec![0.0f32; n * d];

        for blk in &self.blocks {
            // ── self-attention ───────────────────────────────────────────────
            for i in 0..n {
                rms_norm_into(&x[i * d..(i + 1) * d], &blk.attn_norm, self.cfg.eps, &mut normed[i * d..(i + 1) * d]);
            }
            let q = self.mat(&normed, blk.q, d, d, n);
            let k = self.mat(&normed, blk.k, d, d, n);
            let v = self.mat(&normed, blk.v, d, d, n);

            ctx.fill(0.0);
            for head in 0..h {
                let off = head * hd;
                for qi in 0..n {
                    // T5 applies no 1/sqrt(head_dim) scaling here — see the note at the top.
                    for kj in 0..n {
                        let mut dot = 0.0;
                        for t in 0..hd {
                            dot += q[qi * d + off + t] * k[kj * d + off + t];
                        }
                        scores[kj] = dot + blk.rel_b[buckets[qi * n + kj] * h + head];
                    }
                    // Padding must not contribute to any real token's context.
                    softmax(&mut scores);
                    for kj in 0..n {
                        let w = scores[kj];
                        if w == 0.0 {
                            continue;
                        }
                        for t in 0..hd {
                            ctx[qi * d + off + t] += w * v[kj * d + off + t];
                        }
                    }
                }
            }
            let attn_out = self.mat(&ctx, blk.o, d, d, n);
            for i in 0..n * d {
                x[i] += attn_out[i];
            }

            // ── gated feed-forward ───────────────────────────────────────────
            for i in 0..n {
                rms_norm_into(&x[i * d..(i + 1) * d], &blk.ffn_norm, self.cfg.eps, &mut normed[i * d..(i + 1) * d]);
            }
            let mut gate = self.mat(&normed, blk.gate, d, self.cfg.d_ff, n);
            let up = self.mat(&normed, blk.up, d, self.cfg.d_ff, n);
            for i in 0..gate.len() {
                gate[i] = gelu(gate[i]) * up[i];
            }
            let ff = self.mat(&gate, blk.down, self.cfg.d_ff, d, n);
            for i in 0..n * d {
                x[i] += ff[i];
            }
        }

        // Only valid positions are normalised and emitted; padded ones stay zero.
        let mut out = vec![0.0f32; total * d];
        for i in 0..n {
            rms_norm_into(&x[i * d..(i + 1) * d], &self.out_norm, self.cfg.eps, &mut out[i * d..(i + 1) * d]);
        }
        out
    }

    /// `[n, in_dim] @ W[in_dim, out_dim] -> [n, out_dim]`, reading W straight from quantised data.
    fn mat(&self, x: &[f32], t: &TensorInfo, in_dim: usize, out_dim: usize, n: usize) -> Vec<f32> {
        dequant::gemm(x, self.gguf.tensor_data(t), t.ggml_type, in_dim, out_dim, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucketing_places_near_positions_in_their_own_bucket() {
        // Bidirectional: 32 buckets total, 16 per direction, first 8 of each are exact.
        assert_eq!(relative_bucket(0, 32, 128), 0);
        assert_eq!(relative_bucket(1, 32, 128), 16 + 1, "positive offsets sit in the upper half");
        assert_eq!(relative_bucket(-1, 32, 128), 1);
        assert_eq!(relative_bucket(7, 32, 128), 16 + 7);
        assert_eq!(relative_bucket(-7, 32, 128), 7);
    }

    #[test]
    fn bucketing_is_logarithmic_beyond_the_exact_range() {
        // Distant positions must compress into the remaining buckets, never run off the end.
        for rel in [8, 16, 64, 127, 128, 1000, 100_000] {
            let b = relative_bucket(rel, 32, 128);
            assert!((16..32).contains(&b), "rel {rel} -> bucket {b} out of the positive half");
            let b = relative_bucket(-rel, 32, 128);
            assert!((0..16).contains(&b), "rel -{rel} -> bucket {b} out of the negative half");
        }
    }

    #[test]
    fn bucketing_is_monotonic_in_distance() {
        let mut prev = 0;
        for rel in 0..2000 {
            let b = relative_bucket(rel, 32, 128);
            assert!(b >= prev, "bucket must not decrease as distance grows ({rel})");
            prev = b;
        }
    }

    #[test]
    fn gelu_matches_known_values() {
        // tanh-approximation GELU reference points.
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-4, "gelu(1) = {}", gelu(1.0));
        assert!((gelu(-1.0) + 0.158_808).abs() < 1e-4, "gelu(-1) = {}", gelu(-1.0));
        assert!(gelu(10.0) > 9.99);
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::t5_tokenizer::T5Tokenizer;
    use std::path::Path;

    const PACK: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack";
    const REF_DIR: &str = "/tmp/saient_ref";

    /// Reference dumps are `SQD1` + u32 ndim + i64 dims + f32 data, ggml order (ne0 fastest),
    /// so `[4096, 512, 1]` on disk is `[seq][d_model]` row-major — the same layout `forward`
    /// returns.
    fn read_dump(p: &Path) -> Option<(Vec<i64>, Vec<f32>)> {
        let b = std::fs::read(p).ok()?;
        if &b[..4] != b"SQD1" { return None; }
        let nd = u32::from_le_bytes(b[4..8].try_into().ok()?) as usize;
        let mut dims = Vec::with_capacity(nd);
        for i in 0..nd {
            let o = 8 + i * 8;
            dims.push(i64::from_le_bytes(b[o..o + 8].try_into().ok()?));
        }
        let off = 8 + nd * 8;
        let vals: Vec<f32> = b[off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Some((dims, vals))
    }

    fn compare(tag: &str, prompt: &str, dump: &str) {
        let gp = Path::new(PACK).join("umt5-xxl-encoder-Q4_K_M.gguf");
        let rp = Path::new(REF_DIR).join(dump);
        if !gp.exists() || !rp.exists() {
            eprintln!("skipping {tag}: model or reference dump missing");
            return;
        }
        let t0 = std::time::Instant::now();
        let (dims, want) = read_dump(&rp).expect("reference dump unreadable");
        eprintln!("  [t] read_dump {:?}", t0.elapsed());
        let t1 = std::time::Instant::now();
        let gguf = GgufFile::open(&gp).expect("gguf");
        eprintln!("  [t] gguf_open {:?}", t1.elapsed());
        let t2 = std::time::Instant::now();
        let tk = T5Tokenizer::from_gguf(&gguf).expect("tokenizer");
        eprintln!("  [t] tokenizer {:?}", t2.elapsed());
        let t3 = std::time::Instant::now();
        let enc = Umt5Encoder::load(&gguf).expect("encoder");
        eprintln!("  [t] enc_load {:?}", t3.elapsed());

        let d = dims[0] as usize;
        let seq = dims[1] as usize;
        let n_valid = tk.encode(prompt).len();
        let ids = tk.encode_padded(prompt, seq);
        let t4 = std::time::Instant::now();
        let got = enc.forward(&ids, n_valid);
        eprintln!("  [t] forward {:?}", t4.elapsed());
        assert_eq!(got.len(), want.len(), "{tag}: element count");

        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b.iter()) {
                dot += *x as f64 * *y as f64;
                na += *x as f64 * *x as f64;
                nb += *y as f64 * *y as f64;
            }
            if na == 0.0 || nb == 0.0 { return if na == nb { 1.0 } else { 0.0 }; }
            dot / (na.sqrt() * nb.sqrt())
        };

        eprintln!("{tag}: n_valid={n_valid} seq={seq} d={d}");
        // Per-position, so a core that is right but mis-assembled is distinguishable from a core
        // that is simply wrong.
        for i in 0..n_valid.min(6) {
            let (g, w) = (&got[i * d..(i + 1) * d], &want[i * d..(i + 1) * d]);
            let md = g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            eprintln!("  pos {i}: cosine={:.6} max_abs={md:.6}", cos(g, w));
        }
        let pads_zero = want[n_valid * d..].iter().all(|&v| v == 0.0);
        let ours_zero = got[n_valid * d..].iter().all(|&v| v == 0.0);
        eprintln!("  padding zeroed: reference={pads_zero} ours={ours_zero}");

        let valid = n_valid * d;
        let c_valid = cos(&got[..valid], &want[..valid]);
        let c_all = cos(&got, &want);
        let max_abs = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("  cosine(valid)={c_valid:.8} cosine(all)={c_all:.8} max_abs={max_abs:.6}");
        eprintln!("  got[..4]  = {:?}", &got[..4]);
        eprintln!("  want[..4] = {:?}", &want[..4]);

        assert!(c_valid > 0.999, "{tag}: cosine over valid positions {c_valid:.6}");
        assert!(max_abs < 0.05, "{tag}: max abs diff {max_abs:.6}");
    }

    #[test]
    #[ignore = "loads a 3.6GB encoder and runs 24 blocks over 512 tokens; run explicitly"]
    fn cond_matches_reference() {
        compare("cond", "a red fox", "cond_crossattn.bin");
    }

    #[test]
    #[ignore = "loads a 3.6GB encoder and runs 24 blocks over 512 tokens; run explicitly"]
    fn uncond_matches_reference() {
        compare("uncond", "", "uncond_crossattn.bin");
    }
}
