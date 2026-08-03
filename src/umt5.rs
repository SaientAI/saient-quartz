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

use crate::backend::{DeviceTensor, TensorBackend};
use crate::dequant;
use crate::gguf::{GgufFile, TensorInfo, ggml_type_size};
use crate::tensor::Tensor;
use anyhow::{Context, Result, anyhow, bail};

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
                .and_then(|v| {
                    if let crate::gguf::GgufValue::Float32(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
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
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    // The factored form is what matters here, not the constant: this mirrors ggml-cuda's
    // `0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)))` operation order.
    // `C*x*(1 + a*x^2)` and `C*(x + a*x^3)` are algebraically equal but round differently in
    // f32, and the FFN's subsequent Q8 activation quantization makes that difference visible.
    // (The literal itself is not load-bearing — ggml spells it to full precision and every
    // spelling collapses to the same f32, 0x3f4c422a.)
    0.5 * x * (1.0 + (C * x * (1.0 + 0.044715 * x * x)).tanh())
}

#[cfg(test)]
fn gelu_via_exp(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    let argument = C * x * (1.0 + 0.044715 * x * x);
    x / (1.0 + (-2.0 * argument).exp())
}

#[cfg(test)]
fn gelu_via_exp2(x: f32) -> f32 {
    const C: f32 = 0.797_884_6;
    const LOG2_E: f32 = std::f32::consts::LOG2_E;
    let argument = C * x * (1.0 + 0.044715 * x * x);
    x / (1.0 + (-2.0 * argument * LOG2_E).exp2())
}

#[inline]
fn rms_norm_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    if x.len() == 4096 && std::env::var_os("QUARTZ_UMT5_CUDA_NORM").is_some() {
        rms_norm_cuda_into(x, w, eps, out);
        return;
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] / rms * w[i];
    }
}

/// CPU oracle for the 1024-thread CUDA RMSNorm reduction used by UMT5's 4096-wide rows.
fn rms_norm_cuda_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), 4096);
    assert_eq!(w.len(), x.len());
    assert_eq!(out.len(), x.len());

    let mut partials = [0.0f32; 1024];
    for thread in 0..1024 {
        let mut sum = 0.0f32;
        for column in (thread..x.len()).step_by(1024) {
            sum = x[column].mul_add(x[column], sum);
        }
        partials[thread] = sum;
    }
    for warp in partials.chunks_exact_mut(32) {
        for offset in [16, 8, 4, 2, 1] {
            let previous = warp.to_vec();
            for lane in 0..32 {
                warp[lane] = previous[lane] + previous[lane ^ offset];
            }
        }
    }
    let mut warp_sums = [0.0f32; 32];
    for (warp, sum) in warp_sums.iter_mut().enumerate() {
        *sum = partials[warp * 32];
    }
    for offset in [16, 8, 4, 2, 1] {
        let previous = warp_sums;
        for lane in 0..32 {
            warp_sums[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    let scale = (warp_sums[0] / x.len() as f32 + eps).sqrt().recip();
    for index in 0..x.len() {
        out[index] = (scale * x[index]) * w[index];
    }
}

fn softmax(x: &mut [f32]) {
    if x.len() <= 512 && std::env::var_os("QUARTZ_UMT5_CUDA_SOFTMAX_512").is_some() {
        softmax_cuda_512(x);
        return;
    }
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

fn softmax_cuda_512(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut lanes = [0.0f32; 512];
    for (output, &value) in lanes.iter_mut().zip(x.iter()) {
        *output = (value - max).exp();
    }
    for warp in lanes.chunks_exact_mut(32) {
        for offset in [16, 8, 4, 2, 1] {
            let previous: [f32; 32] = (*warp).try_into().unwrap();
            for lane in 0..32 {
                warp[lane] = previous[lane] + previous[lane ^ offset];
            }
        }
    }
    let mut warp_sums = [0.0f32; 32];
    for (warp, output) in warp_sums.iter_mut().take(16).enumerate() {
        *output = lanes[warp * 32];
    }
    for offset in [16, 8, 4, 2, 1] {
        let previous = warp_sums;
        for lane in 0..32 {
            warp_sums[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    let inverse = warp_sums[0].recip();
    for output in x {
        *output = (*output - max).exp() * inverse;
    }
}

fn round_to_tf32(value: f32) -> f32 {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return value;
    }
    let rounded = bits.wrapping_add(0x0fff + ((bits >> 13) & 1));
    f32::from_bits(rounded & 0xffff_e000)
}

fn attention_dot(query: &[f32], key: &[f32], tf32: bool, chunks8: bool) -> f32 {
    debug_assert_eq!(query.len(), key.len());
    if tf32 && chunks8 {
        let mut sum = 0.0f32;
        for (query_chunk, key_chunk) in query.chunks(8).zip(key.chunks(8)) {
            let mut chunk = 0.0f32;
            for (&query, &key) in query_chunk.iter().zip(key_chunk) {
                chunk = round_to_tf32(query).mul_add(round_to_tf32(key), chunk);
            }
            sum += chunk;
        }
        return sum;
    }
    let mut sum = 0.0f32;
    for (&query, &key) in query.iter().zip(key) {
        sum = if tf32 {
            round_to_tf32(query).mul_add(round_to_tf32(key), sum)
        } else {
            query * key + sum
        };
    }
    sum
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Umt5TracePoint {
    Prelude,
    PostAttention(usize),
    PostBlock(usize),
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
            map.get(n)
                .copied()
                .ok_or_else(|| anyhow!("missing tensor {n}"))
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
            ggml_type_size(t.ggml_type, d * 2),
            row_bytes * 2,
            "embedding row is not block-aligned for ggml_type {}",
            t.ggml_type
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
        self.forward_traced(ids, n_valid, self.blocks.len(), |_, _| {})
    }

    fn forward_traced(
        &self,
        ids: &[u32],
        n_valid: usize,
        block_limit: usize,
        mut trace: impl FnMut(Umt5TracePoint, &[f32]),
    ) -> Vec<f32> {
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
        trace(Umt5TracePoint::Prelude, &x);

        // Bucket indices depend only on positions, so they are computed once and reused by every
        // block — it is the bias *values* that are per-block, not the bucketing.
        let mut buckets = vec![0usize; n * n];
        for qi in 0..n {
            for kj in 0..n {
                buckets[qi * n + kj] = relative_bucket(
                    kj as i32 - qi as i32,
                    self.cfg.n_buckets,
                    self.cfg.max_distance,
                );
            }
        }

        let mut normed = vec![0.0f32; n * d];
        let mut scores = vec![0.0f32; n];
        let mut ctx = vec![0.0f32; n * d];
        // Diagnostic only: emulates the reference's tf32 attention rounding so the residual
        // encoder delta can be attributed. Unset means the ordinary f32 path. It is announced
        // when active so a stray environment variable can never quietly change the numerics.
        let tf32_attention_mode = std::env::var("QUARTZ_UMT5_TF32_ATTENTION").ok();
        if let Some(mode) = tf32_attention_mode.as_deref() {
            eprintln!("UMT5 DIAGNOSTIC: tf32 attention emulation active (mode={mode:?})");
        }
        let tf32_attention = tf32_attention_mode.is_some();
        let tf32_attention_chunks8 = tf32_attention_mode.as_deref() == Some("chunks8");

        for (block_index, blk) in self.blocks.iter().take(block_limit).enumerate() {
            // ── self-attention ───────────────────────────────────────────────
            for i in 0..n {
                rms_norm_into(
                    &x[i * d..(i + 1) * d],
                    &blk.attn_norm,
                    self.cfg.eps,
                    &mut normed[i * d..(i + 1) * d],
                );
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
                        let dot = attention_dot(
                            &q[qi * d + off..qi * d + off + hd],
                            &k[kj * d + off..kj * d + off + hd],
                            tf32_attention,
                            tf32_attention_chunks8,
                        );
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
                            let output = &mut ctx[qi * d + off + t];
                            let value = v[kj * d + off + t];
                            *output = if tf32_attention {
                                round_to_tf32(w).mul_add(round_to_tf32(value), *output)
                            } else {
                                w * value + *output
                            };
                        }
                    }
                }
            }
            let attn_out = self.mat(&ctx, blk.o, d, d, n);
            for i in 0..n * d {
                x[i] += attn_out[i];
            }
            trace(Umt5TracePoint::PostAttention(block_index), &x);

            // ── gated feed-forward ───────────────────────────────────────────
            for i in 0..n {
                rms_norm_into(
                    &x[i * d..(i + 1) * d],
                    &blk.ffn_norm,
                    self.cfg.eps,
                    &mut normed[i * d..(i + 1) * d],
                );
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
            trace(Umt5TracePoint::PostBlock(block_index), &x);
        }

        // Only valid positions are normalised and emitted; padded ones stay zero.
        let mut out = vec![0.0f32; total * d];
        for i in 0..n {
            rms_norm_into(
                &x[i * d..(i + 1) * d],
                &self.out_norm,
                self.cfg.eps,
                &mut out[i * d..(i + 1) * d],
            );
        }
        out
    }

    /// Encode through a backend while keeping every activation resident between the initial
    /// embedding upload and the final context download.
    ///
    /// UMT5-XXL's weights are deliberately prepared one projection at a time. A block contains
    /// roughly 386 MB of FP16 projection weights, but no operation needs all seven matrices at
    /// once. Dropping each prepared handle after its dispatch bounds device-local weight residency
    /// without changing the canonical graph.
    pub(crate) fn forward_with_backend(
        &self,
        backend: &dyn TensorBackend,
        ids: &[u32],
        n_valid: usize,
    ) -> Result<Vec<f32>> {
        let (d, heads, head_dim) = (self.cfg.d_model, self.cfg.n_heads, self.cfg.head_dim);
        let total = ids.len();
        let n = n_valid.min(total);
        if n == 0 {
            bail!("UMT5 backend execution requires at least one valid token");
        }
        if heads.checked_mul(head_dim) != Some(d) {
            bail!("UMT5 head dimensions do not reconstruct the model width");
        }

        let mut embeddings = vec![0.0f32; n * d];
        for (row, &id) in ids[..n].iter().enumerate() {
            self.embed_row(id, &mut embeddings[row * d..(row + 1) * d]);
        }
        let mut state = backend.upload_tensor(&Tensor::new(vec![n, d], embeddings)?)?;

        let mut buckets = vec![0usize; n * n];
        for query in 0..n {
            for key in 0..n {
                buckets[query * n + key] = relative_bucket(
                    key as i32 - query as i32,
                    self.cfg.n_buckets,
                    self.cfg.max_distance,
                );
            }
        }

        for block in &self.blocks {
            let attn_norm =
                backend.prepare_vector(&Tensor::new(vec![d], block.attn_norm.clone())?)?;
            let normalized = backend.rms_norm_device(&state, &attn_norm, self.cfg.eps)?;
            drop(attn_norm);

            let query = self.linear_with_backend(backend, &normalized, block.q, d, d)?;
            let key = self.linear_with_backend(backend, &normalized, block.k, d, d)?;
            let value = self.linear_with_backend(backend, &normalized, block.v, d, d)?;
            drop(normalized);

            // The attention backend returns [head, query, key]. UMT5 stores its learned
            // relative table as [bucket, head], so construct the resident bias in exactly that
            // order rather than relying on a broadcast or transpose convention.
            let mut relative_bias = vec![0.0f32; heads * n * n];
            for head in 0..heads {
                for query_row in 0..n {
                    for key_row in 0..n {
                        relative_bias[(head * n + query_row) * n + key_row] =
                            block.rel_b[buckets[query_row * n + key_row] * heads + head];
                    }
                }
            }
            let relative_bias =
                backend.upload_tensor(&Tensor::new(vec![heads, n, n], relative_bias)?)?;
            let scores = backend.attention_scores_device(&query, &key, heads, head_dim, 1.0)?;
            drop(query);
            drop(key);
            let scores = backend.add_device(&scores, &relative_bias)?;
            drop(relative_bias);
            let probabilities = backend.softmax_device(&scores)?;
            let context =
                backend.attention_values_device(&probabilities, &value, heads, head_dim)?;
            drop(probabilities);
            drop(value);
            let attention = self.linear_with_backend(backend, &context, block.o, d, d)?;
            drop(context);
            state = backend.add_device(&state, &attention)?;
            drop(attention);

            let ffn_norm =
                backend.prepare_vector(&Tensor::new(vec![d], block.ffn_norm.clone())?)?;
            let normalized = backend.rms_norm_device(&state, &ffn_norm, self.cfg.eps)?;
            drop(ffn_norm);
            let gate =
                self.linear_with_backend(backend, &normalized, block.gate, d, self.cfg.d_ff)?;
            let gate = backend.gelu_tanh_device(&gate)?;
            let up = self.linear_with_backend(backend, &normalized, block.up, d, self.cfg.d_ff)?;
            drop(normalized);
            let gated = backend.multiply_device(&gate, &up)?;
            drop(gate);
            drop(up);

            // stable-diffusion.cpp scales this projection's input by 1/32 and restores the
            // output by 32. Both factors are exact powers of two; preserving the sequence keeps
            // the backend graph aligned with the captured CUDA reference's overflow guard.
            let gated = backend.scale_device(&gated, 1.0 / 32.0)?;
            let feed_forward =
                self.linear_with_backend(backend, &gated, block.down, self.cfg.d_ff, d)?;
            drop(gated);
            let feed_forward = backend.scale_device(&feed_forward, 32.0)?;
            state = backend.add_device(&state, &feed_forward)?;
            drop(feed_forward);
        }

        let output_norm = backend.prepare_vector(&Tensor::new(vec![d], self.out_norm.clone())?)?;
        let state = backend.rms_norm_device(&state, &output_norm, self.cfg.eps)?;
        drop(output_norm);
        let state = backend.download_tensor(&state)?;
        if state.shape() != [n, d] {
            bail!(
                "UMT5 backend returned shape {:?}, expected [{n}, {d}]",
                state.shape()
            );
        }
        let mut output = vec![0.0; total * d];
        output[..n * d].copy_from_slice(state.data());
        Ok(output)
    }

    fn linear_with_backend(
        &self,
        backend: &dyn TensorBackend,
        input: &DeviceTensor,
        info: &TensorInfo,
        input_width: usize,
        output_width: usize,
    ) -> Result<DeviceTensor> {
        if info.n_elems() != input_width.saturating_mul(output_width) {
            bail!(
                "UMT5 tensor {} has {} values, expected {}x{}",
                info.name,
                info.n_elems(),
                output_width,
                input_width
            );
        }
        let weight = Tensor::new(
            vec![output_width, input_width],
            dequant::dequant(self.gguf.tensor_data(info), info.ggml_type, info.n_elems()),
        )
        .with_context(|| format!("materializing UMT5 tensor {}", info.name))?;
        let prepared = backend
            .prepare_linear(&weight, None)
            .with_context(|| format!("preparing UMT5 tensor {}", info.name))?;
        backend
            .linear_prepared(input, &prepared)
            .with_context(|| format!("executing UMT5 tensor {}", info.name))
    }

    /// `[n, in_dim] @ W[in_dim, out_dim] -> [n, out_dim]`, reading W straight from quantised data.
    fn mat(&self, x: &[f32], t: &TensorInfo, in_dim: usize, out_dim: usize, n: usize) -> Vec<f32> {
        let data = self.gguf.tensor_data(t);
        let mode = std::env::var("QUARTZ_UMT5_MATMUL").unwrap_or_default();
        let is_qkv = t.name.contains(".attn_q.")
            || t.name.contains(".attn_k.")
            || t.name.contains(".attn_v.");
        let is_attention = is_qkv || t.name.contains(".attn_o.");
        let is_ffn = t.name.contains(".ffn_");
        // Diagnostic oracle for the pinned 36-SM CUDA capture device. Stream-K partitioning is
        // device-dependent, so keep this opt-in and explicit rather than changing model semantics.
        let use_stream_k = matches!(
            mode.as_str(),
            "cuda_stream_k_36" | "cuda_stream_k_36_attention" | "cuda_stream_k_36_except_q6_down"
        );
        let use_mmq = match mode.as_str() {
            "all" | "cuda_stream_k_36" => true,
            "cuda_stream_k_36_attention" => is_attention,
            "cuda_stream_k_36_except_q6_down" => !t.name.contains(".ffn_down."),
            "qkv" => is_qkv,
            "attention" => is_attention,
            "ffn" => is_ffn,
            "q4" => t.ggml_type == 12,
            "q6" => t.ggml_type == 14,
            _ => false,
        };
        match (use_mmq, use_stream_k, t.ggml_type) {
            // The pinned CUDA reference executes K-quants through MMQ: activations are first
            // quantized to Q8_1, then integer dots are scaled using FP16-rounded metadata. Using
            // a dequantized FP32 dot is mathematically cleaner but measurably different after 24
            // UMT5 blocks, so reproduce the actual reference execution path here.
            (true, true, 12) => dequant::gemm_q4k_q8_1_mmq_stream_k(
                x,
                data,
                in_dim,
                out_dim,
                n,
                self.cfg.context,
                36,
            ),
            (true, true, 14) => dequant::gemm_q6k_q8_1_mmq_stream_k(
                x,
                data,
                in_dim,
                out_dim,
                n,
                self.cfg.context,
                36,
            ),
            (true, false, 12) => dequant::gemm_q4k_q8_1_mmq(x, data, in_dim, out_dim, n),
            (true, false, 14) => dequant::gemm_q6k_q8_1_mmq(x, data, in_dim, out_dim, n),
            _ => dequant::gemm(x, data, t.ggml_type, in_dim, out_dim, n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucketing_places_near_positions_in_their_own_bucket() {
        // Bidirectional: 32 buckets total, 16 per direction, first 8 of each are exact.
        assert_eq!(relative_bucket(0, 32, 128), 0);
        assert_eq!(
            relative_bucket(1, 32, 128),
            16 + 1,
            "positive offsets sit in the upper half"
        );
        assert_eq!(relative_bucket(-1, 32, 128), 1);
        assert_eq!(relative_bucket(7, 32, 128), 16 + 7);
        assert_eq!(relative_bucket(-7, 32, 128), 7);
    }

    #[test]
    fn bucketing_is_logarithmic_beyond_the_exact_range() {
        // Distant positions must compress into the remaining buckets, never run off the end.
        for rel in [8, 16, 64, 127, 128, 1000, 100_000] {
            let b = relative_bucket(rel, 32, 128);
            assert!(
                (16..32).contains(&b),
                "rel {rel} -> bucket {b} out of the positive half"
            );
            let b = relative_bucket(-rel, 32, 128);
            assert!(
                (0..16).contains(&b),
                "rel -{rel} -> bucket {b} out of the negative half"
            );
        }
    }

    #[test]
    fn bucketing_is_monotonic_in_distance() {
        let mut prev = 0;
        for rel in 0..2000 {
            let b = relative_bucket(rel, 32, 128);
            assert!(
                b >= prev,
                "bucket must not decrease as distance grows ({rel})"
            );
            prev = b;
        }
    }

    #[test]
    fn gelu_matches_known_values() {
        // tanh-approximation GELU reference points.
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!(
            (gelu(1.0) - 0.841_192).abs() < 1e-4,
            "gelu(1) = {}",
            gelu(1.0)
        );
        assert!(
            (gelu(-1.0) + 0.158_808).abs() < 1e-4,
            "gelu(-1) = {}",
            gelu(-1.0)
        );
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

    /// Opt-in block-0 traces from the instrumented CUDA runner. These are diagnostics, not parity
    /// claims, and are regenerated on demand — so they may legitimately be absent.
    const REF_DIR: &str = "/tmp/saient_ref";

    /// Committed encoder-output fixtures. A parity test must never pass because its reference
    /// went missing, so these live in the repository rather than in `/tmp`. 507 of the 512 rows
    /// are zero padding, so both blobs compress to ~120 KB in git.
    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/t5");

    /// Reference dumps are `SQD1` + u32 ndim + i64 dims + f32 data, ggml order (ne0 fastest),
    /// so `[4096, 512, 1]` on disk is `[seq][d_model]` row-major — the same layout `forward`
    /// returns.
    fn read_dump(p: &Path) -> Option<(Vec<i64>, Vec<f32>)> {
        let b = std::fs::read(p).ok()?;
        if &b[..4] != b"SQD1" {
            return None;
        }
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

    struct Block0AttentionTrace {
        input: Vec<f32>,
        norm: Vec<f32>,
        query: Vec<f32>,
        key: Vec<f32>,
        value: Vec<f32>,
        scores_biased: Vec<f32>,
        probabilities: Vec<f32>,
        context: Vec<f32>,
        projected: Vec<f32>,
        post_attention: Vec<f32>,
    }

    fn scalar_block0_attention_trace(
        encoder: &Umt5Encoder<'_>,
        ids: &[u32],
        n: usize,
    ) -> Block0AttentionTrace {
        let (d, heads, head_dim) = (
            encoder.cfg.d_model,
            encoder.cfg.n_heads,
            encoder.cfg.head_dim,
        );
        let block = &encoder.blocks[0];
        let mut input = vec![0.0; n * d];
        for (row, &id) in ids[..n].iter().enumerate() {
            encoder.embed_row(id, &mut input[row * d..(row + 1) * d]);
        }
        let mut norm = vec![0.0; n * d];
        for row in 0..n {
            rms_norm_into(
                &input[row * d..(row + 1) * d],
                &block.attn_norm,
                encoder.cfg.eps,
                &mut norm[row * d..(row + 1) * d],
            );
        }
        let query = encoder.mat(&norm, block.q, d, d, n);
        let key = encoder.mat(&norm, block.k, d, d, n);
        let value = encoder.mat(&norm, block.v, d, d, n);
        let mut scores = vec![0.0; n];
        let mut scores_biased = vec![0.0; heads * n * n];
        let mut probabilities = vec![0.0; heads * n * n];
        let mut context = vec![0.0; n * d];
        // Diagnostic only: emulates the reference's tf32 attention rounding so the residual
        // encoder delta can be attributed. Unset means the ordinary f32 path. It is announced
        // when active so a stray environment variable can never quietly change the numerics.
        let tf32_attention_mode = std::env::var("QUARTZ_UMT5_TF32_ATTENTION").ok();
        if let Some(mode) = tf32_attention_mode.as_deref() {
            eprintln!("UMT5 DIAGNOSTIC: tf32 attention emulation active (mode={mode:?})");
        }
        let tf32_attention = tf32_attention_mode.is_some();
        let tf32_attention_chunks8 = tf32_attention_mode.as_deref() == Some("chunks8");
        for head in 0..heads {
            let head_offset = head * head_dim;
            for query_row in 0..n {
                for key_row in 0..n {
                    let dot = attention_dot(
                        &query[query_row * d + head_offset..query_row * d + head_offset + head_dim],
                        &key[key_row * d + head_offset..key_row * d + head_offset + head_dim],
                        tf32_attention,
                        tf32_attention_chunks8,
                    );
                    let bucket = relative_bucket(
                        key_row as i32 - query_row as i32,
                        encoder.cfg.n_buckets,
                        encoder.cfg.max_distance,
                    );
                    scores[key_row] = dot + block.rel_b[bucket * heads + head];
                }
                scores_biased[(head * n + query_row) * n..(head * n + query_row + 1) * n]
                    .copy_from_slice(&scores);
                softmax(&mut scores);
                probabilities[(head * n + query_row) * n..(head * n + query_row + 1) * n]
                    .copy_from_slice(&scores);
                for key_row in 0..n {
                    for channel in 0..head_dim {
                        let output = &mut context[query_row * d + head_offset + channel];
                        let probability = scores[key_row];
                        let value = value[key_row * d + head_offset + channel];
                        *output = if tf32_attention {
                            round_to_tf32(probability).mul_add(round_to_tf32(value), *output)
                        } else {
                            probability * value + *output
                        };
                    }
                }
            }
        }
        let projected = encoder.mat(&context, block.o, d, d, n);
        let post_attention = input
            .iter()
            .zip(&projected)
            .map(|(input, projected)| input + projected)
            .collect();
        Block0AttentionTrace {
            input,
            norm,
            query,
            key,
            value,
            scores_biased,
            probabilities,
            context,
            projected,
            post_attention,
        }
    }

    fn compare(tag: &str, prompt: &str, dump: &str) {
        let gp = Path::new(PACK).join("umt5-xxl-encoder-Q4_K_M.gguf");
        let rp = Path::new(FIXTURES).join(dump);
        // The 3.6 GB encoder is not in the repository, so its absence is a genuine skip. The
        // fixture is committed, so its absence means a broken checkout — never a silent pass.
        assert!(
            rp.exists(),
            "{tag}: missing committed fixture {}",
            rp.display()
        );
        if !gp.exists() {
            eprintln!("skipping {tag}: encoder {} not present", gp.display());
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
            if na == 0.0 || nb == 0.0 {
                return if na == nb { 1.0 } else { 0.0 };
            }
            dot / (na.sqrt() * nb.sqrt())
        };

        eprintln!("{tag}: n_valid={n_valid} seq={seq} d={d}");
        // Per-position, so a core that is right but mis-assembled is distinguishable from a core
        // that is simply wrong.
        for i in 0..n_valid.min(6) {
            let (g, w) = (&got[i * d..(i + 1) * d], &want[i * d..(i + 1) * d]);
            let md = g
                .iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("  pos {i}: cosine={:.6} max_abs={md:.6}", cos(g, w));
        }
        let pads_zero = want[n_valid * d..].iter().all(|&v| v == 0.0);
        let ours_zero = got[n_valid * d..].iter().all(|&v| v == 0.0);
        eprintln!("  padding zeroed: reference={pads_zero} ours={ours_zero}");

        let valid = n_valid * d;
        let c_valid = cos(&got[..valid], &want[..valid]);
        let c_all = cos(&got, &want);
        let max_abs = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("  cosine(valid)={c_valid:.8} cosine(all)={c_all:.8} max_abs={max_abs:.6}");
        eprintln!("  got[..4]  = {:?}", &got[..4]);
        eprintln!("  want[..4] = {:?}", &want[..4]);

        assert!(
            c_valid > 0.999,
            "{tag}: cosine over valid positions {c_valid:.6}"
        );
        assert!(max_abs < 0.05, "{tag}: max abs diff {max_abs:.6}");
    }

    #[cfg(feature = "vulkan")]
    fn compare_vulkan(tag: &str, prompt: &str, dump: &str) {
        use crate::backend::VULKAN_BACKEND;
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let gp = Path::new(PACK).join("umt5-xxl-encoder-Q4_K_M.gguf");
        let rp = Path::new(FIXTURES).join(dump);
        // See `compare`: a missing encoder is a skip, a missing committed fixture is a failure.
        assert!(
            rp.exists(),
            "Vulkan {tag}: missing committed fixture {}",
            rp.display()
        );
        if !gp.exists() {
            eprintln!("skipping Vulkan {tag}: encoder {} not present", gp.display());
            return;
        }
        let (dims, expected_values) = read_dump(&rp).expect("reference dump unreadable");
        let gguf = GgufFile::open(&gp).expect("gguf");
        let tokenizer = T5Tokenizer::from_gguf(&gguf).expect("tokenizer");
        let encoder = Umt5Encoder::load(&gguf).expect("encoder");
        let model_width = dims[0] as usize;
        let sequence = dims[1] as usize;
        let n_valid = tokenizer.encode(prompt).len();
        let ids = tokenizer.encode_padded(prompt, sequence);

        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping Vulkan {tag}: {error:#}");
                return;
            }
            Err(error) => panic!("required Vulkan UMT5 {tag} failed to initialize: {error:#}"),
        };
        let started = std::time::Instant::now();
        let actual_values = encoder
            .forward_with_backend(&VULKAN_BACKEND, &ids, n_valid)
            .unwrap();
        let runtime = started.elapsed();
        let after = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(actual_values.len(), expected_values.len());
        assert!(
            actual_values[n_valid * model_width..]
                .iter()
                .all(|&value| value == 0.0),
            "Vulkan UMT5 padding must be exactly zero"
        );

        let actual = Tensor::new(vec![sequence, model_width], actual_values).unwrap();
        let expected = Tensor::new(vec![sequence, model_width], expected_values).unwrap();
        let metrics = compare_tensors(&actual, &expected).unwrap();
        let expected_weight_uploads = (encoder.cfg.n_layers * 9 + 1) as u64;
        let expected_tensor_uploads = (encoder.cfg.n_layers + 1) as u64;
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            expected_weight_uploads,
            "each UMT5 projection and norm must be staged exactly once"
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            expected_tensor_uploads,
            "only embeddings and per-block relative biases may be host-uploaded"
        );
        assert_eq!(
            after.resident_downloads - before.resident_downloads,
            1,
            "only the final UMT5 context may be downloaded"
        );
        assert_eq!(
            after.resident_allocated_bytes, before.resident_allocated_bytes,
            "UMT5 activations must be released after the final download"
        );
        assert_eq!(
            after.resident_device_local_bytes, before.resident_device_local_bytes,
            "staged UMT5 weights must be released after each projection"
        );
        eprintln!(
            "Vulkan UMT5 {tag}: shape={:?} valid_tokens={} runtime_s={:.3} cosine={:.9} max_abs={:.9} mean_abs={:.9} weight_uploads={} tensor_uploads={} downloads={} uploaded_bytes={} downloaded_bytes={} peak_resident={} peak_device_local={}",
            actual.shape(),
            n_valid,
            runtime.as_secs_f64(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_downloads - before.resident_downloads,
            after.resident_uploaded_bytes - before.resident_uploaded_bytes,
            after.resident_downloaded_bytes - before.resident_downloaded_bytes,
            after.peak_resident_allocated_bytes,
            after.peak_resident_device_local_bytes,
        );
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999,
                maximum_absolute_error: 0.05,
                maximum_mean_absolute_error: 0.05,
            })
            .unwrap();
    }

    #[test]
    #[ignore = "uses opt-in block-0 captures from the pinned CUDA reference runner"]
    fn scalar_prelude_and_block0_locate_first_reference_divergence() {
        use crate::parity::compare_tensors;

        let gguf_path = Path::new(PACK).join("umt5-xxl-encoder-Q4_K_M.gguf");
        if !gguf_path.exists() {
            eprintln!("skipping UMT5 block-0 trace: model missing");
            return;
        }
        let captures = [
            (
                "cond",
                "a red fox",
                "saient_t5_prelude_0.bin",
                "saient_t5_block0_post_attention_0.bin",
                "saient_t5_block0_0.bin",
            ),
            (
                "uncond",
                "",
                "saient_t5_prelude_1.bin",
                "saient_t5_block0_post_attention_1.bin",
                "saient_t5_block0_1.bin",
            ),
        ];
        if captures.iter().any(|(_, _, prelude, attention, block)| {
            !Path::new(REF_DIR).join(prelude).exists()
                || !Path::new(REF_DIR).join(attention).exists()
                || !Path::new(REF_DIR).join(block).exists()
        }) {
            eprintln!("skipping UMT5 block-0 trace: reference captures missing");
            return;
        }

        let gguf = GgufFile::open(&gguf_path).unwrap();
        let tokenizer = T5Tokenizer::from_gguf(&gguf).unwrap();
        let encoder = Umt5Encoder::load(&gguf).unwrap();
        for (tag, prompt, prelude_name, attention_name, block_name) in captures {
            let n_valid = tokenizer.encode(prompt).len();
            let ids = tokenizer.encode_padded(prompt, encoder.cfg.context);
            let attention_trace = scalar_block0_attention_trace(&encoder, &ids, n_valid);
            let block0 = &encoder.blocks[0];
            let mut cuda_attention_norm = vec![0.0; n_valid * encoder.cfg.d_model];
            for row in 0..n_valid {
                rms_norm_cuda_into(
                    &attention_trace.input
                        [row * encoder.cfg.d_model..(row + 1) * encoder.cfg.d_model],
                    &block0.attn_norm,
                    encoder.cfg.eps,
                    &mut cuda_attention_norm
                        [row * encoder.cfg.d_model..(row + 1) * encoder.cfg.d_model],
                );
            }
            let mmq = |tensor: &TensorInfo| match tensor.ggml_type {
                12 => dequant::gemm_q4k_q8_1_mmq(
                    &attention_trace.norm,
                    encoder.gguf.tensor_data(tensor),
                    encoder.cfg.d_model,
                    encoder.cfg.d_model,
                    n_valid,
                ),
                14 => dequant::gemm_q6k_q8_1_mmq(
                    &attention_trace.norm,
                    encoder.gguf.tensor_data(tensor),
                    encoder.cfg.d_model,
                    encoder.cfg.d_model,
                    n_valid,
                ),
                other => panic!("block-0 MMQ diagnostic does not support ggml type {other}"),
            };
            let mmq_query = mmq(block0.q);
            let mmq_key = mmq(block0.k);
            let mmq_value = mmq(block0.v);
            let mmq_from_cuda_norm = |tensor: &TensorInfo| match tensor.ggml_type {
                12 => dequant::gemm_q4k_q8_1_mmq(
                    &cuda_attention_norm,
                    encoder.gguf.tensor_data(tensor),
                    encoder.cfg.d_model,
                    encoder.cfg.d_model,
                    n_valid,
                ),
                14 => dequant::gemm_q6k_q8_1_mmq(
                    &cuda_attention_norm,
                    encoder.gguf.tensor_data(tensor),
                    encoder.cfg.d_model,
                    encoder.cfg.d_model,
                    n_valid,
                ),
                other => panic!("block-0 MMQ diagnostic does not support ggml type {other}"),
            };
            let cuda_norm_query = mmq_from_cuda_norm(block0.q);
            let cuda_norm_key = mmq_from_cuda_norm(block0.k);
            let cuda_norm_value = mmq_from_cuda_norm(block0.v);
            let cuda_norm_query_stream_k = dequant::gemm_q4k_q8_1_mmq_stream_k(
                &cuda_attention_norm,
                encoder.gguf.tensor_data(block0.q),
                encoder.cfg.d_model,
                encoder.cfg.d_model,
                n_valid,
                encoder.cfg.context,
                36,
            );
            let cuda_norm_key_stream_k = dequant::gemm_q4k_q8_1_mmq_stream_k(
                &cuda_attention_norm,
                encoder.gguf.tensor_data(block0.k),
                encoder.cfg.d_model,
                encoder.cfg.d_model,
                n_valid,
                encoder.cfg.context,
                36,
            );
            let cuda_norm_value_stream_k = dequant::gemm_q6k_q8_1_mmq_stream_k(
                &cuda_attention_norm,
                encoder.gguf.tensor_data(block0.v),
                encoder.cfg.d_model,
                encoder.cfg.d_model,
                n_valid,
                encoder.cfg.context,
                36,
            );
            eprintln!(
                "UMT5 {tag} block0 projection types: q={} k={} v={} o={} gate={} up={} down={}",
                block0.q.ggml_type,
                block0.k.ggml_type,
                block0.v.ggml_type,
                block0.o.ggml_type,
                block0.gate.ggml_type,
                block0.up.ggml_type,
                block0.down.ggml_type,
            );
            let mut actual_prelude = Vec::new();
            let mut actual_attention = Vec::new();
            let mut actual_block = Vec::new();
            encoder.forward_traced(&ids, n_valid, 1, |point, values| match point {
                Umt5TracePoint::Prelude => actual_prelude.extend_from_slice(values),
                Umt5TracePoint::PostAttention(0) => actual_attention.extend_from_slice(values),
                Umt5TracePoint::PostBlock(0) => actual_block.extend_from_slice(values),
                Umt5TracePoint::PostAttention(_) | Umt5TracePoint::PostBlock(_) => unreachable!(),
            });
            let (_, expected_prelude) = read_dump(&Path::new(REF_DIR).join(prelude_name)).unwrap();
            let (_, expected_attention) =
                read_dump(&Path::new(REF_DIR).join(attention_name)).unwrap();
            let (_, expected_block) = read_dump(&Path::new(REF_DIR).join(block_name)).unwrap();
            let valid_values = n_valid * encoder.cfg.d_model;
            let actual_prelude = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                actual_prelude[..valid_values].to_vec(),
            )
            .unwrap();
            let actual_attention = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                actual_attention[..valid_values].to_vec(),
            )
            .unwrap();
            let expected_attention = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                expected_attention[..valid_values].to_vec(),
            )
            .unwrap();
            let expected_prelude = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                expected_prelude[..valid_values].to_vec(),
            )
            .unwrap();
            let actual_block = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                actual_block[..valid_values].to_vec(),
            )
            .unwrap();
            let expected_block = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                expected_block[..valid_values].to_vec(),
            )
            .unwrap();
            let prelude = compare_tensors(&actual_prelude, &expected_prelude).unwrap();
            let attention = compare_tensors(&actual_attention, &expected_attention).unwrap();
            let block = compare_tensors(&actual_block, &expected_block).unwrap();
            eprintln!(
                "UMT5 {tag} prelude: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                actual_prelude.shape(),
                prelude.cosine_similarity,
                prelude.maximum_absolute_error,
                prelude.mean_absolute_error,
            );
            eprintln!(
                "UMT5 {tag} post-attention: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                actual_attention.shape(),
                attention.cosine_similarity,
                attention.maximum_absolute_error,
                attention.mean_absolute_error,
            );
            let capture_index = usize::from(tag == "uncond");
            let (_, expected_context_values) = read_dump(&Path::new(REF_DIR).join(format!(
                "saient_live_t5_block0_attention_context_{capture_index}.bin"
            )))
            .unwrap();
            let (_, expected_scores_full) = read_dump(&Path::new(REF_DIR).join(format!(
                "saient_live_t5_block0_attention_scores_biased_{capture_index}.bin"
            )))
            .unwrap();
            let (_, expected_probabilities_full) = read_dump(&Path::new(REF_DIR).join(format!(
                "saient_live_t5_block0_attention_probabilities_{capture_index}.bin"
            )))
            .unwrap();
            let mut expected_scores = Vec::with_capacity(encoder.cfg.n_heads * n_valid * n_valid);
            let mut expected_probabilities = Vec::with_capacity(expected_scores.capacity());
            for head in 0..encoder.cfg.n_heads {
                for query_row in 0..n_valid {
                    let offset = (head * encoder.cfg.context + query_row) * encoder.cfg.context;
                    expected_scores
                        .extend_from_slice(&expected_scores_full[offset..offset + n_valid]);
                    expected_probabilities
                        .extend_from_slice(&expected_probabilities_full[offset..offset + n_valid]);
                }
            }
            for (operation, actual_values, expected_values) in [
                (
                    "attention_scores_biased",
                    attention_trace.scores_biased.as_slice(),
                    expected_scores.as_slice(),
                ),
                (
                    "attention_probabilities",
                    attention_trace.probabilities.as_slice(),
                    expected_probabilities.as_slice(),
                ),
            ] {
                let shape = vec![encoder.cfg.n_heads, n_valid, n_valid];
                let actual = Tensor::new(shape.clone(), actual_values.to_vec()).unwrap();
                let expected = Tensor::new(shape, expected_values.to_vec()).unwrap();
                let metrics = compare_tensors(&actual, &expected).unwrap();
                eprintln!(
                    "UMT5 {tag} block0 {operation}: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                    metrics.cosine_similarity,
                    metrics.maximum_absolute_error,
                    metrics.mean_absolute_error,
                );
            }
            let (_, reference_value) = read_dump(
                &Path::new(REF_DIR).join(format!("saient_live_t5_block0_v_{capture_index}.bin")),
            )
            .unwrap();
            let mut context_reference_probabilities = vec![0.0f32; valid_values];
            let mut context_reference_value = vec![0.0f32; valid_values];
            let tf32_attention = std::env::var_os("QUARTZ_UMT5_TF32_ATTENTION").is_some();
            for head in 0..encoder.cfg.n_heads {
                let head_offset = head * encoder.cfg.head_dim;
                for query_row in 0..n_valid {
                    for channel in 0..encoder.cfg.head_dim {
                        let output_index = query_row * encoder.cfg.d_model + head_offset + channel;
                        for key_row in 0..n_valid {
                            let probability_index =
                                (head * n_valid + query_row) * n_valid + key_row;
                            let current_value = attention_trace.value
                                [key_row * encoder.cfg.d_model + head_offset + channel];
                            let reference_value = reference_value
                                [key_row * encoder.cfg.d_model + head_offset + channel];
                            let reference_probability = expected_probabilities[probability_index];
                            let current_probability =
                                attention_trace.probabilities[probability_index];
                            if tf32_attention {
                                context_reference_probabilities[output_index] =
                                    round_to_tf32(reference_probability).mul_add(
                                        round_to_tf32(current_value),
                                        context_reference_probabilities[output_index],
                                    );
                                context_reference_value[output_index] =
                                    round_to_tf32(current_probability).mul_add(
                                        round_to_tf32(reference_value),
                                        context_reference_value[output_index],
                                    );
                            } else {
                                context_reference_probabilities[output_index] +=
                                    reference_probability * current_value;
                                context_reference_value[output_index] +=
                                    current_probability * reference_value;
                            }
                        }
                    }
                }
            }
            let expected_context = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                expected_context_values[..valid_values].to_vec(),
            )
            .unwrap();
            for (operation, values) in [
                (
                    "attention_context_reference_probabilities",
                    context_reference_probabilities,
                ),
                ("attention_context_reference_value", context_reference_value),
            ] {
                let actual = Tensor::new(vec![n_valid, encoder.cfg.d_model], values).unwrap();
                let metrics = compare_tensors(&actual, &expected_context).unwrap();
                eprintln!(
                    "UMT5 {tag} block0 {operation}: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                    metrics.cosine_similarity,
                    metrics.maximum_absolute_error,
                    metrics.mean_absolute_error,
                );
            }
            let projected_from_reference_context = encoder.mat(
                &expected_context_values[..valid_values],
                block0.o,
                encoder.cfg.d_model,
                encoder.cfg.d_model,
                n_valid,
            );
            for (operation, actual_values) in [
                ("attention_norm", attention_trace.norm.as_slice()),
                ("attention_norm_cuda", cuda_attention_norm.as_slice()),
                ("q", attention_trace.query.as_slice()),
                ("q_mmq", mmq_query.as_slice()),
                ("q_cuda", cuda_norm_query.as_slice()),
                ("q_cuda_stream_k", cuda_norm_query_stream_k.as_slice()),
                ("k", attention_trace.key.as_slice()),
                ("k_mmq", mmq_key.as_slice()),
                ("k_cuda", cuda_norm_key.as_slice()),
                ("k_cuda_stream_k", cuda_norm_key_stream_k.as_slice()),
                ("v", attention_trace.value.as_slice()),
                ("v_mmq", mmq_value.as_slice()),
                ("v_cuda", cuda_norm_value.as_slice()),
                ("v_cuda_stream_k", cuda_norm_value_stream_k.as_slice()),
                ("attention_context", attention_trace.context.as_slice()),
                ("attention_projected", attention_trace.projected.as_slice()),
                ("post_attention", attention_trace.post_attention.as_slice()),
            ] {
                let capture_operation = operation.strip_suffix("_mmq").unwrap_or(operation);
                let capture_operation = capture_operation
                    .strip_suffix("_stream_k")
                    .unwrap_or(capture_operation);
                let capture_operation = capture_operation
                    .strip_suffix("_cuda")
                    .unwrap_or(capture_operation);
                let capture =
                    format!("saient_live_t5_block0_{capture_operation}_{capture_index}.bin");
                let (_, expected_values) = read_dump(&Path::new(REF_DIR).join(capture)).unwrap();
                let actual =
                    Tensor::new(vec![n_valid, encoder.cfg.d_model], actual_values.to_vec())
                        .unwrap();
                let expected = Tensor::new(
                    vec![n_valid, encoder.cfg.d_model],
                    expected_values[..valid_values].to_vec(),
                )
                .unwrap();
                let operation_metrics = compare_tensors(&actual, &expected).unwrap();
                eprintln!(
                    "UMT5 {tag} block0 {operation}: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                    operation_metrics.cosine_similarity,
                    operation_metrics.maximum_absolute_error,
                    operation_metrics.mean_absolute_error,
                );
            }
            let (_, expected_projected_values) = read_dump(&Path::new(REF_DIR).join(format!(
                "saient_live_t5_block0_attention_projected_{capture_index}.bin"
            )))
            .unwrap();
            let projected_from_reference_context = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                projected_from_reference_context,
            )
            .unwrap();
            let expected_projected = Tensor::new(
                vec![n_valid, encoder.cfg.d_model],
                expected_projected_values[..valid_values].to_vec(),
            )
            .unwrap();
            let projected_metrics =
                compare_tensors(&projected_from_reference_context, &expected_projected).unwrap();
            eprintln!(
                "UMT5 {tag} block0 attention_projected_from_reference_context: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                projected_metrics.cosine_similarity,
                projected_metrics.maximum_absolute_error,
                projected_metrics.mean_absolute_error,
            );

            // Isolate the FFN by starting from the reference post-attention state. This prevents
            // a small attention error from obscuring which FFN primitive first diverges.
            let (_, reference_post_attention) = read_dump(&Path::new(REF_DIR).join(format!(
                "saient_live_t5_block0_post_attention_{capture_index}.bin"
            )))
            .unwrap();
            let reference_post_attention = &reference_post_attention[..valid_values];
            let mut ffn_norm = vec![0.0; valid_values];
            for row in 0..n_valid {
                rms_norm_into(
                    &reference_post_attention
                        [row * encoder.cfg.d_model..(row + 1) * encoder.cfg.d_model],
                    &block0.ffn_norm,
                    encoder.cfg.eps,
                    &mut ffn_norm[row * encoder.cfg.d_model..(row + 1) * encoder.cfg.d_model],
                );
            }
            let gate_pre_gelu = encoder.mat(
                &ffn_norm,
                block0.gate,
                encoder.cfg.d_model,
                encoder.cfg.d_ff,
                n_valid,
            );
            let gate = gate_pre_gelu.iter().copied().map(gelu).collect::<Vec<_>>();
            let gate_exp = gate_pre_gelu
                .iter()
                .copied()
                .map(gelu_via_exp)
                .collect::<Vec<_>>();
            let gate_exp2 = gate_pre_gelu
                .iter()
                .copied()
                .map(gelu_via_exp2)
                .collect::<Vec<_>>();
            let up = encoder.mat(
                &ffn_norm,
                block0.up,
                encoder.cfg.d_model,
                encoder.cfg.d_ff,
                n_valid,
            );
            let product = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| gate * up)
                .collect::<Vec<_>>();
            let down = encoder.mat(
                &product,
                block0.down,
                encoder.cfg.d_ff,
                encoder.cfg.d_model,
                n_valid,
            );
            let scaled_product = product.iter().map(|value| value / 32.0).collect::<Vec<_>>();
            let scaled_down = encoder
                .mat(
                    &scaled_product,
                    block0.down,
                    encoder.cfg.d_ff,
                    encoder.cfg.d_model,
                    n_valid,
                )
                .into_iter()
                .map(|value| value * 32.0)
                .collect::<Vec<_>>();
            for (operation, width, actual_values) in [
                ("ffn_norm", encoder.cfg.d_model, ffn_norm.as_slice()),
                (
                    "ffn_gate_pre_gelu",
                    encoder.cfg.d_ff,
                    gate_pre_gelu.as_slice(),
                ),
                ("ffn_gate", encoder.cfg.d_ff, gate.as_slice()),
                ("ffn_up", encoder.cfg.d_ff, up.as_slice()),
                ("ffn_product", encoder.cfg.d_ff, product.as_slice()),
                ("ffn_down", encoder.cfg.d_model, down.as_slice()),
                (
                    "ffn_down_scaled",
                    encoder.cfg.d_model,
                    scaled_down.as_slice(),
                ),
            ] {
                let capture_operation = operation.strip_suffix("_scaled").unwrap_or(operation);
                let (_, expected_values) = read_dump(&Path::new(REF_DIR).join(format!(
                    "saient_live_t5_block0_{capture_operation}_{capture_index}.bin"
                )))
                .unwrap();
                let actual = Tensor::new(vec![n_valid, width], actual_values.to_vec()).unwrap();
                let expected = Tensor::new(
                    vec![n_valid, width],
                    expected_values[..n_valid * width].to_vec(),
                )
                .unwrap();
                let metrics = compare_tensors(&actual, &expected).unwrap();
                eprintln!(
                    "UMT5 {tag} block0 {operation}_from_reference_post_attention: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                    metrics.cosine_similarity,
                    metrics.maximum_absolute_error,
                    metrics.mean_absolute_error,
                );
            }
            let (_, expected_gate_values) = read_dump(&Path::new(REF_DIR).join(format!(
                "saient_live_t5_block0_ffn_gate_{capture_index}.bin"
            )))
            .unwrap();
            let expected_gate = Tensor::new(
                vec![n_valid, encoder.cfg.d_ff],
                expected_gate_values[..n_valid * encoder.cfg.d_ff].to_vec(),
            )
            .unwrap();
            for (name, values) in [("exp", gate_exp), ("exp2", gate_exp2)] {
                let values = Tensor::new(vec![n_valid, encoder.cfg.d_ff], values).unwrap();
                let metrics = compare_tensors(&values, &expected_gate).unwrap();
                eprintln!(
                    "UMT5 {tag} block0 ffn_gate_{name}_from_reference_post_attention: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                    metrics.cosine_similarity,
                    metrics.maximum_absolute_error,
                    metrics.mean_absolute_error,
                );
            }

            // Follow the FFN from the implementation's own post-attention state as well. The
            // reference-input comparisons above prove each isolated primitive; these comparisons
            // show where the remaining post-attention error is amplified in the composed block.
            let current_post_attention = actual_attention.data();
            let mut current_ffn_norm = vec![0.0; valid_values];
            for row in 0..n_valid {
                rms_norm_into(
                    &current_post_attention
                        [row * encoder.cfg.d_model..(row + 1) * encoder.cfg.d_model],
                    &block0.ffn_norm,
                    encoder.cfg.eps,
                    &mut current_ffn_norm
                        [row * encoder.cfg.d_model..(row + 1) * encoder.cfg.d_model],
                );
            }
            let current_gate_pre_gelu = encoder.mat(
                &current_ffn_norm,
                block0.gate,
                encoder.cfg.d_model,
                encoder.cfg.d_ff,
                n_valid,
            );
            let current_gate = current_gate_pre_gelu
                .iter()
                .copied()
                .map(gelu)
                .collect::<Vec<_>>();
            let current_up = encoder.mat(
                &current_ffn_norm,
                block0.up,
                encoder.cfg.d_model,
                encoder.cfg.d_ff,
                n_valid,
            );
            let current_product = current_gate
                .iter()
                .zip(&current_up)
                .map(|(gate, up)| gate * up)
                .collect::<Vec<_>>();
            let current_down = encoder.mat(
                &current_product,
                block0.down,
                encoder.cfg.d_ff,
                encoder.cfg.d_model,
                n_valid,
            );
            for (operation, width, actual_values) in [
                ("ffn_norm", encoder.cfg.d_model, current_ffn_norm.as_slice()),
                (
                    "ffn_gate_pre_gelu",
                    encoder.cfg.d_ff,
                    current_gate_pre_gelu.as_slice(),
                ),
                ("ffn_gate", encoder.cfg.d_ff, current_gate.as_slice()),
                ("ffn_up", encoder.cfg.d_ff, current_up.as_slice()),
                ("ffn_product", encoder.cfg.d_ff, current_product.as_slice()),
                ("ffn_down", encoder.cfg.d_model, current_down.as_slice()),
            ] {
                let (_, expected_values) = read_dump(&Path::new(REF_DIR).join(format!(
                    "saient_live_t5_block0_{operation}_{capture_index}.bin"
                )))
                .unwrap();
                let actual = Tensor::new(vec![n_valid, width], actual_values.to_vec()).unwrap();
                let expected = Tensor::new(
                    vec![n_valid, width],
                    expected_values[..n_valid * width].to_vec(),
                )
                .unwrap();
                let metrics = compare_tensors(&actual, &expected).unwrap();
                eprintln!(
                    "UMT5 {tag} block0 {operation}_composed: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                    metrics.cosine_similarity,
                    metrics.maximum_absolute_error,
                    metrics.mean_absolute_error,
                );
            }
            eprintln!(
                "UMT5 {tag} block0: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                actual_block.shape(),
                block.cosine_similarity,
                block.maximum_absolute_error,
                block.mean_absolute_error,
            );
            assert_eq!(
                prelude.maximum_absolute_error, 0.0,
                "{tag}: token-row dequantization diverges before block 0"
            );
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "uses opt-in block-0 captures from the pinned CUDA reference runner"]
    fn vulkan_gelu_matches_captured_cuda_fast_math() {
        use crate::backend::VULKAN_BACKEND;
        use crate::parity::compare_tensors;

        for (tag, capture_index, rows) in [("cond", 0, 5), ("uncond", 1, 2)] {
            let (_, input) = read_dump(&Path::new(REF_DIR).join(format!(
                "saient_live_t5_block0_ffn_gate_pre_gelu_{capture_index}.bin"
            )))
            .expect("missing captured CUDA GELU input");
            let (_, expected) = read_dump(&Path::new(REF_DIR).join(format!(
                "saient_live_t5_block0_ffn_gate_{capture_index}.bin"
            )))
            .expect("missing captured CUDA GELU output");
            let values = rows * 10_240;
            let input = Tensor::new(vec![rows, 10_240], input[..values].to_vec()).unwrap();
            let expected = Tensor::new(vec![rows, 10_240], expected[..values].to_vec()).unwrap();
            let input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
            let actual = VULKAN_BACKEND.gelu_tanh_device(&input).unwrap();
            let actual = VULKAN_BACKEND.download_tensor(&actual).unwrap();
            let metrics = compare_tensors(&actual, &expected).unwrap();
            eprintln!(
                "UMT5 {tag} Vulkan GELU vs CUDA fast-math: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                actual.shape(),
                metrics.cosine_similarity,
                metrics.maximum_absolute_error,
                metrics.mean_absolute_error,
            );
            assert_eq!(actual.shape(), expected.shape());
            assert!(metrics.cosine_similarity >= 0.999_999_9);
        }
    }

    #[test]
    #[ignore = "uses opt-in block-0 attention captures from the pinned CUDA reference runner"]
    fn cuda_attention_gemm_reduction_diagnostic() {
        use crate::parity::compare_tensors;

        const ROWS: usize = 512;
        const VALID: usize = 5;
        const WIDTH: usize = 4096;
        const HEADS: usize = 64;
        const HEAD_DIM: usize = 64;

        let (_, query) = read_dump(&Path::new(REF_DIR).join("saient_live_t5_block0_q_0.bin"))
            .expect("missing captured query");
        let (_, key) = read_dump(&Path::new(REF_DIR).join("saient_live_t5_block0_k_0.bin"))
            .expect("missing captured key");
        let (_, expected_scores) = read_dump(
            &Path::new(REF_DIR).join("saient_live_t5_block0_attention_scores_scaled_0.bin"),
        )
        .expect("missing captured scaled attention scores");

        let mut expected = Vec::with_capacity(HEADS * VALID * VALID);
        let mut sequential = Vec::with_capacity(expected.capacity());
        let mut sequential_truncated = Vec::with_capacity(expected.capacity());
        let mut chunks8 = Vec::with_capacity(expected.capacity());
        let mut chunks8_adjacent = Vec::with_capacity(expected.capacity());
        let mut chunks8_xor = Vec::with_capacity(expected.capacity());
        let mut chunks8_f64 = Vec::with_capacity(expected.capacity());
        let mut mma8_f64 = Vec::with_capacity(expected.capacity());
        let mut lanes8 = Vec::with_capacity(expected.capacity());
        for head in 0..HEADS {
            let offset = head * HEAD_DIM;
            for query_row in 0..VALID {
                for key_row in 0..VALID {
                    expected.push(expected_scores[(head * ROWS + query_row) * ROWS + key_row]);
                    let mut seq = 0.0f32;
                    let mut seq_truncated = 0.0f32;
                    let mut chunked = 0.0f32;
                    let mut chunked_adjacent = 0.0f32;
                    let mut chunked_xor = 0.0f32;
                    let mut chunked_f64 = 0.0f32;
                    let mut mma_f64 = 0.0f32;
                    let mut lanes = [0.0f32; 8];
                    for channel in 0..HEAD_DIM {
                        let q = round_to_tf32(query[query_row * WIDTH + offset + channel]);
                        let k = round_to_tf32(key[key_row * WIDTH + offset + channel]);
                        seq = q.mul_add(k, seq);
                        let q_truncated = f32::from_bits(
                            query[query_row * WIDTH + offset + channel].to_bits() & 0xffff_e000,
                        );
                        let k_truncated = f32::from_bits(
                            key[key_row * WIDTH + offset + channel].to_bits() & 0xffff_e000,
                        );
                        seq_truncated = q_truncated.mul_add(k_truncated, seq_truncated);
                        lanes[channel % 8] = q.mul_add(k, lanes[channel % 8]);
                        if channel % 8 == 7 {
                            let mut tile = 0.0f32;
                            let mut products = [0.0f32; 8];
                            for tile_channel in channel - 7..=channel {
                                let q =
                                    round_to_tf32(query[query_row * WIDTH + offset + tile_channel]);
                                let k = round_to_tf32(key[key_row * WIDTH + offset + tile_channel]);
                                tile = q.mul_add(k, tile);
                                products[tile_channel - (channel - 7)] = q * k;
                            }
                            chunked += tile;
                            let mut adjacent = products;
                            let mut adjacent_width = 8;
                            while adjacent_width > 1 {
                                for lane in 0..adjacent_width / 2 {
                                    adjacent[lane] = adjacent[lane * 2] + adjacent[lane * 2 + 1];
                                }
                                adjacent_width /= 2;
                            }
                            chunked_adjacent += adjacent[0];
                            let mut xor = products;
                            for xor_offset in [4, 2, 1] {
                                let previous = xor;
                                for lane in 0..8 {
                                    xor[lane] = previous[lane] + previous[lane ^ xor_offset];
                                }
                            }
                            chunked_xor += xor[0];
                            let exact_chunk =
                                products.iter().map(|value| f64::from(*value)).sum::<f64>();
                            chunked_f64 += exact_chunk as f32;
                            mma_f64 = (f64::from(mma_f64) + exact_chunk) as f32;
                        }
                    }
                    sequential.push(seq);
                    sequential_truncated.push(seq_truncated);
                    chunks8.push(chunked);
                    chunks8_adjacent.push(chunked_adjacent);
                    chunks8_xor.push(chunked_xor);
                    chunks8_f64.push(chunked_f64);
                    mma8_f64.push(mma_f64);
                    let mut width = 4;
                    while width > 0 {
                        for lane in 0..width {
                            lanes[lane] += lanes[lane + width];
                        }
                        width /= 2;
                    }
                    lanes8.push(lanes[0]);
                }
            }
        }
        let expected = Tensor::new(vec![HEADS, VALID, VALID], expected).unwrap();
        for (name, values) in [
            ("sequential", sequential),
            ("sequential_truncated", sequential_truncated),
            ("chunks8", chunks8),
            ("chunks8_adjacent", chunks8_adjacent),
            ("chunks8_xor", chunks8_xor),
            ("chunks8_f64", chunks8_f64),
            ("mma8_f64", mma8_f64),
            ("lanes8", lanes8),
        ] {
            let actual = Tensor::new(vec![HEADS, VALID, VALID], values).unwrap();
            let metrics = compare_tensors(&actual, &expected).unwrap();
            let exact = actual
                .data()
                .iter()
                .zip(expected.data())
                .filter(|(actual, expected)| actual.to_bits() == expected.to_bits())
                .count();
            eprintln!(
                "UMT5 CUDA attention scores {name}: cosine={:.9} max_abs={:.9} mean_abs={:.9} exact={exact}/{}",
                metrics.cosine_similarity,
                metrics.maximum_absolute_error,
                metrics.mean_absolute_error,
                actual.data().len(),
            );
        }
        drop(expected_scores);

        let (_, value) = read_dump(&Path::new(REF_DIR).join("saient_live_t5_block0_v_0.bin"))
            .expect("missing captured value");
        let (_, probabilities) = read_dump(
            &Path::new(REF_DIR).join("saient_live_t5_block0_attention_probabilities_0.bin"),
        )
        .expect("missing captured attention probabilities");
        let (_, biased_scores) = read_dump(
            &Path::new(REF_DIR).join("saient_live_t5_block0_attention_scores_biased_0.bin"),
        )
        .expect("missing captured biased attention scores");
        let mut expected_probability = Vec::with_capacity(HEADS * VALID * VALID);
        let mut scalar_probability = Vec::with_capacity(expected_probability.capacity());
        let mut tree_probability = Vec::with_capacity(expected_probability.capacity());
        for head in 0..HEADS {
            for query_row in 0..VALID {
                let row_offset = (head * ROWS + query_row) * ROWS;
                let row = &biased_scores[row_offset..row_offset + ROWS];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut scalar = row[..VALID]
                    .iter()
                    .map(|value| (*value - max).exp())
                    .collect::<Vec<_>>();
                let scalar_sum = scalar.iter().copied().sum::<f32>();
                for value in &mut scalar {
                    *value /= scalar_sum;
                }
                let mut lanes = [0.0f32; ROWS];
                for lane in 0..ROWS {
                    lanes[lane] = (row[lane] - max).exp();
                }
                for warp in lanes.chunks_exact_mut(32) {
                    for offset in [16, 8, 4, 2, 1] {
                        let previous = warp.to_vec();
                        for lane in 0..32 {
                            warp[lane] = previous[lane] + previous[lane ^ offset];
                        }
                    }
                }
                let mut warp_sums = [0.0f32; 32];
                for (warp, value) in warp_sums.iter_mut().take(16).enumerate() {
                    *value = lanes[warp * 32];
                }
                for offset in [16, 8, 4, 2, 1] {
                    let previous = warp_sums;
                    for lane in 0..32 {
                        warp_sums[lane] = previous[lane] + previous[lane ^ offset];
                    }
                }
                let inverse = warp_sums[0].recip();
                for key_row in 0..VALID {
                    expected_probability.push(probabilities[row_offset + key_row]);
                    scalar_probability.push(scalar[key_row]);
                    tree_probability.push((row[key_row] - max).exp() * inverse);
                }
            }
        }
        let expected_probability =
            Tensor::new(vec![HEADS, VALID, VALID], expected_probability).unwrap();
        for (name, values) in [
            ("scalar", scalar_probability),
            ("cuda_tree", tree_probability),
        ] {
            let actual = Tensor::new(vec![HEADS, VALID, VALID], values).unwrap();
            let metrics = compare_tensors(&actual, &expected_probability).unwrap();
            eprintln!(
                "UMT5 CUDA attention probabilities {name}: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                metrics.cosine_similarity,
                metrics.maximum_absolute_error,
                metrics.mean_absolute_error,
            );
        }
        drop(biased_scores);
        let (_, expected_context) = read_dump(
            &Path::new(REF_DIR).join("saient_live_t5_block0_attention_context_heads_0.bin"),
        )
        .expect("missing captured head-major attention context");
        let mut expected = Vec::with_capacity(HEADS * VALID * HEAD_DIM);
        let mut valid_sequential = Vec::with_capacity(expected.capacity());
        let mut full_sequential = Vec::with_capacity(expected.capacity());
        for head in 0..HEADS {
            let offset = head * HEAD_DIM;
            for query_row in 0..VALID {
                for channel in 0..HEAD_DIM {
                    expected.push(expected_context[(head * ROWS + query_row) * HEAD_DIM + channel]);
                    let mut valid_sum = 0.0f32;
                    let mut full_sum = 0.0f32;
                    for key_row in 0..ROWS {
                        let probability = round_to_tf32(
                            probabilities[(head * ROWS + query_row) * ROWS + key_row],
                        );
                        let value = round_to_tf32(value[key_row * WIDTH + offset + channel]);
                        full_sum = probability.mul_add(value, full_sum);
                        if key_row < VALID {
                            valid_sum = probability.mul_add(value, valid_sum);
                        }
                    }
                    valid_sequential.push(valid_sum);
                    full_sequential.push(full_sum);
                }
            }
        }
        let expected = Tensor::new(vec![HEADS, VALID, HEAD_DIM], expected).unwrap();
        for (name, values) in [
            ("valid_sequential", valid_sequential),
            ("full_sequential", full_sequential),
        ] {
            let actual = Tensor::new(vec![HEADS, VALID, HEAD_DIM], values).unwrap();
            let metrics = compare_tensors(&actual, &expected).unwrap();
            eprintln!(
                "UMT5 CUDA attention context {name}: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                metrics.cosine_similarity,
                metrics.maximum_absolute_error,
                metrics.mean_absolute_error,
            );
        }
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

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads UMT5-XXL and runs 24 block-staged Vulkan blocks; run explicitly"]
    fn cond_vulkan_matches_reference() {
        compare_vulkan("cond", "a red fox", "cond_crossattn.bin");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads UMT5-XXL and runs 24 block-staged Vulkan blocks; run explicitly"]
    fn uncond_vulkan_matches_reference() {
        compare_vulkan("uncond", "", "uncond_crossattn.bin");
    }
}
