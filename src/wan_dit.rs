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

use crate::backend::{DeviceTensor, LinearWeightHandle, PreparedVectorHandle, TensorBackend};
use crate::dequant;
use crate::gguf::{GgufFile, TensorInfo};
use crate::tensor::Tensor;
use crate::wan_rope;
use anyhow::{Context, Result, anyhow, bail};

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

pub(crate) struct PreparedTimeEmbedding {
    time0: LinearWeightHandle,
    time2: LinearWeightHandle,
    time_projection: LinearWeightHandle,
}

pub(crate) struct PreparedWanEnvelope {
    patch_embedding: LinearWeightHandle,
    text0: LinearWeightHandle,
    text2: LinearWeightHandle,
    time: PreparedTimeEmbedding,
    head_modulation: PreparedVectorHandle,
    head: LinearWeightHandle,
}

pub(crate) struct PreparedBlockPreAttention {
    block_index: usize,
    modulation: PreparedVectorHandle,
}

pub(crate) struct PreparedBlockSelfAttention {
    block_index: usize,
    query: LinearWeightHandle,
    key: LinearWeightHandle,
    value: LinearWeightHandle,
    output: LinearWeightHandle,
    norm_query: PreparedVectorHandle,
    norm_key: PreparedVectorHandle,
}

pub(crate) struct SelfAttentionQkv {
    query: DeviceTensor,
    key: DeviceTensor,
    value: DeviceTensor,
}

pub(crate) struct PreparedBlockCrossAttention {
    block_index: usize,
    query: LinearWeightHandle,
    key: LinearWeightHandle,
    value: LinearWeightHandle,
    output: LinearWeightHandle,
    norm_query: PreparedVectorHandle,
    norm_key: PreparedVectorHandle,
    norm3_weight: PreparedVectorHandle,
    norm3_bias: PreparedVectorHandle,
}

pub(crate) struct PreparedBlockFfn {
    block_index: usize,
    input: LinearWeightHandle,
    output: LinearWeightHandle,
}

/// All immutable resources needed to execute one Wan transformer block.
///
/// Keeping this as a block-scoped object lets callers stage one block, execute it, and drop its
/// weights before staging the next block. That is the intended bounded-memory path for the full
/// 30-block DiT.
pub(crate) struct PreparedWanBlock {
    block_index: usize,
    pre_attention: PreparedBlockPreAttention,
    self_attention: PreparedBlockSelfAttention,
    cross_attention: PreparedBlockCrossAttention,
    ffn: PreparedBlockFfn,
}

pub(crate) struct BlockResidualOutput {
    branch: DeviceTensor,
    residual: DeviceTensor,
}

impl<'a> WanDit<'a> {
    pub fn load(gguf: &'a GgufFile, cfg: WanConfig) -> Result<Self> {
        let map = gguf.tensor_map();
        let p = "model.diffusion_model";
        let info = |n: String| -> Result<&'a TensorInfo> {
            map.get(n.as_str())
                .copied()
                .ok_or_else(|| anyhow!("missing tensor {n}"))
        };
        let vals = |t: &TensorInfo| dequant::dequant(gguf.tensor_data(t), t.ggml_type, t.n_elems());
        let lin = |base: String, in_dim: usize, out_dim: usize| -> Result<Linear<'a>> {
            let w = info(format!("{base}.weight"))?;
            let b = map.get(format!("{base}.bias").as_str()).map(|t| vals(t));
            Ok(Linear {
                w,
                b,
                in_dim,
                out_dim,
            })
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
            head: lin(
                format!("{p}.head.head"),
                d,
                cfg.out_dim * cfg.patch.0 * cfg.patch.1 * cfg.patch.2,
            )?,
            blocks,
            gguf,
            cfg,
        })
    }

    fn apply(&self, l: &Linear<'a>, x: &[f32], n: usize) -> Vec<f32> {
        let mut y = dequant::gemm(
            x,
            self.gguf.tensor_data(l.w),
            l.w.ggml_type,
            l.in_dim,
            l.out_dim,
            n,
        );
        if let Some(b) = &l.b {
            for t in 0..n {
                for j in 0..l.out_dim {
                    y[t * l.out_dim + j] += b[j];
                }
            }
        }
        y
    }

    fn materialize_linear(&self, linear: &Linear<'a>) -> Result<(Tensor, Option<Tensor>)> {
        let weight = Tensor::new(
            vec![linear.out_dim, linear.in_dim],
            dequant::dequant(
                self.gguf.tensor_data(linear.w),
                linear.w.ggml_type,
                linear.w.n_elems(),
            ),
        )?;
        let bias = linear
            .b
            .as_ref()
            .map(|bias| Tensor::new(vec![linear.out_dim], bias.clone()))
            .transpose()?;
        Ok((weight, bias))
    }

    fn prepare_backend_linear(
        &self,
        backend: &dyn TensorBackend,
        linear: &Linear<'a>,
    ) -> Result<LinearWeightHandle> {
        let (weight, bias) = self.materialize_linear(linear)?;
        backend.prepare_linear(&weight, bias.as_ref())
    }

    pub(crate) fn prepare_time_embedding(
        &self,
        backend: &dyn TensorBackend,
    ) -> Result<PreparedTimeEmbedding> {
        Ok(PreparedTimeEmbedding {
            time0: self.prepare_backend_linear(backend, &self.time0)?,
            time2: self.prepare_backend_linear(backend, &self.time2)?,
            time_projection: self.prepare_backend_linear(backend, &self.time_proj)?,
        })
    }

    pub(crate) fn prepare_wan_envelope(
        &self,
        backend: &dyn TensorBackend,
    ) -> Result<PreparedWanEnvelope> {
        let patch_volume = self.cfg.patch.0 * self.cfg.patch.1 * self.cfg.patch.2;
        let patch_weight = Tensor::new(
            vec![self.cfg.dim, self.cfg.in_dim * patch_volume],
            self.patch_w.clone(),
        )?;
        let patch_bias = Tensor::new(vec![self.cfg.dim], self.patch_b.clone())?;
        let head_modulation = Tensor::new(vec![2 * self.cfg.dim], self.head_mod.clone())?;
        Ok(PreparedWanEnvelope {
            patch_embedding: backend.prepare_linear(&patch_weight, Some(&patch_bias))?,
            text0: self.prepare_backend_linear(backend, &self.text0)?,
            text2: self.prepare_backend_linear(backend, &self.text2)?,
            time: self.prepare_time_embedding(backend)?,
            head_modulation: backend.prepare_vector(&head_modulation)?,
            head: self.prepare_backend_linear(backend, &self.head)?,
        })
    }

    fn time_embedding_scalar(&self, timestep: f32) -> (Tensor, Tensor) {
        let te = timestep_embedding(timestep, self.cfg.freq_dim);
        let mut e = self.apply(&self.time0, &te, 1);
        for value in &mut e {
            *value = silu(*value);
        }
        let e = self.apply(&self.time2, &e, 1);
        let e_silu = e.iter().map(|&value| silu(value)).collect::<Vec<_>>();
        let e0 = self.apply(&self.time_proj, &e_silu, 1);
        (
            Tensor::new(vec![1, self.cfg.dim], e).expect("Wan time embedding shape is static"),
            Tensor::new(vec![1, self.cfg.dim * 6], e0)
                .expect("Wan time projection shape is static"),
        )
    }

    pub(crate) fn time_embedding_with_backend(
        &self,
        timestep: f32,
        backend: &dyn TensorBackend,
        prepared: &PreparedTimeEmbedding,
    ) -> Result<(Tensor, Tensor)> {
        let (e, e0) = self.time_embedding_device(timestep, backend, prepared)?;
        let e_host = backend.download_tensor(&e)?;
        let e0_host = backend.download_tensor(&e0)?;
        Ok((e_host, e0_host))
    }

    pub(crate) fn time_embedding_device(
        &self,
        timestep: f32,
        backend: &dyn TensorBackend,
        prepared: &PreparedTimeEmbedding,
    ) -> Result<(DeviceTensor, DeviceTensor)> {
        let timestep = Tensor::new(
            vec![1, self.cfg.freq_dim],
            timestep_embedding(timestep, self.cfg.freq_dim),
        )?;
        let timestep = backend.upload_tensor(&timestep)?;
        let e = backend.linear_prepared(&timestep, &prepared.time0)?;
        let e = backend.silu_device(&e)?;
        let e = backend.linear_prepared(&e, &prepared.time2)?;
        let e_silu = backend.silu_device(&e)?;
        let e0 = backend.linear_prepared(&e_silu, &prepared.time_projection)?;
        Ok((e, e0))
    }

    pub(crate) fn prepare_block_pre_attention(
        &self,
        block_index: usize,
        backend: &dyn TensorBackend,
    ) -> Result<PreparedBlockPreAttention> {
        let block = self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?;
        let modulation = Tensor::new(vec![self.cfg.dim * 6], block.modulation.clone())?;
        Ok(PreparedBlockPreAttention {
            block_index,
            modulation: backend.prepare_vector(&modulation)?,
        })
    }

    fn block_pre_attention_scalar(
        &self,
        block_index: usize,
        input: &Tensor,
        e0: &Tensor,
    ) -> Result<Tensor> {
        let block = self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?;
        let [rows, width]: [usize; 2] = input
            .shape()
            .try_into()
            .context("Wan block pre-attention input must be rank two")?;
        if rows == 0 || width != self.cfg.dim {
            bail!(
                "Wan block pre-attention input shape {:?} must be [rows, {}]",
                input.shape(),
                self.cfg.dim
            );
        }
        if e0.len() != self.cfg.dim * 6 {
            bail!(
                "Wan shared modulation has {} values, expected {}",
                e0.len(),
                self.cfg.dim * 6
            );
        }
        let modulation = e0
            .data()
            .iter()
            .zip(&block.modulation)
            .map(|(shared, block)| shared + block)
            .collect::<Vec<_>>();
        let shift = &modulation[..width];
        let scale = &modulation[width..2 * width];
        let mut output = input.data().to_vec();
        for row in output.chunks_exact_mut(width) {
            layer_norm(row, self.cfg.eps);
            for column in 0..width {
                row[column] = row[column] * (1.0 + scale[column]) + shift[column];
            }
        }
        Tensor::new(input.shape().to_vec(), output)
    }

    pub(crate) fn block_pre_attention_with_backend(
        &self,
        input: &DeviceTensor,
        e0: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedBlockPreAttention,
    ) -> Result<DeviceTensor> {
        let modulation = self.block_modulation_with_backend(e0, backend, prepared)?;
        self.block_pre_attention_from_modulation_with_backend(input, &modulation, backend)
    }

    pub(crate) fn block_modulation_with_backend(
        &self,
        e0: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedBlockPreAttention,
    ) -> Result<DeviceTensor> {
        if e0.shape().iter().product::<usize>() != self.cfg.dim * 6 {
            bail!(
                "Wan resident shared modulation shape {:?} must contain {} values",
                e0.shape(),
                self.cfg.dim * 6
            );
        }
        if prepared.block_index >= self.blocks.len() {
            bail!("prepared Wan block index is out of range");
        }
        backend.add_vector_device(e0, &prepared.modulation)
    }

    pub(crate) fn block_pre_attention_from_modulation_with_backend(
        &self,
        input: &DeviceTensor,
        modulation: &DeviceTensor,
        backend: &dyn TensorBackend,
    ) -> Result<DeviceTensor> {
        if input.shape().len() != 2 || input.shape()[1] != self.cfg.dim {
            bail!(
                "Wan resident block input shape {:?} must be [rows, {}]",
                input.shape(),
                self.cfg.dim
            );
        }
        if modulation.shape().iter().product::<usize>() != self.cfg.dim * 6 {
            bail!(
                "Wan resident block modulation shape {:?} must contain {} values",
                modulation.shape(),
                self.cfg.dim * 6
            );
        }
        let normalized = backend.layer_norm_device(input, None, None, self.cfg.eps)?;
        backend.wan_modulate_device(&normalized, modulation, 0, 1)
    }

    pub(crate) fn prepare_block_self_attention(
        &self,
        block_index: usize,
        backend: &dyn TensorBackend,
    ) -> Result<PreparedBlockSelfAttention> {
        let attention = &self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?
            .self_attn;
        let norm_query = Tensor::new(vec![self.cfg.dim], attention.norm_q.clone())?;
        let norm_key = Tensor::new(vec![self.cfg.dim], attention.norm_k.clone())?;
        Ok(PreparedBlockSelfAttention {
            block_index,
            query: self.prepare_backend_linear(backend, &attention.q)?,
            key: self.prepare_backend_linear(backend, &attention.k)?,
            value: self.prepare_backend_linear(backend, &attention.v)?,
            output: self.prepare_backend_linear(backend, &attention.o)?,
            norm_query: backend.prepare_vector(&norm_query)?,
            norm_key: backend.prepare_vector(&norm_key)?,
        })
    }

    fn block_self_attention_qkv_scalar(
        &self,
        block_index: usize,
        input: &Tensor,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let attention = &self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?
            .self_attn;
        let [rows, width]: [usize; 2] = input
            .shape()
            .try_into()
            .context("Wan self-attention input must be rank two")?;
        if rows == 0 || width != self.cfg.dim {
            bail!(
                "Wan self-attention input shape {:?} must be [rows, {}]",
                input.shape(),
                self.cfg.dim
            );
        }
        let position_stride = self.cfg.head_dim / 2 * wan_rope::PAIR_STRIDE;
        if positions.len() != rows * position_stride {
            bail!(
                "Wan RoPE position tensor has {} values, expected {}",
                positions.len(),
                rows * position_stride
            );
        }

        let mut query = self.apply(&attention.q, input.data(), rows);
        let mut key = self.apply(&attention.k, input.data(), rows);
        let value = self.apply(&attention.v, input.data(), rows);
        for row in query.chunks_exact_mut(width) {
            rms_norm(row, &attention.norm_q, self.cfg.eps);
        }
        for row in key.chunks_exact_mut(width) {
            rms_norm(row, &attention.norm_k, self.cfg.eps);
        }
        for row in 0..rows {
            let position = &positions.data()[row * position_stride..(row + 1) * position_stride];
            for head in 0..self.cfg.num_heads {
                let offset = row * width + head * self.cfg.head_dim;
                wan_rope::apply_rope(&mut query[offset..offset + self.cfg.head_dim], position);
                wan_rope::apply_rope(&mut key[offset..offset + self.cfg.head_dim], position);
            }
        }
        Ok((
            Tensor::new(input.shape().to_vec(), query)?,
            Tensor::new(input.shape().to_vec(), key)?,
            Tensor::new(input.shape().to_vec(), value)?,
        ))
    }

    pub(crate) fn block_self_attention_qkv_with_backend(
        &self,
        input: &DeviceTensor,
        positions: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedBlockSelfAttention,
    ) -> Result<SelfAttentionQkv> {
        if prepared.block_index >= self.blocks.len() {
            bail!("prepared Wan self-attention block index is out of range");
        }
        if input.shape().len() != 2 || input.shape()[1] != self.cfg.dim {
            bail!(
                "Wan resident self-attention input shape {:?} must be [rows, {}]",
                input.shape(),
                self.cfg.dim
            );
        }
        let rows = input.shape()[0];
        let expected_positions = rows * self.cfg.head_dim / 2 * wan_rope::PAIR_STRIDE;
        if positions.shape().iter().product::<usize>() != expected_positions {
            bail!(
                "Wan resident RoPE position shape {:?} must contain {expected_positions} values",
                positions.shape()
            );
        }

        let query = backend.linear_prepared(input, &prepared.query)?;
        let key = backend.linear_prepared(input, &prepared.key)?;
        let value = backend.linear_prepared(input, &prepared.value)?;
        let query = backend.rms_norm_device(&query, &prepared.norm_query, self.cfg.eps)?;
        let key = backend.rms_norm_device(&key, &prepared.norm_key, self.cfg.eps)?;
        let query =
            backend.rope_device(&query, positions, self.cfg.num_heads, self.cfg.head_dim)?;
        let key = backend.rope_device(&key, positions, self.cfg.num_heads, self.cfg.head_dim)?;
        Ok(SelfAttentionQkv { query, key, value })
    }

    fn block_self_attention_scalar(
        &self,
        block_index: usize,
        input: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let attention = &self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?
            .self_attn;
        let [rows, width]: [usize; 2] = input
            .shape()
            .try_into()
            .context("Wan self-attention input must be rank two")?;
        let position_values = rows * self.cfg.head_dim / 2 * wan_rope::PAIR_STRIDE;
        if rows == 0 || width != self.cfg.dim || positions.len() != position_values {
            bail!("Wan scalar self-attention input or position shape is invalid");
        }
        Tensor::new(
            input.shape().to_vec(),
            self.attention(
                attention,
                input.data(),
                input.data(),
                rows,
                rows,
                Some(positions.data()),
            ),
        )
    }

    pub(crate) fn block_self_attention_output_with_backend(
        &self,
        qkv: &SelfAttentionQkv,
        backend: &dyn TensorBackend,
        prepared: &PreparedBlockSelfAttention,
    ) -> Result<DeviceTensor> {
        if prepared.block_index >= self.blocks.len() {
            bail!("prepared Wan self-attention block index is out of range");
        }
        let scale = 1.0 / (self.cfg.head_dim as f32).sqrt();
        let scores = backend.attention_scores_device(
            &qkv.query,
            &qkv.key,
            self.cfg.num_heads,
            self.cfg.head_dim,
            scale,
        )?;
        let probabilities = backend.softmax_device(&scores)?;
        let context = backend.attention_values_device(
            &probabilities,
            &qkv.value,
            self.cfg.num_heads,
            self.cfg.head_dim,
        )?;
        backend.linear_prepared(&context, &prepared.output)
    }

    pub(crate) fn block_self_attention_with_backend(
        &self,
        input: &DeviceTensor,
        positions: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedBlockSelfAttention,
    ) -> Result<DeviceTensor> {
        let qkv =
            self.block_self_attention_qkv_with_backend(input, positions, backend, prepared)?;
        self.block_self_attention_output_with_backend(&qkv, backend, prepared)
    }

    fn block_self_attention_residual_scalar(
        &self,
        block_index: usize,
        input: &Tensor,
        e0: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let block = self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?;
        if e0.len() != self.cfg.dim * 6 {
            bail!("Wan shared modulation has the wrong size");
        }
        let pre_attention = self.block_pre_attention_scalar(block_index, input, e0)?;
        let attention = self.block_self_attention_scalar(block_index, &pre_attention, positions)?;
        let gate_offset = 2 * self.cfg.dim;
        let mut output = input.data().to_vec();
        for (index, value) in output.iter_mut().enumerate() {
            let channel = index % self.cfg.dim;
            let gate = e0.data()[gate_offset + channel] + block.modulation[gate_offset + channel];
            *value += attention.data()[index] * gate;
        }
        Tensor::new(input.shape().to_vec(), output)
    }

    pub(crate) fn block_self_attention_residual_with_backend(
        &self,
        residual: &DeviceTensor,
        attention: &DeviceTensor,
        modulation: &DeviceTensor,
        backend: &dyn TensorBackend,
    ) -> Result<DeviceTensor> {
        if residual.shape() != attention.shape()
            || residual.shape().last().copied() != Some(self.cfg.dim)
        {
            bail!("Wan self-attention residual tensors have incompatible shapes");
        }
        let gated = backend.multiply_vector_chunk_device(attention, modulation, 2)?;
        backend.add_device(residual, &gated)
    }

    pub(crate) fn prepare_block_cross_attention(
        &self,
        block_index: usize,
        backend: &dyn TensorBackend,
    ) -> Result<PreparedBlockCrossAttention> {
        let block = self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?;
        let attention = &block.cross_attn;
        let norm_query = Tensor::new(vec![self.cfg.dim], attention.norm_q.clone())?;
        let norm_key = Tensor::new(vec![self.cfg.dim], attention.norm_k.clone())?;
        let norm3_weight = Tensor::new(vec![self.cfg.dim], block.norm3_w.clone())?;
        let norm3_bias = Tensor::new(vec![self.cfg.dim], block.norm3_b.clone())?;
        Ok(PreparedBlockCrossAttention {
            block_index,
            query: self.prepare_backend_linear(backend, &attention.q)?,
            key: self.prepare_backend_linear(backend, &attention.k)?,
            value: self.prepare_backend_linear(backend, &attention.v)?,
            output: self.prepare_backend_linear(backend, &attention.o)?,
            norm_query: backend.prepare_vector(&norm_query)?,
            norm_key: backend.prepare_vector(&norm_key)?,
            norm3_weight: backend.prepare_vector(&norm3_weight)?,
            norm3_bias: backend.prepare_vector(&norm3_bias)?,
        })
    }

    fn block_cross_attention_scalar(
        &self,
        block_index: usize,
        input: &Tensor,
        context: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let block = self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?;
        let [rows, width]: [usize; 2] = input
            .shape()
            .try_into()
            .context("Wan cross-attention input must be rank two")?;
        let [context_rows, context_width]: [usize; 2] = context
            .shape()
            .try_into()
            .context("Wan cross-attention context must be rank two")?;
        if rows == 0 || context_rows == 0 || width != self.cfg.dim || context_width != self.cfg.dim
        {
            bail!("Wan cross-attention input or context shape is invalid");
        }
        let mut normalized = input.data().to_vec();
        for row in normalized.chunks_exact_mut(width) {
            layer_norm_affine(row, &block.norm3_w, &block.norm3_b, self.cfg.eps);
        }
        let branch = self.attention(
            &block.cross_attn,
            &normalized,
            context.data(),
            rows,
            context_rows,
            None,
        );
        let residual = input
            .data()
            .iter()
            .zip(&branch)
            .map(|(input, branch)| input + branch)
            .collect();
        Ok((
            Tensor::new(input.shape().to_vec(), branch)?,
            Tensor::new(input.shape().to_vec(), residual)?,
        ))
    }

    pub(crate) fn block_cross_attention_with_backend(
        &self,
        input: &DeviceTensor,
        context: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedBlockCrossAttention,
    ) -> Result<BlockResidualOutput> {
        if prepared.block_index >= self.blocks.len() {
            bail!("prepared Wan cross-attention block index is out of range");
        }
        if input.shape().len() != 2
            || context.shape().len() != 2
            || input.shape()[1] != self.cfg.dim
            || context.shape()[1] != self.cfg.dim
        {
            bail!("Wan resident cross-attention input or context shape is invalid");
        }
        let normalized = backend.layer_norm_device(
            input,
            Some(&prepared.norm3_weight),
            Some(&prepared.norm3_bias),
            self.cfg.eps,
        )?;
        let query = backend.linear_prepared(&normalized, &prepared.query)?;
        let key = backend.linear_prepared(context, &prepared.key)?;
        let value = backend.linear_prepared(context, &prepared.value)?;
        let query = backend.rms_norm_device(&query, &prepared.norm_query, self.cfg.eps)?;
        let key = backend.rms_norm_device(&key, &prepared.norm_key, self.cfg.eps)?;
        let scale = 1.0 / (self.cfg.head_dim as f32).sqrt();
        let scores = backend.attention_scores_device(
            &query,
            &key,
            self.cfg.num_heads,
            self.cfg.head_dim,
            scale,
        )?;
        let probabilities = backend.softmax_device(&scores)?;
        let attention = backend.attention_values_device(
            &probabilities,
            &value,
            self.cfg.num_heads,
            self.cfg.head_dim,
        )?;
        let branch = backend.linear_prepared(&attention, &prepared.output)?;
        let residual = backend.add_device(input, &branch)?;
        Ok(BlockResidualOutput { branch, residual })
    }

    pub(crate) fn prepare_block_ffn(
        &self,
        block_index: usize,
        backend: &dyn TensorBackend,
    ) -> Result<PreparedBlockFfn> {
        let block = self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?;
        Ok(PreparedBlockFfn {
            block_index,
            input: self.prepare_backend_linear(backend, &block.ffn0)?,
            output: self.prepare_backend_linear(backend, &block.ffn2)?,
        })
    }

    fn block_ffn_scalar(
        &self,
        block_index: usize,
        input: &Tensor,
        e0: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let block = self
            .blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?;
        let [rows, width]: [usize; 2] = input
            .shape()
            .try_into()
            .context("Wan FFN input must be rank two")?;
        if rows == 0 || width != self.cfg.dim || e0.len() != 6 * self.cfg.dim {
            bail!("Wan FFN input or shared modulation shape is invalid");
        }
        let modulation = e0
            .data()
            .iter()
            .zip(&block.modulation)
            .map(|(shared, block)| shared + block)
            .collect::<Vec<_>>();
        let shift = &modulation[3 * width..4 * width];
        let scale = &modulation[4 * width..5 * width];
        let gate = &modulation[5 * width..6 * width];
        let mut normalized = input.data().to_vec();
        for row in normalized.chunks_exact_mut(width) {
            layer_norm(row, self.cfg.eps);
            for column in 0..width {
                row[column] = row[column] * (1.0 + scale[column]) + shift[column];
            }
        }
        let mut branch = self.apply(&block.ffn0, &normalized, rows);
        for value in &mut branch {
            *value = gelu(*value);
        }
        let branch = self.apply(&block.ffn2, &branch, rows);
        let residual = input
            .data()
            .iter()
            .enumerate()
            .map(|(index, input)| input + branch[index] * gate[index % width])
            .collect();
        Ok((
            Tensor::new(input.shape().to_vec(), branch)?,
            Tensor::new(input.shape().to_vec(), residual)?,
        ))
    }

    pub(crate) fn block_ffn_with_backend(
        &self,
        input: &DeviceTensor,
        modulation: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedBlockFfn,
    ) -> Result<BlockResidualOutput> {
        if prepared.block_index >= self.blocks.len()
            || input.shape().len() != 2
            || input.shape()[1] != self.cfg.dim
            || modulation.shape().iter().product::<usize>() != 6 * self.cfg.dim
        {
            bail!("Wan resident FFN input or prepared block is invalid");
        }
        let normalized = backend.layer_norm_device(input, None, None, self.cfg.eps)?;
        let normalized = backend.wan_modulate_device(&normalized, modulation, 3, 4)?;
        let branch = backend.linear_prepared(&normalized, &prepared.input)?;
        let branch = backend.gelu_tanh_device(&branch)?;
        let branch = backend.linear_prepared(&branch, &prepared.output)?;
        let gated = backend.multiply_vector_chunk_device(&branch, modulation, 5)?;
        let residual = backend.add_device(input, &gated)?;
        Ok(BlockResidualOutput { branch, residual })
    }

    pub(crate) fn prepare_wan_block(
        &self,
        block_index: usize,
        backend: &dyn TensorBackend,
    ) -> Result<PreparedWanBlock> {
        self.blocks
            .get(block_index)
            .with_context(|| format!("Wan block index {block_index} is out of range"))?;
        Ok(PreparedWanBlock {
            block_index,
            pre_attention: self.prepare_block_pre_attention(block_index, backend)?,
            self_attention: self.prepare_block_self_attention(block_index, backend)?,
            cross_attention: self.prepare_block_cross_attention(block_index, backend)?,
            ffn: self.prepare_block_ffn(block_index, backend)?,
        })
    }

    fn wan_block_scalar(
        &self,
        block_index: usize,
        input: &Tensor,
        context: &Tensor,
        e0: &Tensor,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let self_residual =
            self.block_self_attention_residual_scalar(block_index, input, e0, positions)?;
        let (_, cross_residual) =
            self.block_cross_attention_scalar(block_index, &self_residual, context)?;
        let (_, output) = self.block_ffn_scalar(block_index, &cross_residual, e0)?;
        Ok(output)
    }

    pub(crate) fn wan_block_with_backend(
        &self,
        input: &DeviceTensor,
        context: &DeviceTensor,
        e0: &DeviceTensor,
        positions: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedWanBlock,
    ) -> Result<DeviceTensor> {
        if prepared.block_index >= self.blocks.len()
            || prepared.pre_attention.block_index != prepared.block_index
            || prepared.self_attention.block_index != prepared.block_index
            || prepared.cross_attention.block_index != prepared.block_index
            || prepared.ffn.block_index != prepared.block_index
        {
            bail!("Wan prepared block resources have inconsistent indices");
        }

        let modulation =
            self.block_modulation_with_backend(e0, backend, &prepared.pre_attention)?;
        let pre_attention =
            self.block_pre_attention_from_modulation_with_backend(input, &modulation, backend)?;
        let self_attention = self.block_self_attention_with_backend(
            &pre_attention,
            positions,
            backend,
            &prepared.self_attention,
        )?;
        let self_residual = self.block_self_attention_residual_with_backend(
            input,
            &self_attention,
            &modulation,
            backend,
        )?;
        let cross = self.block_cross_attention_with_backend(
            &self_residual,
            context,
            backend,
            &prepared.cross_attention,
        )?;
        let output =
            self.block_ffn_with_backend(&cross.residual, &modulation, backend, &prepared.ffn)?;
        Ok(output.residual)
    }

    pub(crate) fn wan_blocks_with_backend(
        &self,
        input: &DeviceTensor,
        context: &DeviceTensor,
        e0: &DeviceTensor,
        positions: &DeviceTensor,
        backend: &dyn TensorBackend,
    ) -> Result<DeviceTensor> {
        let mut state = input.clone();
        for block_index in 0..self.cfg.num_layers {
            let prepared = self.prepare_wan_block(block_index, backend)?;
            state =
                self.wan_block_with_backend(&state, context, e0, positions, backend, &prepared)?;
            // `prepared` is intentionally block-scoped. Its device-local weights are released
            // here before the next block is staged.
        }
        Ok(state)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_backend(
        &self,
        latent: &DeviceTensor,
        t: usize,
        h: usize,
        w: usize,
        timestep: f32,
        context: &DeviceTensor,
        n_ctx: usize,
        backend: &dyn TensorBackend,
        prepared: &PreparedWanEnvelope,
    ) -> Result<DeviceTensor> {
        if latent.shape() != [self.cfg.in_dim, t, h, w] {
            bail!(
                "Wan resident latent shape {:?} must be [{},{t},{h},{w}]",
                latent.shape(),
                self.cfg.in_dim
            );
        }
        if context.shape() != [n_ctx, self.cfg.text_dim] || n_ctx == 0 {
            bail!(
                "Wan resident text context shape {:?} must be [{n_ctx},{}]",
                context.shape(),
                self.cfg.text_dim
            );
        }
        let (patch_t, patch_h, patch_w) = self.cfg.patch;
        if t % patch_t != 0 || h % patch_h != 0 || w % patch_w != 0 {
            bail!("Wan resident latent dimensions must be divisible by the patch size");
        }
        let token_count = (t / patch_t) * (h / patch_h) * (w / patch_w);

        let patches = backend.patchify_device(latent, self.cfg.patch)?;
        let tokens = backend.linear_prepared(&patches, &prepared.patch_embedding)?;
        let (e, e0) = self.time_embedding_device(timestep, backend, &prepared.time)?;
        let text = backend.linear_prepared(context, &prepared.text0)?;
        let text = backend.gelu_tanh_device(&text)?;
        let text = backend.linear_prepared(&text, &prepared.text2)?;
        let positions = Tensor::new(
            vec![token_count, self.cfg.head_dim / 2, wan_rope::PAIR_STRIDE],
            wan_rope::wan_pe(
                t,
                h,
                w,
                patch_t,
                patch_h,
                patch_w,
                &self.cfg.axes_dim,
                self.cfg.theta,
            ),
        )?;
        let positions = backend.upload_tensor(&positions)?;
        let tokens = self.wan_blocks_with_backend(&tokens, &text, &e0, &positions, backend)?;
        let normalized = backend.layer_norm_device(&tokens, None, None, self.cfg.eps)?;
        let normalized =
            backend.wan_head_modulate_device(&normalized, &e, &prepared.head_modulation)?;
        let output_tokens = backend.linear_prepared(&normalized, &prepared.head)?;
        backend.unpatchify_device(&output_tokens, self.cfg.out_dim, (t, h, w), self.cfg.patch)
    }

    /// Gather each `1x2x2` patch of the latent into 64 values and project to `dim`.
    ///
    /// The Conv3d kernel equals its stride, so this is a patchify rather than a convolution. The
    /// weight is `[KW, KH, KD, in*out]` with the flattened index `out * in_channels + in`, which
    /// is PyTorch's `[out, in, kd, kh, kw]` ordering seen through ggml's reversed dims.
    fn patchify(
        &self,
        latent: &[f32],
        t: usize,
        h: usize,
        w: usize,
    ) -> (Vec<f32>, usize, usize, usize) {
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
                                    let li = ((ci * t + ti * pt + kd) * h + hj * ph + kh) * w
                                        + wk * pw
                                        + kw;
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
        let (e, e0) = self.time_embedding_scalar(timestep);
        let e = e.data();
        let e0 = e0.data();

        // text embedding: 4096 -> dim
        let mut ctx = self.apply(&self.text0, context, n_ctx);
        for v in ctx.iter_mut() {
            *v = gelu(*v);
        }
        let ctx = self.apply(&self.text2, &ctx, n_ctx);

        let pe = wan_rope::wan_pe(
            t,
            h,
            w,
            cfg.patch.0,
            cfg.patch.1,
            cfg.patch.2,
            &cfg.axes_dim,
            cfg.theta,
        );

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
                layer_norm_affine(
                    &mut y[i * d..(i + 1) * d],
                    &blk.norm3_w,
                    &blk.norm3_b,
                    cfg.eps,
                );
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
    fn unpatchify(
        &self,
        tok: &[f32],
        tl: usize,
        hl: usize,
        wl: usize,
        t: usize,
        h: usize,
        w: usize,
    ) -> Vec<f32> {
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
                                    let src =
                                        token * oc * kvol + ((kd * ph + kh) * pw + kw) * oc + c;
                                    let dst = ((c * t + ti * pt + kd) * h + hj * ph + kh) * w
                                        + wk * pw
                                        + kw;
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
        assert!(
            e.iter().all(|v| v.abs() <= 1.0 + 1e-6),
            "sin/cos must be bounded"
        );
        // freq 0 gives cos(t)=cos(1000), sin(t)=sin(1000)
        assert!((e[0] - 1000f32.cos()).abs() < 1e-4);
        assert!((e[128] - 1000f32.sin()).abs() < 1e-4);
    }

    #[test]
    fn timestep_zero_is_cos_one_sin_zero() {
        let e = timestep_embedding(0.0, 256);
        assert!(
            e[..128].iter().all(|v| (v - 1.0).abs() < 1e-6),
            "cos half must be 1"
        );
        assert!(
            e[128..].iter().all(|v| v.abs() < 1e-6),
            "sin half must be 0"
        );
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
        assert!(
            (silu(1.0) - 0.731_058_6).abs() < 1e-5,
            "silu(1)={}",
            silu(1.0)
        );
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-4);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads the Wan DiT and validates the complete resident Vulkan time embedding"]
    fn resident_vulkan_time_embedding_reuses_real_wan_weights() {
        use std::{path::Path, time::Instant};

        use crate::{
            backend::VULKAN_BACKEND,
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const DIT: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan2.1_t2v_1.3B_Q4_K.gguf";
        let gguf = GgufFile::open(Path::new(DIT)).unwrap();
        let dit = WanDit::load(&gguf, WanConfig::default()).unwrap();

        let scalar_started = Instant::now();
        let (scalar_e, scalar_e0) = dit.time_embedding_scalar(750.0);
        let scalar_runtime = scalar_started.elapsed();

        let before = crate::vulkan::persistence_stats().unwrap();
        let prepare_started = Instant::now();
        let prepared = dit.prepare_time_embedding(&VULKAN_BACKEND).unwrap();
        let prepare_runtime = prepare_started.elapsed();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();

        let first_started = Instant::now();
        let (first_e, first_e0) = dit
            .time_embedding_with_backend(750.0, &VULKAN_BACKEND, &prepared)
            .unwrap();
        let first_runtime = first_started.elapsed();
        let after_first = crate::vulkan::persistence_stats().unwrap();

        let second_started = Instant::now();
        let (second_e, second_e0) = dit
            .time_embedding_with_backend(750.0, &VULKAN_BACKEND, &prepared)
            .unwrap();
        let second_runtime = second_started.elapsed();
        let after_second = crate::vulkan::persistence_stats().unwrap();

        let e_metrics = compare_tensors(&first_e, &scalar_e).unwrap();
        let e0_metrics = compare_tensors(&first_e0, &scalar_e0).unwrap();
        let repeat_e = compare_tensors(&second_e, &first_e).unwrap();
        let repeat_e0 = compare_tensors(&second_e0, &first_e0).unwrap();
        e_metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_9,
                maximum_absolute_error: 0.05,
                maximum_mean_absolute_error: 0.005,
            })
            .unwrap();
        e0_metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_9,
                maximum_absolute_error: 0.05,
                maximum_mean_absolute_error: 0.005,
            })
            .unwrap();
        repeat_e
            .require(ParityTolerance {
                minimum_cosine_similarity: 1.0,
                maximum_absolute_error: 0.0,
                maximum_mean_absolute_error: 0.0,
            })
            .unwrap();
        repeat_e0
            .require(ParityTolerance {
                minimum_cosine_similarity: 1.0,
                maximum_absolute_error: 0.0,
                maximum_mean_absolute_error: 0.0,
            })
            .unwrap();

        let d = dit.cfg.dim;
        let expected_weight_bytes =
            ((d * dit.cfg.freq_dim + d * d + 6 * d * d) * std::mem::size_of::<u16>()
                + (d + d + 6 * d) * std::mem::size_of::<f32>()) as u64;
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            3,
            "one grouped upload is required for each of the three linear layers"
        );
        assert_eq!(
            after_prepare.resident_uploaded_bytes - before.resident_uploaded_bytes,
            expected_weight_bytes
        );
        assert_eq!(
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            expected_weight_bytes,
            "all three prepared matrices and biases must reside in DEVICE_LOCAL memory"
        );
        assert!(
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes
                >= expected_weight_bytes,
            "device-local Vulkan allocations must cover all prepared weights"
        );
        assert_eq!(
            after_first.resident_weight_uploads, after_prepare.resident_weight_uploads,
            "executing the prepared layer must not upload weights"
        );
        assert_eq!(
            after_second.resident_weight_uploads, after_prepare.resident_weight_uploads,
            "repeated execution must reuse every prepared weight"
        );
        assert_eq!(
            after_second.resident_device_local_bytes, after_prepare.resident_device_local_bytes,
            "execution must not replace the prepared device-local buffers"
        );
        assert_eq!(
            after_second.resident_tensor_uploads - before.resident_tensor_uploads,
            2,
            "each invocation uploads only its timestep embedding"
        );
        assert_eq!(
            after_second.resident_downloads - before.resident_downloads,
            4,
            "each invocation downloads e and e0"
        );
        assert_eq!(
            after_first.resident_allocated_bytes, after_prepare.resident_allocated_bytes,
            "temporary activations must be released after the first invocation"
        );
        assert_eq!(
            after_second.resident_allocated_bytes, after_prepare.resident_allocated_bytes,
            "temporary activations must be released after the second invocation"
        );

        println!(
            "Wan resident time embedding: e_shape={:?} e_cosine={:.9} e_max={:.9} e_mean={:.9} e0_shape={:?} e0_cosine={:.9} e0_max={:.9} e0_mean={:.9} scalar_ms={:.3} prepare_ms={:.3} first_ms={:.3} second_ms={:.3} weight_uploads={} weight_bytes={} resident_bytes={} peak_resident_bytes={} device_local_bytes={} device_local_allocation_bytes={}",
            first_e.shape(),
            e_metrics.cosine_similarity,
            e_metrics.maximum_absolute_error,
            e_metrics.mean_absolute_error,
            first_e0.shape(),
            e0_metrics.cosine_similarity,
            e0_metrics.maximum_absolute_error,
            e0_metrics.mean_absolute_error,
            scalar_runtime.as_secs_f64() * 1_000.0,
            prepare_runtime.as_secs_f64() * 1_000.0,
            first_runtime.as_secs_f64() * 1_000.0,
            second_runtime.as_secs_f64() * 1_000.0,
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            expected_weight_bytes,
            after_prepare.resident_allocated_bytes - before.resident_allocated_bytes,
            after_second.peak_resident_allocated_bytes,
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes,
        );
        crate::vulkan::print_statistics();

        drop(prepared);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes, before.resident_allocated_bytes,
            "dropping prepared weights must release their resident buffers"
        );
        assert_eq!(
            after_drop.resident_device_local_bytes, before.resident_device_local_bytes,
            "dropping prepared weights must release logical device-local residency"
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes,
            "dropping prepared weights must free device-local allocations"
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads Wan weights and validates the resident block-0 pre-attention subgraph"]
    fn resident_vulkan_block0_pre_attention_matches_scalar() {
        use std::{path::Path, time::Instant};

        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const DIT: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan2.1_t2v_1.3B_Q4_K.gguf";
        const ROWS: usize = 7;
        let gguf = GgufFile::open(Path::new(DIT)).unwrap();
        let dit = WanDit::load(&gguf, WanConfig::default()).unwrap();
        let input = Tensor::new(
            vec![ROWS, dit.cfg.dim],
            (0..ROWS * dit.cfg.dim)
                .map(|index| ((index * 17) as f32 * 0.007_812_5).sin() * 1.25)
                .collect(),
        )
        .unwrap();
        let scalar_started = Instant::now();
        let (_, scalar_e0) = dit.time_embedding_scalar(750.0);
        let scalar = dit
            .block_pre_attention_scalar(0, &input, &scalar_e0)
            .unwrap();
        let scalar_runtime = scalar_started.elapsed();

        let before = crate::vulkan::persistence_stats().unwrap();
        let prepare_started = Instant::now();
        let prepared_time = dit.prepare_time_embedding(&VULKAN_BACKEND).unwrap();
        let prepared_block = dit.prepare_block_pre_attention(0, &VULKAN_BACKEND).unwrap();
        let prepare_runtime = prepare_started.elapsed();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();

        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let time_started = Instant::now();
        let (device_e, device_e0) = dit
            .time_embedding_device(750.0, &VULKAN_BACKEND, &prepared_time)
            .unwrap();
        let time_runtime = time_started.elapsed();
        drop(device_e);

        let first_started = Instant::now();
        let first_device = dit
            .block_pre_attention_with_backend(
                &device_input,
                &device_e0,
                &VULKAN_BACKEND,
                &prepared_block,
            )
            .unwrap();
        let first_runtime = first_started.elapsed();
        let first = VULKAN_BACKEND.download_tensor(&first_device).unwrap();
        drop(first_device);
        let after_first = crate::vulkan::persistence_stats().unwrap();

        let second_started = Instant::now();
        let second_device = dit
            .block_pre_attention_with_backend(
                &device_input,
                &device_e0,
                &VULKAN_BACKEND,
                &prepared_block,
            )
            .unwrap();
        let second_runtime = second_started.elapsed();
        let second = VULKAN_BACKEND.download_tensor(&second_device).unwrap();
        drop(second_device);
        let after_second = crate::vulkan::persistence_stats().unwrap();

        let metrics = compare_tensors(&first, &scalar).unwrap();
        let repeat_metrics = compare_tensors(&second, &first).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 0.001,
                maximum_mean_absolute_error: 0.000_1,
            })
            .unwrap();
        repeat_metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 1.0,
                maximum_absolute_error: 0.0,
                maximum_mean_absolute_error: 0.0,
            })
            .unwrap();
        assert_eq!(first.shape(), &[ROWS, dit.cfg.dim]);

        let d = dit.cfg.dim;
        let time_weight_bytes = ((d * dit.cfg.freq_dim + d * d + 6 * d * d)
            * std::mem::size_of::<u16>()
            + (d + d + 6 * d) * std::mem::size_of::<f32>()) as u64;
        let block_modulation_bytes = (6 * d * std::mem::size_of::<f32>()) as u64;
        let expected_device_local_bytes = time_weight_bytes + block_modulation_bytes;
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            4,
            "three time linears and one block vector require four staging submissions"
        );
        assert_eq!(
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            expected_device_local_bytes
        );
        assert!(
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes
                >= expected_device_local_bytes
        );
        assert_eq!(
            after_second.resident_weight_uploads, after_prepare.resident_weight_uploads,
            "time and block execution must not upload prepared weights"
        );
        assert_eq!(
            after_second.resident_tensor_uploads - before.resident_tensor_uploads,
            2,
            "the token input and timestep embedding are each uploaded once"
        );
        assert_eq!(
            after_second.resident_downloads - before.resident_downloads,
            2,
            "only the two final block outputs are downloaded"
        );
        assert_eq!(
            after_first.resident_allocated_bytes, after_second.resident_allocated_bytes,
            "temporary tensors must be released after every block invocation"
        );
        println!(
            "Wan block-0 pre-attention: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9} repeat_max={:.9} scalar_ms={:.3} prepare_ms={:.3} time_ms={:.3} first_ms={:.3} second_ms={:.3} weight_uploads={} device_local_bytes={} device_local_allocation_bytes={} resident_bytes={} peak_resident_bytes={}",
            first.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            repeat_metrics.maximum_absolute_error,
            scalar_runtime.as_secs_f64() * 1_000.0,
            prepare_runtime.as_secs_f64() * 1_000.0,
            time_runtime.as_secs_f64() * 1_000.0,
            first_runtime.as_secs_f64() * 1_000.0,
            second_runtime.as_secs_f64() * 1_000.0,
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes,
            after_second.resident_allocated_bytes - before.resident_allocated_bytes,
            after_second.peak_resident_allocated_bytes,
        );
        crate::vulkan::print_statistics();

        drop(device_e0);
        drop(device_input);
        drop(prepared_block);
        drop(prepared_time);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads Wan weights and validates the complete resident Vulkan block 0"]
    fn resident_vulkan_block0_matches_scalar() {
        use std::{path::Path, time::Instant};

        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const DIT: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan2.1_t2v_1.3B_Q4_K.gguf";
        const ROWS: usize = 7;
        const CONTEXT_ROWS: usize = 5;
        let gguf = GgufFile::open(Path::new(DIT)).unwrap();
        let dit = WanDit::load(&gguf, WanConfig::default()).unwrap();
        let input = Tensor::new(
            vec![ROWS, dit.cfg.dim],
            (0..ROWS * dit.cfg.dim)
                .map(|index| ((index * 17) as f32 * 0.007_812_5).sin() * 1.25)
                .collect(),
        )
        .unwrap();
        let context = Tensor::new(
            vec![CONTEXT_ROWS, dit.cfg.dim],
            (0..CONTEXT_ROWS * dit.cfg.dim)
                .map(|index| ((index * 23) as f32 * 0.003_906_25).cos() * 0.875)
                .collect(),
        )
        .unwrap();
        let ids = (0..ROWS)
            .map(|column| [0.0, 0.0, column as f32])
            .collect::<Vec<_>>();
        let positions = Tensor::new(
            vec![ROWS, dit.cfg.head_dim / 2, wan_rope::PAIR_STRIDE],
            wan_rope::embed_nd(&ids, &dit.cfg.axes_dim, dit.cfg.theta),
        )
        .unwrap();

        let scalar_started = Instant::now();
        let (_, scalar_e0) = dit.time_embedding_scalar(750.0);
        let scalar_pre_attention = dit
            .block_pre_attention_scalar(0, &input, &scalar_e0)
            .unwrap();
        let (scalar_query, scalar_key, scalar_value) = dit
            .block_self_attention_qkv_scalar(0, &scalar_pre_attention, &positions)
            .unwrap();
        let scalar_attention = dit
            .block_self_attention_scalar(0, &scalar_pre_attention, &positions)
            .unwrap();
        let scalar_residual = dit
            .block_self_attention_residual_scalar(0, &input, &scalar_e0, &positions)
            .unwrap();
        let (scalar_cross_branch, scalar_cross_residual) = dit
            .block_cross_attention_scalar(0, &scalar_residual, &context)
            .unwrap();
        let (scalar_ffn_branch, scalar_block_output) = dit
            .block_ffn_scalar(0, &scalar_cross_residual, &scalar_e0)
            .unwrap();
        let scalar_runtime = scalar_started.elapsed();

        let before = crate::vulkan::persistence_stats().unwrap();
        let prepare_started = Instant::now();
        let prepared_time = dit.prepare_time_embedding(&VULKAN_BACKEND).unwrap();
        let prepared_block = dit.prepare_block_pre_attention(0, &VULKAN_BACKEND).unwrap();
        let prepared_attention = dit
            .prepare_block_self_attention(0, &VULKAN_BACKEND)
            .unwrap();
        let prepared_cross = dit
            .prepare_block_cross_attention(0, &VULKAN_BACKEND)
            .unwrap();
        let prepared_ffn = dit.prepare_block_ffn(0, &VULKAN_BACKEND).unwrap();
        let prepare_runtime = prepare_started.elapsed();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();

        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let device_positions = VULKAN_BACKEND.upload_tensor(&positions).unwrap();
        let device_context = VULKAN_BACKEND.upload_tensor(&context).unwrap();
        let execute_started = Instant::now();
        let (device_e, device_e0) = dit
            .time_embedding_device(750.0, &VULKAN_BACKEND, &prepared_time)
            .unwrap();
        drop(device_e);
        let device_modulation = dit
            .block_modulation_with_backend(&device_e0, &VULKAN_BACKEND, &prepared_block)
            .unwrap();
        let device_pre_attention = dit
            .block_pre_attention_from_modulation_with_backend(
                &device_input,
                &device_modulation,
                &VULKAN_BACKEND,
            )
            .unwrap();
        let qkv = dit
            .block_self_attention_qkv_with_backend(
                &device_pre_attention,
                &device_positions,
                &VULKAN_BACKEND,
                &prepared_attention,
            )
            .unwrap();
        let device_attention = dit
            .block_self_attention_output_with_backend(&qkv, &VULKAN_BACKEND, &prepared_attention)
            .unwrap();
        let device_residual = dit
            .block_self_attention_residual_with_backend(
                &device_input,
                &device_attention,
                &device_modulation,
                &VULKAN_BACKEND,
            )
            .unwrap();
        let device_cross = dit
            .block_cross_attention_with_backend(
                &device_residual,
                &device_context,
                &VULKAN_BACKEND,
                &prepared_cross,
            )
            .unwrap();
        let device_ffn = dit
            .block_ffn_with_backend(
                &device_cross.residual,
                &device_modulation,
                &VULKAN_BACKEND,
                &prepared_ffn,
            )
            .unwrap();
        let execute_runtime = execute_started.elapsed();
        let query = VULKAN_BACKEND.download_tensor(&qkv.query).unwrap();
        let key = VULKAN_BACKEND.download_tensor(&qkv.key).unwrap();
        let value = VULKAN_BACKEND.download_tensor(&qkv.value).unwrap();
        let attention = VULKAN_BACKEND.download_tensor(&device_attention).unwrap();
        let residual = VULKAN_BACKEND.download_tensor(&device_residual).unwrap();
        let cross_branch = VULKAN_BACKEND
            .download_tensor(&device_cross.branch)
            .unwrap();
        let cross_residual = VULKAN_BACKEND
            .download_tensor(&device_cross.residual)
            .unwrap();
        let ffn_branch = VULKAN_BACKEND.download_tensor(&device_ffn.branch).unwrap();
        let block_output = VULKAN_BACKEND
            .download_tensor(&device_ffn.residual)
            .unwrap();
        let after_execute = crate::vulkan::persistence_stats().unwrap();

        let query_metrics = compare_tensors(&query, &scalar_query).unwrap();
        let key_metrics = compare_tensors(&key, &scalar_key).unwrap();
        let value_metrics = compare_tensors(&value, &scalar_value).unwrap();
        let attention_metrics = compare_tensors(&attention, &scalar_attention).unwrap();
        let residual_metrics = compare_tensors(&residual, &scalar_residual).unwrap();
        let cross_branch_metrics = compare_tensors(&cross_branch, &scalar_cross_branch).unwrap();
        let cross_residual_metrics =
            compare_tensors(&cross_residual, &scalar_cross_residual).unwrap();
        let ffn_branch_metrics = compare_tensors(&ffn_branch, &scalar_ffn_branch).unwrap();
        let block_output_metrics = compare_tensors(&block_output, &scalar_block_output).unwrap();
        println!(
            "Wan block-0 self-attention raw parity: query=cos:{:.9}/max:{:.9}/mean:{:.9} key=cos:{:.9}/max:{:.9}/mean:{:.9} value=cos:{:.9}/max:{:.9}/mean:{:.9} output=cos:{:.9}/max:{:.9}/mean:{:.9}",
            query_metrics.cosine_similarity,
            query_metrics.maximum_absolute_error,
            query_metrics.mean_absolute_error,
            key_metrics.cosine_similarity,
            key_metrics.maximum_absolute_error,
            key_metrics.mean_absolute_error,
            value_metrics.cosine_similarity,
            value_metrics.maximum_absolute_error,
            value_metrics.mean_absolute_error,
            attention_metrics.cosine_similarity,
            attention_metrics.maximum_absolute_error,
            attention_metrics.mean_absolute_error,
        );
        for (name, metrics, maximum_error, mean_error) in [
            ("query", &query_metrics, 0.0015, 0.0002),
            ("key", &key_metrics, 0.0015, 0.0002),
            ("value", &value_metrics, 0.002, 0.0003),
        ] {
            metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 0.999_999,
                    maximum_absolute_error: maximum_error,
                    maximum_mean_absolute_error: mean_error,
                })
                .unwrap_or_else(|error| panic!("block-0 {name} parity failed: {error:#}"));
        }
        attention_metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 0.002,
                maximum_mean_absolute_error: 0.000_4,
            })
            .unwrap();
        println!(
            "Wan block-0 self-attention residual raw parity: cosine={:.9} max={:.9} mean={:.9}",
            residual_metrics.cosine_similarity,
            residual_metrics.maximum_absolute_error,
            residual_metrics.mean_absolute_error,
        );
        residual_metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 0.0025,
                maximum_mean_absolute_error: 0.000_05,
            })
            .unwrap();
        println!(
            "Wan block-0 remaining branches raw parity: cross=cos:{:.9}/max:{:.9}/mean:{:.9} cross_residual=cos:{:.9}/max:{:.9}/mean:{:.9} ffn=cos:{:.9}/max:{:.9}/mean:{:.9} block=cos:{:.9}/max:{:.9}/mean:{:.9}",
            cross_branch_metrics.cosine_similarity,
            cross_branch_metrics.maximum_absolute_error,
            cross_branch_metrics.mean_absolute_error,
            cross_residual_metrics.cosine_similarity,
            cross_residual_metrics.maximum_absolute_error,
            cross_residual_metrics.mean_absolute_error,
            ffn_branch_metrics.cosine_similarity,
            ffn_branch_metrics.maximum_absolute_error,
            ffn_branch_metrics.mean_absolute_error,
            block_output_metrics.cosine_similarity,
            block_output_metrics.maximum_absolute_error,
            block_output_metrics.mean_absolute_error,
        );
        for (name, metrics, maximum_error, mean_error) in [
            ("cross branch", &cross_branch_metrics, 0.003, 0.0005),
            ("cross residual", &cross_residual_metrics, 0.004, 0.0005),
            ("FFN branch", &ffn_branch_metrics, 0.01, 0.002),
            ("block output", &block_output_metrics, 0.01, 0.002),
        ] {
            metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 0.999_999,
                    maximum_absolute_error: maximum_error,
                    maximum_mean_absolute_error: mean_error,
                })
                .unwrap_or_else(|error| panic!("block-0 {name} parity failed: {error:#}"));
        }
        assert_eq!(query.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(key.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(value.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(attention.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(residual.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(cross_branch.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(cross_residual.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(ffn_branch.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(block_output.shape(), &[ROWS, dit.cfg.dim]);

        let d = dit.cfg.dim;
        let time_weight_bytes = ((d * dit.cfg.freq_dim + d * d + 6 * d * d)
            * std::mem::size_of::<u16>()
            + (d + d + 6 * d) * std::mem::size_of::<f32>()) as u64;
        let block_modulation_bytes = (6 * d * std::mem::size_of::<f32>()) as u64;
        let attention_linear_bytes =
            (4 * (d * d * std::mem::size_of::<u16>() + d * std::mem::size_of::<f32>())) as u64;
        let attention_norm_bytes = (2 * d * std::mem::size_of::<f32>()) as u64;
        let cross_norm_bytes = (4 * d * std::mem::size_of::<f32>()) as u64;
        let ffn_linear_bytes = (2 * d * dit.cfg.ffn_dim * std::mem::size_of::<u16>()
            + (dit.cfg.ffn_dim + d) * std::mem::size_of::<f32>())
            as u64;
        let expected_device_local_bytes = time_weight_bytes
            + block_modulation_bytes
            + 2 * attention_linear_bytes
            + attention_norm_bytes
            + cross_norm_bytes
            + ffn_linear_bytes;
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            20,
            "time, block modulation, both attention branches, norm3, and FFN are staged once"
        );
        assert_eq!(
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            expected_device_local_bytes
        );
        assert_eq!(
            after_execute.resident_weight_uploads, after_prepare.resident_weight_uploads,
            "resident QKV execution must not upload weights"
        );
        assert_eq!(
            after_execute.resident_tensor_uploads - before.resident_tensor_uploads,
            4,
            "token input, context, RoPE positions, and timestep are uploaded once"
        );
        assert_eq!(
            after_execute.resident_downloads - before.resident_downloads,
            9,
            "only named block-boundary tensors are downloaded for parity"
        );
        println!(
            "Wan block-0 self-attention: shape={:?} query_cosine={:.9} query_max={:.9} query_mean={:.9} key_cosine={:.9} key_max={:.9} key_mean={:.9} value_cosine={:.9} value_max={:.9} value_mean={:.9} output_cosine={:.9} output_max={:.9} output_mean={:.9} residual_cosine={:.9} residual_max={:.9} residual_mean={:.9} scalar_ms={:.3} prepare_ms={:.3} execute_ms={:.3} weight_uploads={} device_local_bytes={} device_local_allocation_bytes={} resident_bytes={} peak_resident_bytes={}",
            query.shape(),
            query_metrics.cosine_similarity,
            query_metrics.maximum_absolute_error,
            query_metrics.mean_absolute_error,
            key_metrics.cosine_similarity,
            key_metrics.maximum_absolute_error,
            key_metrics.mean_absolute_error,
            value_metrics.cosine_similarity,
            value_metrics.maximum_absolute_error,
            value_metrics.mean_absolute_error,
            attention_metrics.cosine_similarity,
            attention_metrics.maximum_absolute_error,
            attention_metrics.mean_absolute_error,
            residual_metrics.cosine_similarity,
            residual_metrics.maximum_absolute_error,
            residual_metrics.mean_absolute_error,
            scalar_runtime.as_secs_f64() * 1_000.0,
            prepare_runtime.as_secs_f64() * 1_000.0,
            execute_runtime.as_secs_f64() * 1_000.0,
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes,
            after_execute.resident_allocated_bytes - before.resident_allocated_bytes,
            after_execute.peak_resident_allocated_bytes,
        );
        println!(
            "Wan block-0 complete: cross_cosine={:.9} cross_max={:.9} cross_mean={:.9} cross_residual_cosine={:.9} cross_residual_max={:.9} cross_residual_mean={:.9} ffn_cosine={:.9} ffn_max={:.9} ffn_mean={:.9} block_cosine={:.9} block_max={:.9} block_mean={:.9}",
            cross_branch_metrics.cosine_similarity,
            cross_branch_metrics.maximum_absolute_error,
            cross_branch_metrics.mean_absolute_error,
            cross_residual_metrics.cosine_similarity,
            cross_residual_metrics.maximum_absolute_error,
            cross_residual_metrics.mean_absolute_error,
            ffn_branch_metrics.cosine_similarity,
            ffn_branch_metrics.maximum_absolute_error,
            ffn_branch_metrics.mean_absolute_error,
            block_output_metrics.cosine_similarity,
            block_output_metrics.maximum_absolute_error,
            block_output_metrics.mean_absolute_error,
        );
        crate::vulkan::print_statistics();

        drop(device_ffn);
        drop(device_cross);
        drop(device_residual);
        drop(device_attention);
        drop(qkv);
        drop(device_pre_attention);
        drop(device_modulation);
        drop(device_e0);
        drop(device_context);
        drop(device_positions);
        drop(device_input);
        drop(prepared_ffn);
        drop(prepared_cross);
        drop(prepared_attention);
        drop(prepared_block);
        drop(prepared_time);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads Wan weights and validates staged execution and eviction for blocks 0 and 1"]
    fn resident_vulkan_blocks0_and1_stage_evict_match_scalar() {
        use std::{path::Path, time::Instant};

        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const DIT: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan2.1_t2v_1.3B_Q4_K.gguf";
        const ROWS: usize = 7;
        const CONTEXT_ROWS: usize = 5;
        let gguf = GgufFile::open(Path::new(DIT)).unwrap();
        let dit = WanDit::load(&gguf, WanConfig::default()).unwrap();
        let input = Tensor::new(
            vec![ROWS, dit.cfg.dim],
            (0..ROWS * dit.cfg.dim)
                .map(|index| ((index * 17) as f32 * 0.007_812_5).sin() * 1.25)
                .collect(),
        )
        .unwrap();
        let context = Tensor::new(
            vec![CONTEXT_ROWS, dit.cfg.dim],
            (0..CONTEXT_ROWS * dit.cfg.dim)
                .map(|index| ((index * 23) as f32 * 0.003_906_25).cos() * 0.875)
                .collect(),
        )
        .unwrap();
        let ids = (0..ROWS)
            .map(|column| [0.0, 0.0, column as f32])
            .collect::<Vec<_>>();
        let positions = Tensor::new(
            vec![ROWS, dit.cfg.head_dim / 2, wan_rope::PAIR_STRIDE],
            wan_rope::embed_nd(&ids, &dit.cfg.axes_dim, dit.cfg.theta),
        )
        .unwrap();

        let scalar_started = Instant::now();
        let (_, scalar_e0) = dit.time_embedding_scalar(750.0);
        let mut scalar_state = input.clone();
        let mut scalar_outputs = Vec::with_capacity(2);
        for block_index in 0..2 {
            scalar_state = dit
                .wan_block_scalar(block_index, &scalar_state, &context, &scalar_e0, &positions)
                .unwrap();
            scalar_outputs.push(scalar_state.clone());
        }
        let scalar_runtime = scalar_started.elapsed();

        let before = crate::vulkan::persistence_stats().unwrap();
        let prepared_time = dit.prepare_time_embedding(&VULKAN_BACKEND).unwrap();
        let mut device_state = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let device_context = VULKAN_BACKEND.upload_tensor(&context).unwrap();
        let device_positions = VULKAN_BACKEND.upload_tensor(&positions).unwrap();
        let (device_e, device_e0) = dit
            .time_embedding_device(750.0, &VULKAN_BACKEND, &prepared_time)
            .unwrap();
        drop(device_e);
        let staged_baseline = crate::vulkan::persistence_stats().unwrap();

        let d = dit.cfg.dim;
        let block_modulation_bytes = (6 * d * std::mem::size_of::<f32>()) as u64;
        let attention_linear_bytes =
            (4 * (d * d * std::mem::size_of::<u16>() + d * std::mem::size_of::<f32>())) as u64;
        let self_attention_norm_bytes = (2 * d * std::mem::size_of::<f32>()) as u64;
        let cross_attention_norm_bytes = (4 * d * std::mem::size_of::<f32>()) as u64;
        let ffn_linear_bytes = (2 * d * dit.cfg.ffn_dim * std::mem::size_of::<u16>()
            + (dit.cfg.ffn_dim + d) * std::mem::size_of::<f32>())
            as u64;
        let expected_block_device_local_bytes = block_modulation_bytes
            + 2 * attention_linear_bytes
            + self_attention_norm_bytes
            + cross_attention_norm_bytes
            + ffn_linear_bytes;

        let mut prepare_ms = Vec::with_capacity(2);
        let mut execute_ms = Vec::with_capacity(2);
        let mut output_metrics = Vec::with_capacity(2);
        for block_index in 0..2 {
            let before_prepare = crate::vulkan::persistence_stats().unwrap();
            let prepare_started = Instant::now();
            let prepared = dit.prepare_wan_block(block_index, &VULKAN_BACKEND).unwrap();
            prepare_ms.push(prepare_started.elapsed().as_secs_f64() * 1_000.0);
            let after_prepare = crate::vulkan::persistence_stats().unwrap();
            assert_eq!(
                after_prepare.resident_weight_uploads - before_prepare.resident_weight_uploads,
                17,
                "each block must stage exactly its modulation, attention, norm, and FFN resources"
            );
            assert_eq!(
                after_prepare.resident_device_local_bytes
                    - before_prepare.resident_device_local_bytes,
                expected_block_device_local_bytes,
                "the prepared block must own one block of device-local resources"
            );

            let execute_started = Instant::now();
            let next_state = dit
                .wan_block_with_backend(
                    &device_state,
                    &device_context,
                    &device_e0,
                    &device_positions,
                    &VULKAN_BACKEND,
                    &prepared,
                )
                .unwrap();
            execute_ms.push(execute_started.elapsed().as_secs_f64() * 1_000.0);
            device_state = next_state;
            let output = VULKAN_BACKEND.download_tensor(&device_state).unwrap();
            let metrics = compare_tensors(&output, &scalar_outputs[block_index]).unwrap();
            metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 0.999_999,
                    maximum_absolute_error: 0.015,
                    maximum_mean_absolute_error: 0.001,
                })
                .unwrap_or_else(|error| {
                    panic!("staged Wan block {block_index} parity failed: {error:#}")
                });
            output_metrics.push(metrics);

            drop(prepared);
            let after_evict = crate::vulkan::persistence_stats().unwrap();
            assert_eq!(
                after_evict.resident_device_local_bytes,
                staged_baseline.resident_device_local_bytes,
                "block {block_index} weights must be evicted before staging the next block"
            );
            assert_eq!(
                after_evict.resident_device_local_allocation_bytes,
                staged_baseline.resident_device_local_allocation_bytes,
                "block {block_index} device allocations must be released after eviction"
            );
            assert_eq!(
                after_evict.resident_allocated_bytes, staged_baseline.resident_allocated_bytes,
                "only same-sized persistent activations may survive a block boundary"
            );
        }
        let after_blocks = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_blocks.resident_weight_uploads - staged_baseline.resident_weight_uploads,
            34
        );
        assert_eq!(
            after_blocks.resident_downloads - staged_baseline.resident_downloads,
            2,
            "one named parity output is downloaded per block"
        );
        for (block_index, metrics) in output_metrics.iter().enumerate() {
            println!(
                "Wan staged block {block_index}: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9} prepare_ms={:.3} execute_ms={:.3}",
                scalar_outputs[block_index].shape(),
                metrics.cosine_similarity,
                metrics.maximum_absolute_error,
                metrics.mean_absolute_error,
                prepare_ms[block_index],
                execute_ms[block_index],
            );
        }
        println!(
            "Wan staged blocks 0-1: scalar_ms={:.3} block_device_local_bytes={} resident_baseline_bytes={} peak_resident_bytes={} weight_uploads={} downloads={}",
            scalar_runtime.as_secs_f64() * 1_000.0,
            expected_block_device_local_bytes,
            staged_baseline.resident_allocated_bytes - before.resident_allocated_bytes,
            after_blocks.peak_resident_allocated_bytes,
            after_blocks.resident_weight_uploads - before.resident_weight_uploads,
            after_blocks.resident_downloads - before.resident_downloads,
        );
        crate::vulkan::print_statistics();

        drop(device_e0);
        drop(device_positions);
        drop(device_context);
        drop(device_state);
        drop(prepared_time);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads Wan weights and validates all 30 staged resident Vulkan blocks"]
    fn resident_vulkan_all_30_blocks_match_scalar() {
        use std::{path::Path, time::Instant};

        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const DIT: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan2.1_t2v_1.3B_Q4_K.gguf";
        const ROWS: usize = 3;
        const CONTEXT_ROWS: usize = 4;
        let gguf = GgufFile::open(Path::new(DIT)).unwrap();
        let dit = WanDit::load(&gguf, WanConfig::default()).unwrap();
        assert_eq!(dit.cfg.num_layers, 30);
        let input = Tensor::new(
            vec![ROWS, dit.cfg.dim],
            (0..ROWS * dit.cfg.dim)
                .map(|index| ((index * 17) as f32 * 0.007_812_5).sin() * 1.25)
                .collect(),
        )
        .unwrap();
        let context = Tensor::new(
            vec![CONTEXT_ROWS, dit.cfg.dim],
            (0..CONTEXT_ROWS * dit.cfg.dim)
                .map(|index| ((index * 23) as f32 * 0.003_906_25).cos() * 0.875)
                .collect(),
        )
        .unwrap();
        let ids = (0..ROWS)
            .map(|column| [0.0, 0.0, column as f32])
            .collect::<Vec<_>>();
        let positions = Tensor::new(
            vec![ROWS, dit.cfg.head_dim / 2, wan_rope::PAIR_STRIDE],
            wan_rope::embed_nd(&ids, &dit.cfg.axes_dim, dit.cfg.theta),
        )
        .unwrap();

        let scalar_started = Instant::now();
        let (_, scalar_e0) = dit.time_embedding_scalar(750.0);
        let mut scalar_output = input.clone();
        for block_index in 0..dit.cfg.num_layers {
            scalar_output = dit
                .wan_block_scalar(
                    block_index,
                    &scalar_output,
                    &context,
                    &scalar_e0,
                    &positions,
                )
                .unwrap();
        }
        let scalar_runtime = scalar_started.elapsed();

        let before = crate::vulkan::persistence_stats().unwrap();
        let prepared_time = dit.prepare_time_embedding(&VULKAN_BACKEND).unwrap();
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let device_context = VULKAN_BACKEND.upload_tensor(&context).unwrap();
        let device_positions = VULKAN_BACKEND.upload_tensor(&positions).unwrap();
        let (device_e, device_e0) = dit
            .time_embedding_device(750.0, &VULKAN_BACKEND, &prepared_time)
            .unwrap();
        drop(device_e);
        let staged_baseline = crate::vulkan::persistence_stats().unwrap();

        let execute_started = Instant::now();
        let device_output = dit
            .wan_blocks_with_backend(
                &device_input,
                &device_context,
                &device_e0,
                &device_positions,
                &VULKAN_BACKEND,
            )
            .unwrap();
        let execute_runtime = execute_started.elapsed();
        let output = VULKAN_BACKEND.download_tensor(&device_output).unwrap();
        let after_execute = crate::vulkan::persistence_stats().unwrap();
        let metrics = compare_tensors(&output, &scalar_output).unwrap();
        println!(
            "Wan all-30 staged block raw parity: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
        );
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 0.02,
                maximum_mean_absolute_error: 0.001,
            })
            .unwrap();
        assert_eq!(output.shape(), &[ROWS, dit.cfg.dim]);
        assert_eq!(
            after_execute.resident_weight_uploads - before.resident_weight_uploads,
            3 + 17 * dit.cfg.num_layers as u64,
            "three time projections and 17 resources per transformer block must be staged"
        );
        assert_eq!(
            after_execute.resident_tensor_uploads - before.resident_tensor_uploads,
            4,
            "input, context, positions, and timestep are the only host activation uploads"
        );
        assert_eq!(
            after_execute.resident_downloads - before.resident_downloads,
            1,
            "only the final 30-block output is downloaded"
        );
        assert_eq!(
            after_execute.resident_device_local_bytes, staged_baseline.resident_device_local_bytes,
            "all transformer-block weights must be evicted when the loop returns"
        );
        assert_eq!(
            after_execute.resident_device_local_allocation_bytes,
            staged_baseline.resident_device_local_allocation_bytes,
            "all transformer-block allocations must be released when the loop returns"
        );
        assert_eq!(
            after_execute.resident_allocated_bytes,
            staged_baseline.resident_allocated_bytes + (ROWS * dit.cfg.dim * 4) as u64,
            "the returned output is the only additional activation at the loop boundary"
        );
        println!(
            "Wan all-30 staged blocks: scalar_ms={:.3} stage_and_execute_ms={:.3} resident_bytes={} peak_resident_bytes={} device_local_bytes={} peak_device_local_bytes={} weight_uploads={} downloads={}",
            scalar_runtime.as_secs_f64() * 1_000.0,
            execute_runtime.as_secs_f64() * 1_000.0,
            after_execute.resident_allocated_bytes - before.resident_allocated_bytes,
            after_execute.peak_resident_allocated_bytes,
            after_execute.resident_device_local_bytes - before.resident_device_local_bytes,
            after_execute.peak_resident_device_local_bytes,
            after_execute.resident_weight_uploads - before.resident_weight_uploads,
            after_execute.resident_downloads - before.resident_downloads,
        );
        crate::vulkan::print_statistics();

        drop(device_output);
        drop(device_e0);
        drop(device_positions);
        drop(device_context);
        drop(device_input);
        drop(prepared_time);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads Wan weights and validates the complete resident Vulkan DiT envelope"]
    fn resident_vulkan_complete_dit_matches_scalar() {
        use std::{path::Path, time::Instant};

        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const DIT: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan2.1_t2v_1.3B_Q4_K.gguf";
        const TIME: usize = 1;
        const HEIGHT: usize = 2;
        const WIDTH: usize = 6;
        const CONTEXT_ROWS: usize = 4;
        let gguf = GgufFile::open(Path::new(DIT)).unwrap();
        let dit = WanDit::load(&gguf, WanConfig::default()).unwrap();
        let latent = Tensor::new(
            vec![dit.cfg.in_dim, TIME, HEIGHT, WIDTH],
            (0..dit.cfg.in_dim * TIME * HEIGHT * WIDTH)
                .map(|index| ((index * 19) as f32 * 0.011_718_75).sin() * 0.875)
                .collect(),
        )
        .unwrap();
        let context = Tensor::new(
            vec![CONTEXT_ROWS, dit.cfg.text_dim],
            (0..CONTEXT_ROWS * dit.cfg.text_dim)
                .map(|index| ((index * 29) as f32 * 0.001_953_125).cos() * 0.625)
                .collect(),
        )
        .unwrap();

        let scalar_started = Instant::now();
        let scalar_values = dit.forward(
            latent.data(),
            TIME,
            HEIGHT,
            WIDTH,
            750.0,
            context.data(),
            CONTEXT_ROWS,
        );
        let scalar_output =
            Tensor::new(vec![dit.cfg.out_dim, TIME, HEIGHT, WIDTH], scalar_values).unwrap();
        let scalar_runtime = scalar_started.elapsed();

        let before = crate::vulkan::persistence_stats().unwrap();
        let prepare_started = Instant::now();
        let prepared = dit.prepare_wan_envelope(&VULKAN_BACKEND).unwrap();
        let prepare_runtime = prepare_started.elapsed();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            8,
            "patch, text, time, head modulation, and head resources must be prepared once"
        );
        let device_latent = VULKAN_BACKEND.upload_tensor(&latent).unwrap();
        let device_context = VULKAN_BACKEND.upload_tensor(&context).unwrap();
        let execute_started = Instant::now();
        let device_output = dit
            .forward_with_backend(
                &device_latent,
                TIME,
                HEIGHT,
                WIDTH,
                750.0,
                &device_context,
                CONTEXT_ROWS,
                &VULKAN_BACKEND,
                &prepared,
            )
            .unwrap();
        let execute_runtime = execute_started.elapsed();
        let output = VULKAN_BACKEND.download_tensor(&device_output).unwrap();
        let after_execute = crate::vulkan::persistence_stats().unwrap();
        let metrics = compare_tensors(&output, &scalar_output).unwrap();
        println!(
            "Wan complete resident DiT raw parity: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
        );
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 0.002,
                maximum_mean_absolute_error: 0.000_5,
            })
            .unwrap();
        assert_eq!(output.shape(), &[dit.cfg.out_dim, TIME, HEIGHT, WIDTH]);
        assert_eq!(
            after_execute.resident_weight_uploads - before.resident_weight_uploads,
            8 + 17 * dit.cfg.num_layers as u64,
            "the envelope and each of 30 block stages must upload exactly once"
        );
        assert_eq!(
            after_execute.resident_tensor_uploads - before.resident_tensor_uploads,
            4,
            "latent, text context, timestep, and RoPE positions are the only activation uploads"
        );
        assert_eq!(
            after_execute.resident_downloads - before.resident_downloads,
            1,
            "only the final velocity tensor is downloaded"
        );
        assert_eq!(
            after_execute.resident_device_local_bytes, after_prepare.resident_device_local_bytes,
            "all block weights must be evicted while the prepared DiT envelope remains resident"
        );
        assert_eq!(
            after_execute.resident_device_local_allocation_bytes,
            after_prepare.resident_device_local_allocation_bytes,
            "all block allocations must be released after the complete forward pass"
        );
        println!(
            "Wan complete resident DiT: scalar_ms={:.3} prepare_ms={:.3} stage_and_execute_ms={:.3} envelope_device_local_bytes={} resident_bytes={} peak_resident_bytes={} peak_device_local_bytes={} weight_uploads={} activation_uploads={} downloads={}",
            scalar_runtime.as_secs_f64() * 1_000.0,
            prepare_runtime.as_secs_f64() * 1_000.0,
            execute_runtime.as_secs_f64() * 1_000.0,
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            after_execute.resident_allocated_bytes - before.resident_allocated_bytes,
            after_execute.peak_resident_allocated_bytes,
            after_execute.peak_resident_device_local_bytes,
            after_execute.resident_weight_uploads - before.resident_weight_uploads,
            after_execute.resident_tensor_uploads - before.resident_tensor_uploads,
            after_execute.resident_downloads - before.resident_downloads,
        );
        crate::vulkan::print_statistics();

        drop(device_output);
        drop(device_context);
        drop(device_latent);
        drop(prepared);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes
        );
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    use crate::gguf::GgufFile;
    use std::path::Path;

    const PACK: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack";
    const REF: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/reference/dit");
    /// Committed alongside the DiT fixtures so this parity test cannot pass by skipping.
    const CTX: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/reference/t5/cond_crossattn.bin"
    );

    fn read_dump(p: &Path) -> Option<(Vec<i64>, Vec<f32>)> {
        let b = std::fs::read(p).ok()?;
        if &b[..4] != b"SQD1" {
            return None;
        }
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
        // Committed fixtures must be present; only the 816 MB DiT weights may legitimately be
        // absent from a checkout, so only that is allowed to skip.
        for p in [ip.as_path(), op.as_path(), tp.as_path(), cp] {
            assert!(p.exists(), "missing committed fixture {p:?}");
        }
        if !gp.exists() {
            eprintln!("skipping: DiT weights {gp:?} not present");
            return;
        }

        let (in_dims, latent) = read_dump(&ip).unwrap();
        let (_, want) = read_dump(&op).unwrap();
        let (_, tvec) = read_dump(&tp).unwrap();
        let (ctx_dims, context) = read_dump(cp).unwrap();
        // ggml order: ne0 fastest. [w, h, t, c] on disk is [c][t][h][w] row-major.
        let (w, h, t, _c) = (
            in_dims[0] as usize,
            in_dims[1] as usize,
            in_dims[2] as usize,
            in_dims[3] as usize,
        );
        let n_ctx = ctx_dims[1] as usize;
        eprintln!(
            "latent [c,t,h,w]=[{_c},{t},{h},{w}] timestep={} n_ctx={n_ctx}",
            tvec[0]
        );

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
            for v in &got {
                b.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write("/tmp/saient_ref/dit_out_ours.bin", b);
            eprintln!("  wrote /tmp/saient_ref/dit_out_ours.bin");
        }

        // Where does it agree? A per-channel or per-frame split localises a layout fault; a
        // uniform smear says the maths is wrong rather than the indexing.
        let cosf = |a: &[f32], b: &[f32]| -> f64 {
            let (mut d, mut x, mut y) = (0.0f64, 0.0f64, 0.0f64);
            for (p, q) in a.iter().zip(b) {
                d += *p as f64 * *q as f64;
                x += *p as f64 * *p as f64;
                y += *q as f64 * *q as f64;
            }
            if x == 0.0 || y == 0.0 {
                return 0.0;
            }
            d / (x.sqrt() * y.sqrt())
        };
        let plane = h * w;
        for c in 0..4usize {
            let a = c * t * plane;
            eprintln!(
                "  channel {c}: cosine={:.6}",
                cosf(&got[a..a + t * plane], &want[a..a + t * plane])
            );
        }
        for ti in 0..t {
            let a = ti * plane;
            eprintln!(
                "  frame {ti} (ch0): cosine={:.6}",
                cosf(&got[a..a + plane], &want[a..a + plane])
            );
        }
        // First latent row of channel 0 — if only the very start matches, ordering is suspect.
        eprintln!(
            "  ch0 row0 (w=52): cosine={:.6}",
            cosf(&got[..w], &want[..w])
        );
        eprintln!("  ch0 first 4 patches: got={:?}", &got[..4]);
        eprintln!("  got[..4]  = {:?}", &got[..4]);
        eprintln!("  want[..4] = {:?}", &want[..4]);
        assert!(
            cos > 0.99,
            "DiT velocity does not match reference (cosine {cos:.6})"
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads the 816MB DiT and validates the captured 780-token Vulkan velocity"]
    fn resident_vulkan_matches_reference_velocity() {
        use std::time::Instant;

        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let gp = Path::new(PACK).join("wan2.1_t2v_1.3B_Q4_K.gguf");
        let (ip, op, tp, cp) = (
            Path::new(REF).join("dit_in_0.bin"),
            Path::new(REF).join("dit_out_0.bin"),
            Path::new(REF).join("dit_t_0.bin"),
            Path::new(CTX),
        );
        for path in [&gp, &ip, &op, &tp, cp] {
            assert!(
                path.exists(),
                "required Vulkan DiT fixture {path:?} is missing"
            );
        }
        let (input_dims, latent_values) = read_dump(&ip).unwrap();
        let (output_dims, reference_values) = read_dump(&op).unwrap();
        let (_, timestep) = read_dump(&tp).unwrap();
        let (context_dims, context_values) = read_dump(cp).unwrap();
        assert_eq!(input_dims, output_dims);
        let (width, height, time, channels) = (
            input_dims[0] as usize,
            input_dims[1] as usize,
            input_dims[2] as usize,
            input_dims[3] as usize,
        );
        let context_rows = context_dims[1] as usize;
        let latent = Tensor::new(vec![channels, time, height, width], latent_values).unwrap();
        let context = Tensor::new(vec![context_rows, 4096], context_values).unwrap();
        let reference = Tensor::new(vec![channels, time, height, width], reference_values).unwrap();

        let gguf = GgufFile::open(&gp).unwrap();
        let dit = WanDit::load(&gguf, WanConfig::default()).unwrap();
        assert_eq!(latent.shape(), &[dit.cfg.in_dim, 2, 30, 52]);
        assert_eq!(context.shape(), &[context_rows, dit.cfg.text_dim]);
        let before = crate::vulkan::persistence_stats().unwrap();
        let prepare_started = Instant::now();
        let prepared = dit.prepare_wan_envelope(&VULKAN_BACKEND).unwrap();
        let prepare_runtime = prepare_started.elapsed();
        let device_latent = VULKAN_BACKEND.upload_tensor(&latent).unwrap();
        let device_context = VULKAN_BACKEND.upload_tensor(&context).unwrap();
        let execute_started = Instant::now();
        let device_output = dit
            .forward_with_backend(
                &device_latent,
                time,
                height,
                width,
                timestep[0],
                &device_context,
                context_rows,
                &VULKAN_BACKEND,
                &prepared,
            )
            .unwrap();
        let execute_runtime = execute_started.elapsed();
        let output = VULKAN_BACKEND.download_tensor(&device_output).unwrap();
        let after_execute = crate::vulkan::persistence_stats().unwrap();
        let metrics = compare_tensors(&output, &reference).unwrap();
        println!(
            "Wan captured 780-token Vulkan DiT raw parity: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
        );
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_9,
                maximum_absolute_error: 0.06,
                maximum_mean_absolute_error: 0.01,
            })
            .unwrap();
        assert_eq!(output.shape(), &[16, 2, 30, 52]);
        assert_eq!(
            after_execute.resident_weight_uploads - before.resident_weight_uploads,
            8 + 17 * dit.cfg.num_layers as u64
        );
        assert_eq!(
            after_execute.resident_tensor_uploads - before.resident_tensor_uploads,
            4
        );
        assert_eq!(
            after_execute.resident_downloads - before.resident_downloads,
            1
        );
        println!(
            "Wan captured 780-token Vulkan DiT: prepare_ms={:.3} stage_and_execute_ms={:.3} resident_bytes={} peak_resident_bytes={} device_local_bytes={} peak_device_local_bytes={} uploaded_bytes={} downloaded_bytes={}",
            prepare_runtime.as_secs_f64() * 1_000.0,
            execute_runtime.as_secs_f64() * 1_000.0,
            after_execute.resident_allocated_bytes - before.resident_allocated_bytes,
            after_execute.peak_resident_allocated_bytes,
            after_execute.resident_device_local_bytes - before.resident_device_local_bytes,
            after_execute.peak_resident_device_local_bytes,
            after_execute.resident_uploaded_bytes - before.resident_uploaded_bytes,
            after_execute.resident_downloaded_bytes - before.resident_downloaded_bytes,
        );
        crate::vulkan::print_statistics();

        drop(device_output);
        drop(device_context);
        drop(device_latent);
        drop(prepared);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes
        );
    }
}
