//! Correctness-first Q/V LoRA training on the final dense transformer layer.
//!
//! The prefix layers 0..last-1 are frozen and evaluated once per sample.
//! Only the last query and all causal keys/values affect a last-token loss.
//! We backprop through its complete downstream path (last attention, output
//! projection, SwiGLU, RMSNorm, lm_head) while retaining no gradients for
//! frozen prefix or base weights. This is not yet arbitrary last-K SFT.

use anyhow::{Result, bail};
use std::sync::Arc;

use super::lora::{LoraGrads, LoraLinear};
use super::ops::{
    AttentionResult, attention_last, attention_last_bwd, gguf_transpose, qk_norm_bwd, rms_bwd,
    rope_bwd, silu_derivative,
};
use crate::dequant;
use crate::gguf::{GgufFile, TensorInfo};
use crate::{
    Config, GpuState, Weights, apply_qk_norm, embed_and_layers, embed_and_layers_up_to, forward,
    rms_norm, rope_yarn,
};

#[derive(Clone)]
pub struct AdapterPair {
    pub layer: usize,
    pub q: LoraLinear,
    pub v: LoraLinear,
}
impl AdapterPair {
    pub fn new(cfg: &Config, layer: usize, rank: usize, seed: u64) -> Result<Self> {
        if rank == 0 || rank > 16 || layer >= cfg.n_layers {
            return Err(anyhow::anyhow!("invalid adapter parameters"));
        }
        let q = LoraLinear::new(
            cfg.embed_dim,
            cfg.q_total_dim(),
            rank,
            rank as f32 * 2.0,
            seed,
        );
        let v = LoraLinear::new(
            cfg.embed_dim,
            cfg.kv_total_dim(),
            rank,
            rank as f32 * 2.0,
            seed + 1,
        );
        Ok(Self { layer, q, v })
    }
}
pub struct AdapterPairGrads {
    pub q: LoraGrads,
    pub v: LoraGrads,
}
impl AdapterPairGrads {
    pub fn zeros(p: &AdapterPair) -> Self {
        Self {
            q: p.q.zeros_like_grads(),
            v: p.v.zeros_like_grads(),
        }
    }
    pub fn components(&self) -> [&[f32]; 4] {
        [&self.q.a, &self.q.b, &self.v.a, &self.v.b]
    }
}

/// A sample's only cached data is the frozen layer input and its frozen
/// q/k/v baselines. No retained prefix activations or model gradients.
pub struct Prepared {
    pub inputs: Vec<Vec<f32>>,   // layer-normed inputs to Q/K/V, one per token
    pub k_rot: Vec<Vec<f32>>,    // frozen K + qk-norm + RoPE
    pub v_base: Vec<Vec<f32>>,   // frozen V+bias
    pub q_base_last: Vec<f32>,   // frozen Q+bias before qk-norm/RoPE
    pub residual_last: Vec<f32>, // unnormalised prefix output at final token
    pub target: usize,
}
pub struct Tape {
    pub q_pre: Vec<f32>,
    pub q_rot: Vec<f32>,
    pub values: Vec<Vec<f32>>,
    pub attn: AttentionResult,
    pub z: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub final_pre: Vec<f32>,
    pub logits: Vec<f32>,
}

fn gemv(gguf: &GgufFile, ti: &TensorInfo, x: &[f32], out: usize) -> Vec<f32> {
    dequant::gemv(x, gguf.tensor_data(ti), ti.ggml_type, x.len(), out)
}

pub fn prepare(
    tokens: &[u32],
    target: usize,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> Result<Prepared> {
    if cfg.n_layers < 1
        || cfg.n_experts != 0
        || tokens.is_empty()
        || tokens.len() > 32
        || target >= cfg.vocab_size
    {
        bail!("unsupported Q/V preparation");
    }
    let last = cfg.n_layers - 1;
    let mut kv: Vec<(Vec<f32>, Vec<f32>)> = (0..cfg.n_layers)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    let mut inputs = Vec::with_capacity(tokens.len());
    let mut k_rot = Vec::with_capacity(tokens.len());
    let mut v_base = Vec::with_capacity(tokens.len());
    let mut q_base_last = Vec::new();
    let mut residual_last = Vec::new();
    for (pos, &token) in tokens.iter().enumerate() {
        let x = embed_and_layers_up_to(token as usize, pos, cfg, w, &mut kv, gpu, last);
        let n1 = rms_norm(&x, &w.attn_norm[last], cfg.norm_eps);
        let mut k = gemv(gguf, &w.k_proj[last], &n1, cfg.kv_total_dim());
        let mut v = gemv(gguf, &w.v_proj[last], &n1, cfg.kv_total_dim());
        for i in 0..k.len() {
            k[i] += w.k_bias[last][i];
            v[i] += w.v_bias[last][i];
        }
        apply_qk_norm(
            &mut k,
            &w.k_norm[last],
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.norm_eps,
        );
        for h in 0..cfg.n_kv_heads {
            let r = h * cfg.head_dim..(h + 1) * cfg.head_dim;
            rope_yarn(
                &mut k[r],
                pos,
                cfg.head_dim,
                cfg.rope_theta,
                cfg.yarn_scale,
                cfg.yarn_orig_ctx,
                cfg.rope_neox,
            );
        }
        if pos == tokens.len() - 1 {
            residual_last = x;
            q_base_last = gemv(gguf, &w.q_proj[last], &n1, cfg.q_total_dim());
            for i in 0..q_base_last.len() {
                q_base_last[i] += w.q_bias[last][i];
            }
        }
        inputs.push(n1);
        k_rot.push(k);
        v_base.push(v);
    }
    Ok(Prepared {
        inputs,
        k_rot,
        v_base,
        q_base_last,
        residual_last,
        target,
    })
}

pub fn forward_last(
    sample: &Prepared,
    pair: &AdapterPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> Tape {
    let l = pair.layer;
    assert_eq!(l + 1, cfg.n_layers);
    let n = sample.inputs.len();
    let d = cfg.embed_dim;
    let mut qpre = sample.q_base_last.clone();
    pair.q.forward_add(&sample.inputs[n - 1], &mut qpre);
    let mut qrot = qpre.clone();
    apply_qk_norm(
        &mut qrot,
        &w.q_norm[l],
        cfg.n_heads,
        cfg.head_dim,
        cfg.norm_eps,
    );
    for h in 0..cfg.n_heads {
        let r = h * cfg.head_dim..(h + 1) * cfg.head_dim;
        rope_yarn(
            &mut qrot[r],
            n - 1,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.yarn_scale,
            cfg.yarn_orig_ctx,
            cfg.rope_neox,
        );
    }
    let mut values = sample.v_base.clone();
    for (i, v) in values.iter_mut().enumerate() {
        pair.v.forward_add(&sample.inputs[i], v);
    }
    let mscale = if cfg.yarn_scale > 1.0 && cfg.yarn_orig_ctx > 0 {
        let m = 1.0 + 0.1 * cfg.yarn_scale.ln();
        m * m
    } else {
        1.0
    };
    let scale = (cfg.head_dim as f32).sqrt().recip() * mscale;
    let attn = attention_last(
        &qrot,
        &sample.k_rot,
        &values,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.attn_start(l, n - 1),
        scale,
        &w.attn_sinks[l],
    );
    let mut z = gemv(gguf, &w.out_proj[l], &attn.out, d);
    for i in 0..d {
        z[i] += sample.residual_last[i];
        if !w.out_bias[l].is_empty() {
            z[i] += w.out_bias[l][i];
        }
    }
    let n2 = rms_norm(&z, &w.ffn_norm[l], cfg.norm_eps);
    let gate = gemv(gguf, &w.gate[l], &n2, cfg.ffn_dim);
    let up = gemv(gguf, &w.up[l], &n2, cfg.ffn_dim);
    let hidden: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let down = gemv(gguf, &w.down[l], &hidden, d);
    let final_pre: Vec<f32> = z.iter().zip(down).map(|(&x, y)| x + y).collect();
    let final_hidden = rms_norm(&final_pre, &w.final_norm, cfg.norm_eps);
    let logits = gpu.gemv_lm_head(&final_hidden, &w.lm_head, gguf, d, cfg.vocab_size);
    Tape {
        q_pre: qpre,
        q_rot: qrot,
        values,
        attn,
        z,
        gate,
        up,
        final_pre,
        logits,
    }
}

/// Exact full-vocabulary CE and logits gradient; no approximation to softmax.
pub fn token_ce(logits: &[f32], target: usize) -> (f32, Vec<f32>) {
    assert!(target < logits.len());
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z = logits
        .iter()
        .map(|&v| ((v as f64) - max).exp())
        .sum::<f64>();
    let logz = max + z.ln();
    let loss = (logz - logits[target] as f64) as f32;
    let dlogits: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| ((v as f64 - logz).exp()) as f32 - if i == target { 1.0 } else { 0.0 })
        .collect();
    (loss, dlogits)
}
pub fn loss(
    sample: &Prepared,
    pair: &AdapterPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> f32 {
    let tape = forward_last(sample, pair, gguf, cfg, w, gpu);
    token_ce(&tape.logits, sample.target).0
}

/// This is the full derivative from last-token CE to Q/V adapter parameters.
/// No gradient is needed into the prefix, which is frozen and detached.
pub fn backward(
    sample: &Prepared,
    pair: &AdapterPair,
    t: &Tape,
    dlogits: &[f32],
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    grads: &mut AdapterPairGrads,
) {
    let l = pair.layer;
    let d = cfg.embed_dim;
    // Output head (frozen) + final RMSNorm.
    let g_final_norm = gguf_transpose(gguf, &w.lm_head, dlogits, d, cfg.vocab_size);
    let g_end = rms_bwd(&t.final_pre, &w.final_norm, &g_final_norm, cfg.norm_eps);

    // z -> FFN RMSNorm -> SwiGLU -> down + residual. W^T*dY is REQUIRED
    // even though all base matrices are frozen.
    let g_hidden = gguf_transpose(gguf, &w.down[l], &g_end, cfg.ffn_dim, d);
    let mut g_gate = vec![0.0; cfg.ffn_dim];
    let mut g_up = vec![0.0; cfg.ffn_dim];
    for j in 0..cfg.ffn_dim {
        let sig = 1.0 / (1.0 + (-t.gate[j]).exp());
        let h = t.gate[j] * sig;
        g_gate[j] = g_hidden[j] * t.up[j] * silu_derivative(t.gate[j]);
        g_up[j] = g_hidden[j] * h;
    }
    let n2_gate = gguf_transpose(gguf, &w.gate[l], &g_gate, d, cfg.ffn_dim);
    let n2_up = gguf_transpose(gguf, &w.up[l], &g_up, d, cfg.ffn_dim);
    let mut g_n2 = n2_gate;
    for i in 0..d {
        g_n2[i] += n2_up[i];
    }
    let mut g_z = g_end;
    let g_rms = rms_bwd(&t.z, &w.ffn_norm[l], &g_n2, cfg.norm_eps);
    for i in 0..d {
        g_z[i] += g_rms[i];
    }
    let g_attention = gguf_transpose(gguf, &w.out_proj[l], &g_z, cfg.q_total_dim(), d);

    // Attention softmax + GQA: all causal V gradients and final Q gradient.
    let mscale = if cfg.yarn_scale > 1.0 && cfg.yarn_orig_ctx > 0 {
        let m = 1.0 + 0.1 * cfg.yarn_scale.ln();
        m * m
    } else {
        1.0
    };
    let scale = (cfg.head_dim as f32).sqrt().recip() * mscale;
    let ag = attention_last_bwd(
        &t.q_rot,
        &sample.k_rot,
        &t.values,
        &t.attn,
        &g_attention,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.attn_start(l, sample.inputs.len() - 1),
        scale,
    );

    // Q rotation is orthogonal; qk RMSNorm is often absent in Qwen2.5.
    let mut gq_norm = ag.dq;
    let last_pos = sample.inputs.len() - 1;
    for h in 0..cfg.n_heads {
        let r = h * cfg.head_dim..(h + 1) * cfg.head_dim;
        rope_bwd(
            &mut gq_norm[r],
            last_pos,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.yarn_scale,
            cfg.yarn_orig_ctx,
            cfg.rope_neox,
        );
    }
    let gq = qk_norm_bwd(
        &t.q_pre,
        &w.q_norm[l],
        &gq_norm,
        cfg.n_heads,
        cfg.head_dim,
        cfg.norm_eps,
    );
    let mut unused_dx = vec![0.0; d];
    pair.q
        .backward_add(&sample.inputs[last_pos], &gq, &mut unused_dx, &mut grads.q);
    for i in 0..sample.inputs.len() {
        unused_dx.fill(0.0);
        pair.v
            .backward_add(&sample.inputs[i], &ag.dv[i], &mut unused_dx, &mut grads.v);
    }
}

/// Compare exact next-token scores against original inference for one
/// sample at zero-LoRA B, guarding against a subtle forward mismatch.
pub fn frozen_forward_parity(
    ids: &[u32],
    sample: &Prepared,
    pair: &AdapterPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> Result<f32> {
    let t = forward_last(sample, pair, gguf, cfg, w, gpu);
    let mut kv: Vec<(Vec<f32>, Vec<f32>)> = (0..cfg.n_layers)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    for (pos, &tok) in ids.iter().take(ids.len() - 1).enumerate() {
        embed_and_layers(tok as usize, pos, cfg, w, &mut kv, gpu);
    }
    let original = forward(
        *ids.last().unwrap() as usize,
        ids.len() - 1,
        cfg,
        w,
        &mut kv,
        gpu,
    );
    if original.len() != t.logits.len() {
        bail!("head parity logits length mismatch");
    }
    let worst = original
        .iter()
        .zip(&t.logits)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let denom = original.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
    let relative = worst / denom;
    if !relative.is_finite() || relative > 5e-4 {
        bail!("frozen inference/last-block parity failed rel={relative:.7}");
    }
    Ok(relative)
}
