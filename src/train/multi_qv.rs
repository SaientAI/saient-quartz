//! Multi-layer Q/V LoRA forward through actual Quartz transformer blocks.
//! The frozen prefix is detached; all tokens traverse the trainable suffix.

use super::ops::{AttentionResult, attention_last};
use super::qv::{AdapterPair, AdapterPairGrads};
use crate::dequant;
use crate::gguf::{GgufFile, TensorInfo};
use crate::{
    Config, GpuState, Weights, apply_qk_norm, embed_and_layers_up_to, rms_norm, rope_yarn,
};
use anyhow::{Result, bail};
use std::sync::Arc;

#[derive(Clone)]
pub struct MultiPair {
    pub first_layer: usize,
    pub blocks: Vec<AdapterPair>,
}
impl MultiPair {
    pub fn new(cfg: &Config, layers: usize, rank: usize, seed: u64) -> Result<Self> {
        if !(1..=8).contains(&layers) || layers > cfg.n_layers {
            bail!("expected 1..8 suffix blocks");
        }
        let first_layer = cfg.n_layers - layers;
        let mut blocks = Vec::new();
        for layer in first_layer..cfg.n_layers {
            blocks.push(AdapterPair::new(
                cfg,
                layer,
                rank,
                seed + 17 * layer as u64,
            )?);
        }
        Ok(Self {
            first_layer,
            blocks,
        })
    }
    pub fn grads(&self) -> Vec<AdapterPairGrads> {
        self.blocks.iter().map(AdapterPairGrads::zeros).collect()
    }
    pub fn parameter_count(&self) -> usize {
        self.blocks
            .iter()
            .map(|p| p.q.a.len() + p.q.b.len() + p.v.a.len() + p.v.b.len())
            .sum()
    }
}
pub struct Prepared {
    pub prefix: Vec<Vec<f32>>,
    pub target: usize,
}
#[derive(Clone)]
pub struct TokenTape {
    pub input: Vec<f32>,
    pub n1: Vec<f32>,
    pub q_pre: Vec<f32>,
    pub k_pre: Vec<f32>,
    pub q_rot: Vec<f32>,
    pub k_rot: Vec<f32>,
    pub value: Vec<f32>,
    pub attention: AttentionResult,
    pub z: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub output: Vec<f32>,
}
pub struct LayerTape {
    pub layer: usize,
    pub tokens: Vec<TokenTape>,
}
pub struct Tape {
    pub layers: Vec<LayerTape>,
    pub logits: Vec<f32>,
}

fn mv(gguf: &GgufFile, ti: &TensorInfo, x: &[f32], output: usize) -> Vec<f32> {
    assert_eq!(ti.dims[0] as usize, x.len());
    assert_eq!(ti.dims[1] as usize, output);
    dequant::gemv(x, gguf.tensor_data(ti), ti.ggml_type, x.len(), output)
}
pub(super) fn rope_heads(x: &mut [f32], pos: usize, heads: usize, cfg: &Config) {
    for h in 0..heads {
        let range = h * cfg.head_dim..(h + 1) * cfg.head_dim;
        rope_yarn(
            &mut x[range],
            pos,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.yarn_scale,
            cfg.yarn_orig_ctx,
            cfg.rope_neox,
        );
    }
}
pub(super) fn scale(cfg: &Config) -> f32 {
    let mscale2 = if cfg.yarn_scale > 1.0 && cfg.yarn_orig_ctx > 0 {
        let m = 1.0 + 0.1 * cfg.yarn_scale.ln();
        m * m
    } else {
        1.0
    };
    (cfg.head_dim as f32).sqrt().recip() * mscale2
}
pub fn prepare(
    ids: &[u32],
    target: usize,
    first_layer: usize,
    _gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> Result<Prepared> {
    if ids.is_empty() || ids.len() > 16 || first_layer >= cfg.n_layers || target >= cfg.vocab_size {
        bail!("unsupported frozen-prefix input");
    }
    let mut kv: Vec<(Vec<f32>, Vec<f32>)> = (0..cfg.n_layers)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    let mut prefix = Vec::with_capacity(ids.len());
    for (pos, &id) in ids.iter().enumerate() {
        prefix.push(embed_and_layers_up_to(
            id as usize,
            pos,
            cfg,
            w,
            &mut kv,
            gpu,
            first_layer,
        ));
    }
    Ok(Prepared { prefix, target })
}
pub fn forward(
    sample: &Prepared,
    model: &MultiPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> Tape {
    assert_eq!(model.first_layer + model.blocks.len(), cfg.n_layers);
    let n = sample.prefix.len();
    assert!(n > 0);
    let d = cfg.embed_dim;
    let qd = cfg.q_total_dim();
    let kd = cfg.kv_total_dim();
    let mut input = sample.prefix.clone();
    let mut layers = Vec::with_capacity(model.blocks.len());
    for (bi, pair) in model.blocks.iter().enumerate() {
        let l = model.first_layer + bi;
        assert_eq!(pair.layer, l);
        let mut norms = Vec::with_capacity(n);
        let mut qp = Vec::with_capacity(n);
        let mut kp = Vec::with_capacity(n);
        let mut qr = Vec::with_capacity(n);
        let mut kr = Vec::with_capacity(n);
        let mut vv = Vec::with_capacity(n);
        for t in 0..n {
            let norm = rms_norm(&input[t], &w.attn_norm[l], cfg.norm_eps);
            let mut q = mv(gguf, &w.q_proj[l], &norm, qd);
            let mut k = mv(gguf, &w.k_proj[l], &norm, kd);
            let mut v = mv(gguf, &w.v_proj[l], &norm, kd);
            for i in 0..qd {
                q[i] += w.q_bias[l][i];
            }
            for i in 0..kd {
                k[i] += w.k_bias[l][i];
                v[i] += w.v_bias[l][i];
            }
            pair.q.forward_add(&norm, &mut q);
            pair.v.forward_add(&norm, &mut v);
            let mut qrot = q.clone();
            let mut krot = k.clone();
            apply_qk_norm(
                &mut qrot,
                &w.q_norm[l],
                cfg.n_heads,
                cfg.head_dim,
                cfg.norm_eps,
            );
            apply_qk_norm(
                &mut krot,
                &w.k_norm[l],
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.norm_eps,
            );
            rope_heads(&mut qrot, t, cfg.n_heads, cfg);
            rope_heads(&mut krot, t, cfg.n_kv_heads, cfg);
            norms.push(norm);
            qp.push(q);
            kp.push(k);
            qr.push(qrot);
            kr.push(krot);
            vv.push(v);
        }
        let mut next = Vec::with_capacity(n);
        let mut tokens = Vec::with_capacity(n);
        for t in 0..n {
            let att = attention_last(
                &qr[t],
                &kr[..=t],
                &vv[..=t],
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.attn_start(l, t),
                scale(cfg),
                &w.attn_sinks[l],
            );
            let mut z = mv(gguf, &w.out_proj[l], &att.out, d);
            for i in 0..d {
                z[i] += input[t][i];
                if !w.out_bias[l].is_empty() {
                    z[i] += w.out_bias[l][i];
                }
            }
            let n2 = rms_norm(&z, &w.ffn_norm[l], cfg.norm_eps);
            let gate = mv(gguf, &w.gate[l], &n2, cfg.ffn_dim);
            let up = mv(gguf, &w.up[l], &n2, cfg.ffn_dim);
            let hidden: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let down = mv(gguf, &w.down[l], &hidden, d);
            let output: Vec<f32> = z.iter().zip(down).map(|(&a, b)| a + b).collect();
            next.push(output.clone());
            tokens.push(TokenTape {
                input: input[t].clone(),
                n1: norms[t].clone(),
                q_pre: qp[t].clone(),
                k_pre: kp[t].clone(),
                q_rot: qr[t].clone(),
                k_rot: kr[t].clone(),
                value: vv[t].clone(),
                attention: att,
                z,
                gate,
                up,
                output,
            });
        }
        layers.push(LayerTape { layer: l, tokens });
        input = next;
    }
    let norm = rms_norm(&input[n - 1], &w.final_norm, cfg.norm_eps);
    let logits = gpu.gemv_lm_head(&norm, &w.lm_head, gguf, d, cfg.vocab_size);
    Tape { layers, logits }
}
pub fn loss(
    sample: &Prepared,
    model: &MultiPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> f32 {
    let tape = forward(sample, model, gguf, cfg, w, gpu);
    super::qv::token_ce(&tape.logits, sample.target).0
}
/// All-zero B matrices must reproduce the ordinary model, including
/// the normal causal multi-layer path.
pub fn parity(
    ids: &[u32],
    sample: &Prepared,
    model: &MultiPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> Result<f32> {
    let tape = forward(sample, model, gguf, cfg, w, gpu);
    let mut kv: Vec<(Vec<f32>, Vec<f32>)> = (0..cfg.n_layers)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    let mut normal = Vec::new();
    for (pos, &token) in ids.iter().enumerate() {
        if pos + 1 < ids.len() {
            crate::embed_and_layers(token as usize, pos, cfg, w, &mut kv, gpu);
        } else {
            normal = crate::forward(token as usize, pos, cfg, w, &mut kv, gpu);
        }
    }
    if normal.len() != tape.logits.len() {
        bail!("multi-layer parity size mismatch");
    }
    let worst = normal
        .iter()
        .zip(&tape.logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let max = normal.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
    let rel = worst / max;
    if !rel.is_finite() || rel > 5e-4 {
        bail!("frozen multi-layer mismatch: rel={rel:.8}");
    }
    Ok(rel)
}
