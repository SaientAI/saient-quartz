// tinyq4 — minimal GGUF transformer runtime

mod gguf;
mod dequant;
mod tokenizer;
mod server;
#[cfg(feature = "cuda")]
mod cuda;

use rayon::prelude::*;
use std::sync::Arc;

use gguf::{GgufFile, TensorInfo, ggml_type_size};
use tokenizer::Tokenizer;
use std::path::Path;

// ── Config ────────────────────────────────────────────────────────────────────

pub(crate) struct Config {
    vocab_size:      usize,
    embed_dim:       usize,
    n_heads:         usize,
    n_kv_heads:      usize,
    n_layers:        usize,
    ffn_dim:         usize,
    rope_theta:      f32,
    head_dim:        usize,   // actual Q/K head dim (from attention.key_length)
    n_experts:       usize,   // 0 = dense FFN
    n_experts_used:  usize,   // top-k for MoE routing
    yarn_scale:      f32,     // RoPE scaling factor (1.0 = no scaling)
    yarn_orig_ctx:   usize,   // original context length for YARN
    clamped_swiglu:  bool,    // true = gpt-oss clamped OAI SwiGLU experts; false = standard SiLU
}

impl Config {
    fn from_gguf(g: &GgufFile) -> Self {
        let embed_dim  = g.arch_u32("embedding_length")
            .expect("missing embedding_length") as usize;
        let n_heads = g.arch_u32_any(&["head_count", "attention.head_count"])
            .expect("missing head_count") as usize;
        let n_kv_heads = g.arch_u32_any(&["head_count_kv", "attention.head_count_kv"])
            .unwrap_or(n_heads as u32) as usize;
        let n_layers   = g.arch_u32("block_count")
            .expect("missing block_count") as usize;
        let ffn_dim    = g.arch_u32_any(&[
            "expert_feed_forward_length", "feed_forward_length",
            "attention.feed_forward_length"])
            .expect("missing feed_forward_length") as usize;
        let rope_theta = g.arch_f32_any(&["rope.freq_base", "attention.rope.freq_base"])
            .unwrap_or(10_000.0);

        let head_dim = g.arch_u32_any(&["attention.key_length"])
            .map(|v| v as usize)
            .unwrap_or(embed_dim / n_heads);

        let vocab_size = g.arch_u32("vocab_size")
            .or_else(|| {
                let arch = g.architecture().unwrap_or("llama");
                g.metadata.get(&format!("{}.vocab_size", arch))
                    .and_then(|v| v.as_u32())
            })
            .or_else(|| {
                if let Some(gguf::GgufValue::Array(arr)) =
                    g.metadata.get("tokenizer.ggml.tokens")
                {
                    Some(arr.len() as u32)
                } else { None }
            })
            .expect("cannot determine vocab_size") as usize;

        let n_experts = g.arch_u32_any(&["expert_count", "experts_count", "num_experts"])
            .unwrap_or(0) as usize;
        let n_experts_used = g.arch_u32_any(&["expert_used_count", "experts_used"])
            .unwrap_or(if n_experts > 0 { 2 } else { 0 } as u32) as usize;

        let yarn_scale = g.arch_f32_any(&["rope.scaling.factor"]).unwrap_or(1.0);
        let yarn_orig_ctx = g.arch_u32_any(&["rope.scaling.original_context_length"])
            .map(|v| v as usize)
            .unwrap_or(0);

        // gpt-oss uses the clamped OAI SwiGLU in its experts; every other MoE arch
        // (Mixtral, Qwen2/3-MoE, …) uses standard SiLU SwiGLU. Pick the activation per-arch.
        let clamped_swiglu = g.architecture().unwrap_or("llama").contains("gpt-oss");

        Self { vocab_size, embed_dim, n_heads, n_kv_heads, n_layers, ffn_dim,
               rope_theta, head_dim, n_experts, n_experts_used,
               yarn_scale, yarn_orig_ctx, clamped_swiglu }
    }

    fn q_total_dim(&self) -> usize { self.n_heads    * self.head_dim }
    fn kv_total_dim(&self) -> usize { self.n_kv_heads * self.head_dim }
    fn use_yarn(&self) -> bool { self.yarn_scale > 1.0 && self.yarn_orig_ctx > 0 }
}

// ── MoE per-layer metadata (expert weights accessed lazily from mmap) ─────────

pub(crate) struct MoeLayerInfo {
    router_w:   Vec<f32>,    // [n_experts × embed_dim] — small, dequanted upfront
    router_b:   Vec<f32>,    // [n_experts] or empty
    gate_info:  TensorInfo,  // full 3D tensor info; bytes slice accessed at inference
    up_info:    TensorInfo,
    down_info:  TensorInfo,
    gate_bias:  Vec<f32>,    // [n_experts × ffn_dim] or empty
    up_bias:    Vec<f32>,
    down_bias:  Vec<f32>,    // [n_experts × embed_dim] or empty
    expert_in_out_elems: usize,  // embed_dim × ffn_dim (gate/up per expert)
    expert_down_elems:   usize,  // ffn_dim × embed_dim (down per expert)
}

// ── Weights ───────────────────────────────────────────────────────────────────
// Large weight matrices (attn projections, embed, lm_head) are kept as TensorInfo
// and accessed lazily via fused GEMV — no 33-MB f32 intermediates in RAM.

pub(crate) struct Weights {
    _gguf:      Arc<GgufFile>,   // keeps mmap alive
    // Small pre-dequanted weights (norms, biases, MoE routers)
    attn_norm:  Vec<Vec<f32>>,
    ffn_norm:   Vec<Vec<f32>>,
    final_norm: Vec<f32>,
    q_bias:     Vec<Vec<f32>>,
    k_bias:     Vec<Vec<f32>>,
    v_bias:     Vec<Vec<f32>>,
    out_bias:   Vec<Vec<f32>>,
    attn_sinks: Vec<Vec<f32>>,  // [n_layers][n_heads] sink logit per head
    // Large weights — lazy mmap access, GEMV at inference time
    embed:      TensorInfo,      // token_embd.weight [vocab × embed]
    lm_head:    TensorInfo,      // output.weight     [vocab × embed]
    q_proj:     Vec<TensorInfo>, // [n_layers] each [embed × q_total]
    k_proj:     Vec<TensorInfo>,
    v_proj:     Vec<TensorInfo>,
    out_proj:   Vec<TensorInfo>, // [n_layers] each [q_total × embed]
    // Dense FFN (non-MoE) — lazy mmap access, GEMV at inference time
    gate:       Vec<TensorInfo>,
    up:         Vec<TensorInfo>,
    down:       Vec<TensorInfo>,
    // MoE layers (already lazy via TensorInfo inside MoeLayerInfo)
    moe:        Vec<MoeLayerInfo>,
}

impl Weights {
    fn load_gguf(gguf: Arc<GgufFile>, cfg: &Config) -> Self {
        let keep = Arc::clone(&gguf);
        let g = &*gguf;
        let tmap = g.tensor_map();
        let q_d  = cfg.q_total_dim();
        let kv_d = cfg.kv_total_dim();
        let fd   = cfg.ffn_dim;
        let d    = cfg.embed_dim;

        let dq = |name: &str| -> Vec<f32> {
            let info = tmap.get(name)
                .unwrap_or_else(|| panic!("tensor not found: {}", name));
            dequant::dequant(g.tensor_data(info), info.ggml_type, info.n_elems())
        };
        let dq_or_zero = |name: &str, size: usize| -> Vec<f32> {
            if tmap.contains_key(name) { dq(name) } else { vec![0.0f32; size] }
        };
        let dq_or_empty = |name: &str| -> Vec<f32> {
            if tmap.contains_key(name) { dq(name) } else { Vec::new() }
        };
        let ti = |name: &str| -> TensorInfo {
            (*tmap.get(name)
                .unwrap_or_else(|| panic!("tensor not found: {}", name)))
                .clone()
        };

        let is_moe = cfg.n_experts > 0;
        let expert_in_out_elems = d * fd;
        let expert_down_elems   = fd * d;

        let mut attn_norm  = Vec::new();
        let mut q_proj     = Vec::new();
        let mut k_proj     = Vec::new();
        let mut v_proj     = Vec::new();
        let mut out_proj   = Vec::new();
        let mut q_bias     = Vec::new();
        let mut k_bias     = Vec::new();
        let mut v_bias     = Vec::new();
        let mut out_bias   = Vec::new();
        let mut attn_sinks = Vec::new();
        let mut ffn_norm   = Vec::new();
        let mut gate       = Vec::new();
        let mut up         = Vec::new();
        let mut down       = Vec::new();
        let mut moe        = Vec::new();

        for l in 0..cfg.n_layers {
            attn_norm.push(dq(&format!("blk.{}.attn_norm.weight", l)));
            // Large projections → lazy TensorInfo, GEMV at runtime
            q_proj  .push(ti(&format!("blk.{}.attn_q.weight", l)));
            k_proj  .push(ti(&format!("blk.{}.attn_k.weight", l)));
            v_proj  .push(ti(&format!("blk.{}.attn_v.weight", l)));
            out_proj.push(ti(&format!("blk.{}.attn_output.weight", l)));
            q_bias  .push(dq_or_zero(&format!("blk.{}.attn_q.bias", l),    q_d));
            k_bias  .push(dq_or_zero(&format!("blk.{}.attn_k.bias", l),    kv_d));
            v_bias  .push(dq_or_zero(&format!("blk.{}.attn_v.bias", l),    kv_d));
            out_bias.push(dq_or_empty(&format!("blk.{}.attn_output.bias", l)));
            attn_sinks.push(dq_or_empty(&format!("blk.{}.attn_sinks.weight", l)));

            let post_attn_key = format!("blk.{}.post_attention_norm.weight", l);
            let ffn_norm_key  = format!("blk.{}.ffn_norm.weight", l);
            ffn_norm.push(if tmap.contains_key(post_attn_key.as_str()) {
                dq(&post_attn_key)
            } else {
                dq(&ffn_norm_key)
            });

            if is_moe {
                let router_w = dq(&format!("blk.{}.ffn_gate_inp.weight", l));
                let router_b = dq_or_empty(&format!("blk.{}.ffn_gate_inp.bias", l));
                let gate_info = ti(&format!("blk.{}.ffn_gate_exps.weight", l));
                let up_info   = ti(&format!("blk.{}.ffn_up_exps.weight", l));
                let down_info = ti(&format!("blk.{}.ffn_down_exps.weight", l));
                let gate_bias = dq_or_empty(&format!("blk.{}.ffn_gate_exps.bias", l));
                let up_bias   = dq_or_empty(&format!("blk.{}.ffn_up_exps.bias", l));
                let down_bias = dq_or_empty(&format!("blk.{}.ffn_down_exps.bias", l));
                moe.push(MoeLayerInfo {
                    router_w, router_b,
                    gate_info, up_info, down_info,
                    gate_bias, up_bias, down_bias,
                    expert_in_out_elems,
                    expert_down_elems,
                });
            } else {
                gate.push(ti(&format!("blk.{}.ffn_gate.weight", l)));
                up  .push(ti(&format!("blk.{}.ffn_up.weight", l)));
                down.push(ti(&format!("blk.{}.ffn_down.weight", l)));
            }
        }

        Self {
            _gguf: keep,
            embed:      ti("token_embd.weight"),
            lm_head:    ti("output.weight"),
            final_norm: dq("output_norm.weight"),
            attn_norm, q_proj, k_proj, v_proj, out_proj,
            q_bias, k_bias, v_bias, out_bias, attn_sinks,
            ffn_norm, gate, up, down, moe,
        }
    }
}

// ── GPU dispatch ──────────────────────────────────────────────────────────────
// When compiled with --features cuda, all large GEMV calls go to the RTX GPU.
// When compiled without, GpuState is a zero-size type and every method falls
// through to the same dequant::gemv path as before.

pub(crate) struct GpuState {
    #[cfg(feature = "cuda")] w:  cuda::GpuWeights,
    #[cfg(feature = "cuda")] sc: cuda::GpuScratch,
    #[cfg(feature = "cuda")] fs: cuda::GpuForwardState,
}

impl GpuState {
    #[cfg(not(feature = "cuda"))]
    pub fn new_cpu() -> Self { Self {} }

    #[cfg(feature = "cuda")]
    pub fn new(weights: &Weights, gguf: &GgufFile, cfg: &Config) -> Self {
        let mut w = cuda::GpuWeights::from_weights(weights, gguf);
        let sc = cuda::GpuScratch::new(cfg.q_total_dim(), cfg.vocab_size);
        let fs = cuda::GpuForwardState::new(weights, cfg);
        // Upload lm_head AFTER forward state so KV cache + scratch don't OOM first.
        w.try_upload_lm_head(weights, gguf);
        Self { w, sc, fs }
    }

    #[cfg(feature = "cuda")]
    fn fwd(&self, token: usize, pos: usize, need_logits: bool,
            cfg: &Config, w: &Weights, gguf: &GgufFile) -> Vec<f32> {
        self.fs.forward_gpu_step(token, pos, need_logits, cfg, w, &self.w, gguf)
    }

    fn gemv_q(&self, l: usize, x: &[f32], ti: &TensorInfo, gguf: &GgufFile, id: usize, od: usize) -> Vec<f32> {
        #[cfg(feature = "cuda")] { let _ = (ti, gguf); return self.sc.gemv(x, &self.w.q_proj[l], self.w.q_proj_type[l], id, od); }
        #[cfg(not(feature = "cuda"))] { let _ = l; dequant::gemv(x, gguf.tensor_data(ti), ti.ggml_type, id, od) }
    }
    fn gemv_k(&self, l: usize, x: &[f32], ti: &TensorInfo, gguf: &GgufFile, id: usize, od: usize) -> Vec<f32> {
        #[cfg(feature = "cuda")] { let _ = (ti, gguf); return self.sc.gemv(x, &self.w.k_proj[l], self.w.k_proj_type[l], id, od); }
        #[cfg(not(feature = "cuda"))] { let _ = l; dequant::gemv(x, gguf.tensor_data(ti), ti.ggml_type, id, od) }
    }
    fn gemv_v(&self, l: usize, x: &[f32], ti: &TensorInfo, gguf: &GgufFile, id: usize, od: usize) -> Vec<f32> {
        #[cfg(feature = "cuda")] { let _ = (ti, gguf); return self.sc.gemv(x, &self.w.v_proj[l], self.w.v_proj_type[l], id, od); }
        #[cfg(not(feature = "cuda"))] { let _ = l; dequant::gemv(x, gguf.tensor_data(ti), ti.ggml_type, id, od) }
    }
    fn gemv_out(&self, l: usize, x: &[f32], ti: &TensorInfo, gguf: &GgufFile, id: usize, od: usize) -> Vec<f32> {
        #[cfg(feature = "cuda")] { let _ = (ti, gguf); return self.sc.gemv(x, &self.w.out_proj[l], self.w.out_proj_type[l], id, od); }
        #[cfg(not(feature = "cuda"))] { let _ = l; dequant::gemv(x, gguf.tensor_data(ti), ti.ggml_type, id, od) }
    }
    fn gemv_lm_head(&self, x: &[f32], ti: &TensorInfo, gguf: &GgufFile, id: usize, od: usize) -> Vec<f32> {
        // lm_head kept on CPU to save ~1.5 GB VRAM; BF16 rayon gemv is still fast.
        let _ = self;
        dequant::gemv(x, gguf.tensor_data(ti), ti.ggml_type, id, od)
    }
    fn gemv_moe_gate(&self, ml: usize, expert: usize, x: &[f32], info: &TensorInfo, gguf: &GgufFile, id: usize, od: usize) -> Vec<f32> {
        #[cfg(feature = "cuda")] { let _ = (info, gguf); let sl = cuda::gpu_expert_slice(&self.w.moe_gate[ml], self.w.moe_gate_type[ml], expert, id * od); return self.sc.gemv_slice(x, &sl, id, od); }
        #[cfg(not(feature = "cuda"))] { let _ = ml; dequant::gemv(x, get_expert_slice(gguf, info, expert, id * od), info.ggml_type, id, od) }
    }
    fn gemv_moe_up(&self, ml: usize, expert: usize, x: &[f32], info: &TensorInfo, gguf: &GgufFile, id: usize, od: usize) -> Vec<f32> {
        #[cfg(feature = "cuda")] { let _ = (info, gguf); let sl = cuda::gpu_expert_slice(&self.w.moe_up[ml], self.w.moe_up_type[ml], expert, id * od); return self.sc.gemv_slice(x, &sl, id, od); }
        #[cfg(not(feature = "cuda"))] { let _ = ml; dequant::gemv(x, get_expert_slice(gguf, info, expert, id * od), info.ggml_type, id, od) }
    }
    fn gemv_moe_down(&self, ml: usize, expert: usize, x: &[f32], info: &TensorInfo, gguf: &GgufFile, id: usize, od: usize) -> Vec<f32> {
        #[cfg(feature = "cuda")] { let _ = (info, gguf); let sl = cuda::gpu_expert_slice(&self.w.moe_down[ml], self.w.moe_down_type[ml], expert, id * od); return self.sc.gemv_slice(x, &sl, id, od); }
        #[cfg(not(feature = "cuda"))] { let _ = ml; dequant::gemv(x, get_expert_slice(gguf, info, expert, id * od), info.ggml_type, id, od) }
    }
}

// ── Ops ───────────────────────────────────────────────────────────────────────

fn rms_norm(x: &[f32], w: &[f32]) -> Vec<f32> {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5).sqrt();
    x.iter().enumerate().map(|(i, &v)| v / rms * w[i]).collect()
}

// Used for small pre-dequanted weights (router, norms, and biases).
pub(crate) fn matmul(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    (0..out_dim).into_par_iter().map(|j| {
        let row = &w[j * in_dim..(j + 1) * in_dim];
        x.iter().zip(row).map(|(&a, &b)| a * b).sum()
    }).collect()
}

// NeoX-style RoPE with optional YARN scaling (GPT-oss uses yarn, scale=32, orig_ctx=4096).
fn rope_yarn(x: &mut [f32], pos: usize, dim: usize, theta: f32,
             yarn_scale: f32, yarn_orig_ctx: usize) {
    use std::f32::consts::PI;
    let half = dim / 2;
    let beta_fast: f32 = 32.0;
    let beta_slow: f32 = 1.0;
    let hfw = yarn_orig_ctx as f32 / beta_fast;  // high-freq wavelen threshold
    let lfw = yarn_orig_ctx as f32 / beta_slow;  // low-freq  wavelen threshold

    for i in 0..half {
        let std_inv = 1.0f32 / theta.powf(2.0 * i as f32 / dim as f32);
        let eff_inv = if yarn_scale <= 1.0 || yarn_orig_ctx == 0 {
            std_inv
        } else {
            let wavelen = 2.0 * PI / std_inv;
            let ramp = ((lfw - wavelen) / (lfw - hfw)).clamp(0.0, 1.0);
            ramp * std_inv + (1.0 - ramp) * (std_inv / yarn_scale)
        };
        let angle = pos as f32 * eff_inv;
        let (sin, cos) = angle.sin_cos();
        let x0 = x[i];
        let x1 = x[i + half];
        x[i]        = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = x.iter_mut().map(|v| { *v = (*v - max).exp(); *v }).sum();
    x.iter_mut().for_each(|v| *v /= sum);
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

// GPT-oss OAI SwiGLU: clamp gate to [−∞,7], up to [-7,7], alpha=1.702, offset up by +1.
fn swiglu_oai(gate: f32, up: f32) -> f32 {
    const ALPHA: f32 = 1.702;
    const LIMIT: f32 = 7.0;
    let x = gate.min(LIMIT);
    let y = up.clamp(-LIMIT, LIMIT);
    (x / (1.0 + (-ALPHA * x).exp())) * (y + 1.0)
}

// ── Attention (GQA) ───────────────────────────────────────────────────────────

fn attention(
    q_ti: &TensorInfo, k_ti: &TensorInfo, v_ti: &TensorInfo, o_ti: &TensorInfo,
    gguf: &GgufFile,
    q_bias: &[f32], k_bias: &[f32], v_bias: &[f32],
    sinks:  &[f32],
    x:      &[f32],
    cfg:    &Config,
    pos:    usize,
    kv_cache: &mut Vec<(Vec<f32>, Vec<f32>)>,
    layer:  usize,
    gpu:    &GpuState,
) -> Vec<f32> {
    let d    = cfg.embed_dim;
    let h    = cfg.n_heads;
    let kv   = cfg.n_kv_heads;
    let hd   = cfg.head_dim;
    let q_d  = cfg.q_total_dim();
    let kv_d = cfg.kv_total_dim();

    let mut q = gpu.gemv_q(layer, x, q_ti, gguf, d, q_d);
    let mut k = gpu.gemv_k(layer, x, k_ti, gguf, d, kv_d);
    let mut v = gpu.gemv_v(layer, x, v_ti, gguf, d, kv_d);

    for i in 0..q_d  { q[i] += q_bias[i]; }
    for i in 0..kv_d { k[i] += k_bias[i]; }
    for i in 0..kv_d { v[i] += v_bias[i]; }

    let (ys, yo) = (cfg.yarn_scale, cfg.yarn_orig_ctx);
    for head in 0..h  {
        rope_yarn(&mut q[head*hd..(head+1)*hd], pos, hd, cfg.rope_theta, ys, yo);
    }
    for head in 0..kv {
        rope_yarn(&mut k[head*hd..(head+1)*hd], pos, hd, cfg.rope_theta, ys, yo);
    }

    kv_cache[layer].0.extend_from_slice(&k);
    kv_cache[layer].1.extend_from_slice(&v);

    let seq   = pos + 1;
    // YARN: GGML applies mscale to rope(Q) AND rope(K), so dot product gets mscale².
    let mscale2 = if cfg.yarn_scale > 1.0 && cfg.yarn_orig_ctx > 0 {
        let m = 1.0_f32 + 0.1 * cfg.yarn_scale.ln();
        m * m
    } else { 1.0_f32 };
    let scale = (hd as f32).sqrt().recip() * mscale2;
    let mut attn_out = vec![0.0f32; q_d];

    for head in 0..h {
        let kv_head = (head * kv) / h;
        let q_h = &q[head*hd..(head+1)*hd];

        let mut scores: Vec<f32> = (0..seq).map(|t| {
            let k_t = &kv_cache[layer].0[t * kv_d + kv_head * hd
                                        ..t * kv_d + (kv_head+1) * hd];
            q_h.iter().zip(k_t).map(|(a, b)| a * b).sum::<f32>() * scale
        }).collect();

        let sink = if sinks.len() > head { sinks[head] } else { f32::NEG_INFINITY };

        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max).max(sink);
        let mut sum: f32 = scores.iter_mut().map(|s| { *s = (*s - max).exp(); *s }).sum();
        if sink > f32::NEG_INFINITY { sum += (sink - max).exp(); }
        scores.iter_mut().for_each(|s| *s /= sum);

        let out_h = &mut attn_out[head*hd..(head+1)*hd];
        for t in 0..seq {
            let v_t = &kv_cache[layer].1[t * kv_d + kv_head * hd
                                        ..t * kv_d + (kv_head+1) * hd];
            for i in 0..hd { out_h[i] += scores[t] * v_t[i]; }
        }
    }

    gpu.gemv_out(layer, &attn_out, o_ti, gguf, q_d, d)
}

// ── Dense FFN (SwiGLU) — lazy GGUF GEMV ──────────────────────────────────────

pub(crate) fn ffn_lazy(
    x: &[f32],
    gate_ti: &TensorInfo,
    up_ti: &TensorInfo,
    down_ti: &TensorInfo,
    gguf: &GgufFile,
    cfg: &Config,
) -> Vec<f32> {
    let d  = cfg.embed_dim;
    let fd = cfg.ffn_dim;
    let gate = dequant::gemv(x, gguf.tensor_data(gate_ti), gate_ti.ggml_type, d, fd);
    let up   = dequant::gemv(x, gguf.tensor_data(up_ti),   up_ti.ggml_type,   d, fd);
    let h: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
    dequant::gemv(&h, gguf.tensor_data(down_ti), down_ti.ggml_type, fd, d)
}

// ── MoE FFN — lazy per-expert dequant ─────────────────────────────────────────

fn get_expert_slice<'a>(
    gguf: &'a GgufFile,
    info: &TensorInfo,
    expert: usize,
    n_elems_per_expert: usize,
) -> &'a [u8] {
    let bytes = ggml_type_size(info.ggml_type, n_elems_per_expert);
    let all_data = gguf.tensor_data(info);
    &all_data[expert * bytes..(expert + 1) * bytes]
}

fn moe_ffn(x: &[f32], moe: &MoeLayerInfo, gguf: &GgufFile, cfg: &Config, moe_layer: usize, gpu: &GpuState) -> Vec<f32> {
    let d  = cfg.embed_dim;
    let fd = cfg.ffn_dim;
    let ne = cfg.n_experts;
    let k  = cfg.n_experts_used;

    let mut logits = matmul(x, &moe.router_w, d, ne);
    if !moe.router_b.is_empty() {
        logits.iter_mut().zip(&moe.router_b).for_each(|(v, b)| *v += b);
    }

    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let top_k = &indexed[..k];

    let max_v = top_k.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = top_k.iter().map(|&(_, v)| (v - max_v).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();

    let mut output = vec![0.0f32; d];

    for (i, &(eid, _)) in top_k.iter().enumerate() {
        let rw = exps[i] / sum_exp;

        let mut g = gpu.gemv_moe_gate(moe_layer, eid, x, &moe.gate_info, gguf, d, fd);
        let mut u = gpu.gemv_moe_up  (moe_layer, eid, x, &moe.up_info,   gguf, d, fd);

        if !moe.gate_bias.is_empty() {
            let off = eid * fd;
            g.iter_mut().zip(&moe.gate_bias[off..off+fd]).for_each(|(a, b)| *a += b);
        }
        if !moe.up_bias.is_empty() {
            let off = eid * fd;
            u.iter_mut().zip(&moe.up_bias[off..off+fd]).for_each(|(a, b)| *a += b);
        }

        // gpt-oss = clamped OAI SwiGLU; other MoE archs (Mixtral, Qwen3-MoE) = standard SiLU.
        let h: Vec<f32> = if cfg.clamped_swiglu {
            g.iter().zip(&u).map(|(&gv, &uv)| swiglu_oai(gv, uv)).collect()
        } else {
            g.iter().zip(&u).map(|(&gv, &uv)| silu(gv) * uv).collect()
        };
        let mut out = gpu.gemv_moe_down(moe_layer, eid, &h, &moe.down_info, gguf, fd, d);

        if !moe.down_bias.is_empty() {
            let off = eid * d;
            out.iter_mut().zip(&moe.down_bias[off..off+d]).for_each(|(a, b)| *a += b);
        }

        output.iter_mut().zip(&out).for_each(|(a, b)| *a += rw * b);
    }

    output
}

// ── Forward pass ──────────────────────────────────────────────────────────────

/// Embed `token` and run all transformer layers, populating the KV cache.
/// Returns the final hidden state before final_norm and lm_head.
fn embed_and_layers(
    token: usize,
    pos:   usize,
    cfg:   &Config,
    w:     &Weights,
    kv:    &mut Vec<(Vec<f32>, Vec<f32>)>,
    gpu:   &GpuState,
) -> Vec<f32> {
    let d    = cfg.embed_dim;
    let gguf = &*w._gguf;

    // Lazy single-row embed lookup — dequants 2160B from mmap, not 2.32GB f32.
    let embed_data = gguf.tensor_data(&w.embed);
    let embed_row_bytes = ggml_type_size(w.embed.ggml_type, d);
    assert!(token < cfg.vocab_size, "token {} out of vocab range {}", token, cfg.vocab_size);
    let embed_slice = &embed_data[token * embed_row_bytes..(token+1) * embed_row_bytes];
    let mut x = dequant::dequant(embed_slice, w.embed.ggml_type, d);

    for l in 0..cfg.n_layers {
        let res = x.clone();
        let n1  = rms_norm(&x, &w.attn_norm[l]);
        let mut a = attention(
            &w.q_proj[l], &w.k_proj[l], &w.v_proj[l], &w.out_proj[l],
            gguf,
            &w.q_bias[l], &w.k_bias[l], &w.v_bias[l],
            &w.attn_sinks[l],
            &n1, cfg, pos, kv, l, gpu);
        if !w.out_bias[l].is_empty() {
            a.iter_mut().zip(&w.out_bias[l]).for_each(|(v, b)| *v += b);
        }
        x = res.iter().zip(&a).map(|(r, av)| r + av).collect();

        let res = x.clone();
        let n2  = rms_norm(&x, &w.ffn_norm[l]);
        let f = if cfg.n_experts > 0 {
            moe_ffn(&n2, &w.moe[l], gguf, cfg, l, gpu)
        } else {
            ffn_lazy(&n2, &w.gate[l], &w.up[l], &w.down[l], gguf, cfg)
        };
        x = res.iter().zip(&f).map(|(r, fv)| r + fv).collect();
    }
    x
}

fn forward(
    token: usize,
    pos:   usize,
    cfg:   &Config,
    w:     &Weights,
    kv:    &mut Vec<(Vec<f32>, Vec<f32>)>,
    gpu:   &GpuState,
) -> Vec<f32> {
    let x    = embed_and_layers(token, pos, cfg, w, kv, gpu);
    let d    = cfg.embed_dim;
    let gguf = &*w._gguf;
    let x    = rms_norm(&x, &w.final_norm);
    gpu.gemv_lm_head(&x, &w.lm_head, gguf, d, cfg.vocab_size)
}

// ── Sequential prefill ────────────────────────────────────────────────────────
// For all but the last prompt token, run only the transformer layers to populate
// the KV cache — skipping the 1.16 GB lm_head read that forward() would do.
// The last token runs the full forward() to obtain the first logits.

fn prefill(
    tokens: &[usize],
    cfg:    &Config,
    w:      &Weights,
    kv:     &mut Vec<(Vec<f32>, Vec<f32>)>,
    gpu:    &GpuState,
) -> Vec<f32> {
    assert!(!tokens.is_empty(), "empty prompt");
    // Batched prefill (opt-in via KAIRO_PREFILL=batch) is verified-correct but, on
    // CPU, no faster than the AVX2-fused per-token path here (the win from
    // amortizing weight decode is cancelled by activation memory locality). The
    // real prefill fix is integer SIMD kernels, a separate project. Default to the
    // proven sequential path.
    let use_batched = std::env::var("KAIRO_PREFILL").map(|v| v == "batch").unwrap_or(false);
    if cfg.n_experts == 0 && use_batched {
        return prefill_batched(tokens, cfg, w, kv, gpu);
    }
    let n = tokens.len();
    for i in 0..n - 1 {
        embed_and_layers(tokens[i], i, cfg, w, kv, gpu);
    }
    forward(tokens[n - 1], n - 1, cfg, w, kv, gpu)
}

// Batched prefill for dense models. Runs the whole prompt through each layer with
// batched gemm (each weight decoded once per layer, not once per token) — the CPU
// prefill speedup. Per-token math mirrors attention()/ffn_lazy()/forward() exactly
// so its output matches the sequential path. Returns the last token's logits.
fn prefill_batched(
    tokens: &[usize],
    cfg:    &Config,
    w:      &Weights,
    kv:     &mut Vec<(Vec<f32>, Vec<f32>)>,
    gpu:    &GpuState,
) -> Vec<f32> {
    let n    = tokens.len();
    let d    = cfg.embed_dim;
    let h    = cfg.n_heads;
    let kvh  = cfg.n_kv_heads;
    let hd   = cfg.head_dim;
    let q_d  = cfg.q_total_dim();
    let kv_d = cfg.kv_total_dim();
    let fd   = cfg.ffn_dim;
    let gguf = &*w._gguf;

    // Embed all tokens -> X [n, d]
    let embed_data = gguf.tensor_data(&w.embed);
    let embed_row_bytes = ggml_type_size(w.embed.ggml_type, d);
    let mut x = vec![0.0f32; n * d];
    for t in 0..n {
        let tok = tokens[t];
        assert!(tok < cfg.vocab_size, "token {} out of vocab range {}", tok, cfg.vocab_size);
        let slice = &embed_data[tok * embed_row_bytes..(tok + 1) * embed_row_bytes];
        let row = dequant::dequant(slice, w.embed.ggml_type, d);
        x[t * d..(t + 1) * d].copy_from_slice(&row);
    }

    let (ys, yo) = (cfg.yarn_scale, cfg.yarn_orig_ctx);
    let mscale2 = if cfg.yarn_scale > 1.0 && cfg.yarn_orig_ctx > 0 {
        let m = 1.0_f32 + 0.1 * cfg.yarn_scale.ln();
        m * m
    } else { 1.0_f32 };
    let scale = (hd as f32).sqrt().recip() * mscale2;

    for l in 0..cfg.n_layers {
        let base = kv[l].0.len() / kv_d;  // absolute position of the first new token

        // Attention: normed input -> Q/K/V (batched), bias+RoPE per token.
        let mut n1 = vec![0.0f32; n * d];
        for t in 0..n {
            let nn = rms_norm(&x[t * d..(t + 1) * d], &w.attn_norm[l]);
            n1[t * d..(t + 1) * d].copy_from_slice(&nn);
        }
        let (qti, kti, vti, oti) = (&w.q_proj[l], &w.k_proj[l], &w.v_proj[l], &w.out_proj[l]);
        let mut q = dequant::gemm(&n1, gguf.tensor_data(qti), qti.ggml_type, d, q_d, n);
        let mut k = dequant::gemm(&n1, gguf.tensor_data(kti), kti.ggml_type, d, kv_d, n);
        let mut v = dequant::gemm(&n1, gguf.tensor_data(vti), vti.ggml_type, d, kv_d, n);

        for t in 0..n {
            let pos = base + t;
            let qt = &mut q[t * q_d..(t + 1) * q_d];
            for i in 0..q_d { qt[i] += w.q_bias[l][i]; }
            for head in 0..h { rope_yarn(&mut qt[head * hd..(head + 1) * hd], pos, hd, cfg.rope_theta, ys, yo); }
            let kt = &mut k[t * kv_d..(t + 1) * kv_d];
            for i in 0..kv_d { kt[i] += w.k_bias[l][i]; }
            for head in 0..kvh { rope_yarn(&mut kt[head * hd..(head + 1) * hd], pos, hd, cfg.rope_theta, ys, yo); }
            let vt = &mut v[t * kv_d..(t + 1) * kv_d];
            for i in 0..kv_d { vt[i] += w.v_bias[l][i]; }
        }
        // Append K/V to the cache in position order before attending.
        for t in 0..n {
            kv[l].0.extend_from_slice(&k[t * kv_d..(t + 1) * kv_d]);
            kv[l].1.extend_from_slice(&v[t * kv_d..(t + 1) * kv_d]);
        }

        // Per-token causal attention (each token attends positions 0..=its own).
        let mut attn_out = vec![0.0f32; n * q_d];
        for t in 0..n {
            let pos = base + t;
            let seq = pos + 1;
            let qt = &q[t * q_d..(t + 1) * q_d];
            for head in 0..h {
                let kv_head = (head * kvh) / h;
                let q_h = &qt[head * hd..(head + 1) * hd];
                let sink = if w.attn_sinks[l].len() > head { w.attn_sinks[l][head] } else { f32::NEG_INFINITY };
                let mut scores: Vec<f32> = (0..seq).map(|p| {
                    let k_p = &kv[l].0[p * kv_d + kv_head * hd..p * kv_d + (kv_head + 1) * hd];
                    q_h.iter().zip(k_p).map(|(a, b)| a * b).sum::<f32>() * scale
                }).collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max).max(sink);
                let mut sum: f32 = scores.iter_mut().map(|s| { *s = (*s - max).exp(); *s }).sum();
                if sink > f32::NEG_INFINITY { sum += (sink - max).exp(); }
                scores.iter_mut().for_each(|s| *s /= sum);
                let out_h = &mut attn_out[t * q_d + head * hd..t * q_d + (head + 1) * hd];
                for p in 0..seq {
                    let v_p = &kv[l].1[p * kv_d + kv_head * hd..p * kv_d + (kv_head + 1) * hd];
                    for i in 0..hd { out_h[i] += scores[p] * v_p[i]; }
                }
            }
        }

        // Output projection (batched) + out_bias + residual.
        let ao = dequant::gemm(&attn_out, gguf.tensor_data(oti), oti.ggml_type, q_d, d, n);
        let has_obias = !w.out_bias[l].is_empty();
        for t in 0..n {
            for i in 0..d {
                let mut av = ao[t * d + i];
                if has_obias { av += w.out_bias[l][i]; }
                x[t * d + i] += av;
            }
        }

        // Dense FFN (SwiGLU), batched.
        let mut n2 = vec![0.0f32; n * d];
        for t in 0..n {
            let nn = rms_norm(&x[t * d..(t + 1) * d], &w.ffn_norm[l]);
            n2[t * d..(t + 1) * d].copy_from_slice(&nn);
        }
        let (gti, uti, dti) = (&w.gate[l], &w.up[l], &w.down[l]);
        let gate = dequant::gemm(&n2, gguf.tensor_data(gti), gti.ggml_type, d, fd, n);
        let up   = dequant::gemm(&n2, gguf.tensor_data(uti), uti.ggml_type, d, fd, n);
        let mut hbuf = vec![0.0f32; n * fd];
        for i in 0..n * fd { hbuf[i] = silu(gate[i]) * up[i]; }
        let down = dequant::gemm(&hbuf, gguf.tensor_data(dti), dti.ggml_type, fd, d, n);
        for i in 0..n * d { x[i] += down[i]; }
    }

    let last = &x[(n - 1) * d..n * d];
    let xf = rms_norm(last, &w.final_norm);
    gpu.gemv_lm_head(&xf, &w.lm_head, gguf, d, cfg.vocab_size)
}

fn emit_utf8<F>(pending: &mut Vec<u8>, bytes: &[u8], is_reasoning: bool, on_token: &mut F)
where
    F: FnMut(&str, bool),
{
    pending.extend_from_slice(bytes);
    loop {
        if pending.is_empty() {
            break;
        }
        match std::str::from_utf8(pending) {
            Ok(s) => {
                let piece = s.to_string();
                pending.clear();
                if !piece.is_empty() {
                    on_token(&piece, is_reasoning);
                }
                break;
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    let piece = String::from_utf8_lossy(&pending[..valid]).into_owned();
                    pending.drain(..valid);
                    if !piece.is_empty() {
                        on_token(&piece, is_reasoning);
                    }
                    continue;
                }
                if let Some(len) = e.error_len() {
                    pending.drain(..len);
                    on_token("\u{FFFD}", is_reasoning);
                    continue;
                }
                break;
            }
        }
    }
}

// ── Generation loop ───────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn generate_gpu<F, P>(
    prompt_ids:     &[usize],
    max_new:        usize,
    temperature:    f32,
    top_k:          usize,
    repeat_penalty: f32,
    seed:           Option<u64>,
    cfg:            &Config,
    w:              &Weights,
    gpu:            &GpuState,
    tok:            Option<&Tokenizer>,
    mut on_token:   F,
    mut on_prefill: P,
) -> Vec<usize>
where F: FnMut(&str, bool), P: FnMut(usize, usize)
{
    use rand::{SeedableRng, RngCore};
    use rand::rngs::StdRng;

    assert!(!prompt_ids.is_empty(), "empty prompt");
    let gguf = &*w._gguf;
    let n    = prompt_ids.len();

    // Prefill: all tokens except last populate the KV cache without computing logits
    for i in 0..n - 1 {
        on_prefill(i, n);
        gpu.fwd(prompt_ids[i], i, false, cfg, w, gguf);
    }
    on_prefill(n - 1, n);
    let mut logits = gpu.fwd(prompt_ids[n - 1], n - 1, true, cfg, w, gguf);
    let mut pos    = n;

    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None    => StdRng::seed_from_u64(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().subsec_nanos() as u64),
    };

    // GPT-oss channel tracking.
    // The model can emit either:
    //   <|channel|>analysis<|message|>  (explicit channel token)
    //   analysis<|message|>             (channel name as plain text, no <|channel|>)
    // We handle both via a one-token lookahead: short alphabetic tokens are buffered;
    // if <|message|> follows, we treat the buffer as the channel name.
    let has_native = tok.map(|t| t.message_id != u32::MAX).unwrap_or(false);
    let mut is_reasoning  = false;
    let mut in_chan_decl  = false;
    let mut chan_name_buf = String::new();
    let mut lookahead: Option<Vec<u8>> = None; // one-token channel-name probe

    let mut out = Vec::new();
    let mut pending_utf8 = Vec::new();

    for _ in 0..max_new {
        if repeat_penalty > 1.0 {
            let window_start = out.len().saturating_sub(64);
            for &tok_id in &out[window_start..] {
                let v = &mut logits[tok_id];
                if *v > 0.0 { *v /= repeat_penalty; } else { *v *= repeat_penalty; }
            }
        }

        let next = if temperature <= 0.0 {
            logits.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0
        } else {
            let k = top_k.min(logits.len());
            let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
                .map(|(i, &v)| (i, v / temperature)).collect();
            indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            indexed.truncate(k);
            let max_v = indexed[0].1;
            let mut probs: Vec<f32> = indexed.iter()
                .map(|(_, v)| (*v - max_v).exp()).collect();
            let sum: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= sum);
            let r: f32 = (rng.next_u64() as f64 / u64::MAX as f64) as f32;
            let mut cum = 0.0f32;
            let mut chosen = indexed[0].0;
            for (i, p) in probs.iter().enumerate() {
                cum += p;
                if r <= cum { chosen = indexed[i].0; break; }
            }
            chosen
        };

        let is_eos = match tok { Some(t) => t.is_eos(next as u32), None => next == 2 };
        if is_eos {
            break;
        }

        // GPT-oss <|end|>: closes current channel message, not the full turn.
        // Flush pending output and continue — the model will generate the next channel.
        if let Some(t) = tok {
            if t.is_channel_end(next as u32) {
                if let Some(bytes) = lookahead.take() {
                    emit_utf8(&mut pending_utf8, &bytes, is_reasoning, &mut on_token);
                }
                if !pending_utf8.is_empty() {
                    let piece = String::from_utf8_lossy(&pending_utf8).into_owned();
                    if !piece.is_empty() { on_token(&piece, is_reasoning); }
                    pending_utf8.clear();
                }
                out.push(next);
                logits = gpu.fwd(next, pos, true, cfg, w, gguf);
                pos += 1;
                continue;
            }
        }

        match tok {
            Some(t) => {
                let tid = next as u32;
                if tid == t.channel_id {
                    // Explicit <|channel|> token: flush lookahead (not a channel name),
                    // enter channel declaration mode.
                    if let Some(bytes) = lookahead.take() {
                        emit_utf8(&mut pending_utf8, &bytes, is_reasoning, &mut on_token);
                    }
                    in_chan_decl = true;
                    chan_name_buf.clear();
                } else if tid == t.message_id {
                    // Channel boundary: resolve via lookahead (text<|message|> format)
                    // or via chan_name_buf (<|channel|>text<|message|> format).
                    let name = if let Some(bytes) = lookahead.take() {
                        String::from_utf8_lossy(&bytes).trim().to_lowercase()
                    } else if in_chan_decl {
                        let n = chan_name_buf.trim().to_lowercase();
                        in_chan_decl = false;
                        chan_name_buf.clear();
                        n
                    } else {
                        String::new()
                    };
                    if name == "analysis" {
                        is_reasoning = true;
                        pending_utf8.clear();
                    } else if name == "final" {
                        is_reasoning = false;
                        pending_utf8.clear();
                    }
                } else if in_chan_decl {
                    if !t.is_special(tid) {
                        chan_name_buf.push_str(
                            &String::from_utf8_lossy(&t.token_bytes(tid)));
                        // Safety: if name grows long, it's not a channel name.
                        if chan_name_buf.len() > 20 {
                            emit_utf8(&mut pending_utf8, chan_name_buf.as_bytes(), is_reasoning, &mut on_token);
                            chan_name_buf.clear();
                            in_chan_decl = false;
                        }
                    }
                } else if !t.is_special(tid) {
                    let bytes = t.token_bytes(tid);
                    if has_native {
                        // Buffer short alphabetic tokens — they may be channel names.
                        let decoded = String::from_utf8_lossy(&bytes);
                        let trimmed = decoded.trim();
                        if !trimmed.is_empty()
                            && trimmed.len() <= 15
                            && trimmed.chars().all(|c| c.is_alphabetic())
                        {
                            // Flush previous lookahead first (wasn't followed by <|message|>)
                            if let Some(prev) = lookahead.take() {
                                emit_utf8(&mut pending_utf8, &prev, is_reasoning, &mut on_token);
                            }
                            lookahead = Some(bytes);
                        } else {
                            if let Some(prev) = lookahead.take() {
                                emit_utf8(&mut pending_utf8, &prev, is_reasoning, &mut on_token);
                            }
                            emit_utf8(&mut pending_utf8, &bytes, is_reasoning, &mut on_token);
                        }
                    } else {
                        emit_utf8(&mut pending_utf8, &bytes, is_reasoning, &mut on_token);
                    }
                }
            }
            None => on_token(&format!("[{}]", next), false),
        }

        out.push(next);
        logits = gpu.fwd(next, pos, true, cfg, w, gguf);
        pos += 1;
    }

    if let Some(bytes) = lookahead.take() {
        emit_utf8(&mut pending_utf8, &bytes, is_reasoning, &mut on_token);
    }
    if !pending_utf8.is_empty() {
        let piece = String::from_utf8_lossy(&pending_utf8).into_owned();
        if !piece.is_empty() { on_token(&piece, is_reasoning); }
    }
    out
}

#[cfg_attr(feature = "cuda", allow(unreachable_code))]
pub(crate) fn generate<F, P>(
    prompt_ids:     &[usize],
    max_new:        usize,
    temperature:    f32,
    top_k:          usize,
    repeat_penalty: f32,
    seed:           Option<u64>,
    cfg:            &Config,
    w:              &Weights,
    gpu:            &GpuState,
    tok:            Option<&Tokenizer>,
    mut on_token:   F,
    on_prefill:     P,
) -> Vec<usize>
where
    F: FnMut(&str, bool),
    P: FnMut(usize, usize),
{
    #[cfg(feature = "cuda")]
    return generate_gpu(prompt_ids, max_new, temperature, top_k, repeat_penalty,
                        seed, cfg, w, gpu, tok, on_token, on_prefill);

    let _ = on_prefill; // CPU path has no per-token prefill callback

    use rand::{SeedableRng, RngCore};
    use rand::rngs::StdRng;

    assert!(!prompt_ids.is_empty(), "empty prompt");
    let mut kv: Vec<(Vec<f32>, Vec<f32>)> =
        (0..cfg.n_layers).map(|_| (Vec::new(), Vec::new())).collect();

    let mut logits = prefill(prompt_ids, cfg, w, &mut kv, gpu);
    let mut pos    = prompt_ids.len();

    let mut rng: StdRng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None    => StdRng::seed_from_u64(std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().subsec_nanos() as u64),
    };

    let mut out = Vec::new();
    let mut pending_utf8 = Vec::new();
    for _ in 0..max_new {
        // Apply repeat penalty: penalise tokens seen in the last 64 positions
        if repeat_penalty > 1.0 {
            let window_start = out.len().saturating_sub(64);
            for &tok_id in &out[window_start..] {
                let v = &mut logits[tok_id];
                if *v > 0.0 { *v /= repeat_penalty; } else { *v *= repeat_penalty; }
            }
        }

        let next = if temperature <= 0.0 {
            // Greedy: argmax
            logits.iter().enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap().0
        } else {
            // Temperature + top-k sampling
            let k = top_k.min(logits.len());
            let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
                .map(|(i, &v)| (i, v / temperature))
                .collect();
            indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            indexed.truncate(k);

            // Softmax over top-k
            let max_v = indexed[0].1;
            let mut probs: Vec<f32> = indexed.iter().map(|(_, v)| (*v - max_v).exp()).collect();
            let sum: f32 = probs.iter().sum();
            probs.iter_mut().for_each(|p| *p /= sum);

            // Cumulative sample
            let r: f32 = (rng.next_u64() as f64 / u64::MAX as f64) as f32;
            let mut cum = 0.0f32;
            let mut chosen = indexed[0].0;
            for (i, p) in probs.iter().enumerate() {
                cum += p;
                if r <= cum { chosen = indexed[i].0; break; }
            }
            chosen
        };

        let is_eos = match tok {
            Some(t) => t.is_eos(next as u32),
            None    => next == 2,
        };
        if is_eos { break; }

        match tok {
            Some(t) => {
                if !t.is_special(next as u32) {
                    let bytes = t.token_bytes(next as u32);
                    emit_utf8(&mut pending_utf8, &bytes, false, &mut on_token);
                }
            }
            None => {
                let piece = format!("[{}]", next);
                on_token(&piece, false);
            }
        }

        out.push(next);
        logits = forward(next, pos, cfg, w, &mut kv, gpu);
        pos += 1;
    }
    if !pending_utf8.is_empty() {
        let piece = String::from_utf8_lossy(&pending_utf8).into_owned();
        if !piece.is_empty() {
            on_token(&piece, false);
        }
    }
    out
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    use std::io::Write;

    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("tiny_model.gguf");

    println!("Opening {}...", path);
    let gguf = Arc::new(
        GgufFile::open(Path::new(path))
            .unwrap_or_else(|e| panic!("{}", e))
    );

    println!("Architecture : {}", gguf.architecture().unwrap_or("unknown"));

    let cfg = Config::from_gguf(&gguf);
    println!(
        "embed={} heads={} kv_heads={} head_dim={} layers={} ffn={} vocab={} rope_theta={}",
        cfg.embed_dim, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
        cfg.n_layers, cfg.ffn_dim, cfg.vocab_size, cfg.rope_theta
    );
    if cfg.n_experts > 0 {
        println!("MoE          : {} experts, top-{}", cfg.n_experts, cfg.n_experts_used);
    }
    println!("YARN         : scale={} orig_ctx={}", cfg.yarn_scale, cfg.yarn_orig_ctx);

    let tokenizer = Tokenizer::from_gguf(&gguf);
    println!("Tokenizer    : {}", if tokenizer.is_some() { "loaded" } else { "not found" });

    // ── Tensor list mode ────────────────────────────────────────────────────────
    if args.iter().any(|a| a == "--list-tensors") {
        for t in &gguf.tensors {
            println!("{:60} type={} dims={:?}", t.name, t.ggml_type, t.dims);
        }
        return;
    }

    println!("Loading weights...");
    let weights = Weights::load_gguf(gguf.clone(), &cfg);
    #[cfg(feature = "cuda")]
    let gpu = GpuState::new(&weights, &gguf, &cfg);
    #[cfg(not(feature = "cuda"))]
    let gpu = GpuState::new_cpu();
    println!("Ready.");
    if !weights.attn_sinks.is_empty() && !weights.attn_sinks[0].is_empty() {
        let s = &weights.attn_sinks[0];
        println!("AttnSink L0  : min={:.3} max={:.3} mean={:.3}",
            s.iter().cloned().fold(f32::INFINITY, f32::min),
            s.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            s.iter().sum::<f32>() / s.len() as f32);
    }

    // ── Server mode ─────────────────────────────────────────────────────────────
    if let Some(srv_pos) = args.iter().position(|a| a == "--server") {
        let port: u16 = args.get(srv_pos + 1).and_then(|s| s.parse().ok()).unwrap_or(18081);

        // Spawn background thread to warm the mmap page cache for expert weights.
        // This runs AFTER the server starts listening, so it doesn't delay health checks.
        {
            let gguf_warm = gguf.clone();
            std::thread::spawn(move || {
                let mut touched = 0usize;
                for t in &gguf_warm.tensors {
                    let data = gguf_warm.tensor_data(t);
                    let mut _acc = 0u8;
                    for byte in data.iter().step_by(4096) {
                        _acc = _acc.wrapping_add(*byte);
                    }
                    touched += data.len();
                    if touched > 20 * 1024 * 1024 * 1024 { break; }
                }
                eprintln!("tinyq4: mmap warmup complete ({:.1} GB touched)", touched as f64 / 1e9);
            });
        }

        let state = server::AppState {
            cfg:       Arc::new(cfg),
            weights:   Arc::new(weights),
            gpu:       Arc::new(gpu),
            tokenizer: Arc::new(tokenizer),
            model_id:  Path::new(path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("tinyq4")
                .to_string(),
        };
        println!("Starting server on port {}...", port);
        tokio::runtime::Runtime::new().unwrap()
            .block_on(server::run(port, std::sync::Arc::new(state)));
        return;
    }

    // ── CLI mode ─────────────────────────────────────────────────────────────────
    let prompt_text = args.get(2).map(|s| s.as_str()).unwrap_or("What is 2+2?");
    let max_new: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200);
    let nosys = args.iter().any(|a| a == "--nosys");

    let prompt_ids: Vec<usize> = match &tokenizer {
        Some(t) => {
            let ids = if nosys { t.chat_encode_nosys(prompt_text) } else { t.chat_encode(prompt_text) };
            println!("Prompt       : {:?}", prompt_text);
            println!("Tokens       : {} ids", ids.len());
            ids.into_iter().map(|id| id as usize).collect()
        }
        None => vec![1, 42],
    };

    println!("\n--- Response ---");
    generate(&prompt_ids, max_new, 0.0, 40, 1.0, None, &cfg, &weights, &gpu, tokenizer.as_ref(),
        |piece, _| { print!("{}", piece); std::io::stdout().flush().ok(); },
        |done, total| { eprint!("\rprefill {}/{}...", done + 1, total); },
    );
    println!("\n--- End ---");
}
