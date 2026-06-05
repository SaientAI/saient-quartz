// Safe Rust wrappers around the CUDA kernels compiled from cuda_kernels.cu.

use std::ffi::c_void;
use crate::gguf::{GgufFile, TensorInfo, ggml_type_size};

unsafe extern "C" {
    pub fn cuda_gemv(ggml_type: u32, d_data: *const u8, d_x: *const f32,
                     d_y: *mut f32, in_dim: i32, out_dim: i32);
    fn cuda_alloc(bytes: usize) -> *mut c_void;
    fn cuda_drop(ptr: *mut c_void);
    fn cuda_h2d(dst: *mut c_void, src: *const c_void, bytes: usize);
    fn cuda_d2h(dst: *mut c_void, src: *const c_void, bytes: usize);
    pub fn cuda_sync();
    fn cuda_vec_add(dst: *mut f32, src: *const f32, n: i32);
    fn cuda_vec_scale_add(dst: *mut f32, src: *const f32, scale: f32, n: i32);
    fn cuda_vec_zero(dst: *mut f32, n: i32);
    fn cuda_rms_norm(out: *mut f32, x: *const f32, w: *const f32, n: i32);
    fn cuda_rope_yarn(x: *mut f32, n_heads: i32, head_dim: i32, pos: i32,
                      theta: f32, yarn_scale: f32, yarn_orig_ctx: i32);
    fn cuda_kv_append(k_cache: *mut f32, v_cache: *mut f32,
                      k: *const f32, v: *const f32, pos: i32, kv_dim: i32);
    fn cuda_attn_score(scores: *mut f32, q: *const f32, k_cache: *const f32,
                       seq_len: i32, head_dim: i32, kv_dim: i32,
                       kv_head_off: i32, scale: f32);
    fn cuda_softmax_sink(x: *mut f32, n: i32, sink: f32);
    fn cuda_attn_values(out: *mut f32, scores: *const f32, v_cache: *const f32,
                        seq_len: i32, head_dim: i32, kv_dim: i32, kv_head_off: i32);
    fn cuda_swiglu(out: *mut f32, gate: *const f32, up: *const f32, n: i32);
    fn cuda_swiglu_oai(out: *mut f32, gate: *const f32, up: *const f32, n: i32);
    fn cuda_embed_lookup(out: *mut f32, data: *const u8, ggml_type: u32,
                         token: i32, embed_dim: i32);
    // Batched expert GEMV — runs up to 4 experts simultaneously in a 2D grid
    pub fn cuda_gemv_moe4(
        ggml_type: u32,
        w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
        xi0: *const f32, xi1: *const f32, xi2: *const f32, xi3: *const f32,
        o0: *mut f32, o1: *mut f32, o2: *mut f32, o3: *mut f32,
        n_act: i32, in_dim: i32, out_dim: i32,
    );
    // Batched multi-head attention — replaces per-head loop (192 launches → 3)
    fn cuda_attn_score_all(scores: *mut f32, q: *const f32, k_cache: *const f32,
                            n_heads: i32, n_kv_heads: i32, seq_len: i32,
                            head_dim: i32, kv_dim: i32, max_seq: i32, scale: f32);
    fn cuda_softmax_sink_all(scores: *mut f32, n_heads: i32, seq_len: i32,
                              max_seq: i32, sinks: *const f32);
    fn cuda_attn_values_all(out: *mut f32, scores: *const f32, v_cache: *const f32,
                             n_heads: i32, n_kv_heads: i32, seq_len: i32,
                             head_dim: i32, kv_dim: i32, max_seq: i32);
}

// ── GPU buffer ────────────────────────────────────────────────────────────────

pub struct GpuBuf {
    pub ptr:   *mut c_void,
    pub bytes: usize,
}

unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cuda_drop(self.ptr); }
        }
    }
}

impl GpuBuf {
    pub fn upload(data: &[u8]) -> Self {
        let bytes = data.len();
        let ptr = unsafe { cuda_alloc(bytes) };
        assert!(!ptr.is_null(), "cudaMalloc failed for {} bytes", bytes);
        unsafe { cuda_h2d(ptr, data.as_ptr() as *const c_void, bytes); }
        Self { ptr, bytes }
    }

    pub fn alloc_f32(n: usize) -> Self {
        let bytes = n * 4;
        let ptr = unsafe { cuda_alloc(bytes) };
        assert!(!ptr.is_null(), "cudaMalloc failed for {} f32s", n);
        Self { ptr, bytes }
    }
}

// ── Scratch buffers reused across every gemv call ─────────────────────────────

pub struct GpuScratch {
    d_x: GpuBuf,   // input vector (max embed_dim or q_total_dim)
    d_y: GpuBuf,   // output vector (max vocab_size for lm_head)
}

impl GpuScratch {
    pub fn new(max_in: usize, max_out: usize) -> Self {
        Self {
            d_x: GpuBuf::alloc_f32(max_in),
            d_y: GpuBuf::alloc_f32(max_out),
        }
    }

    pub fn gemv(&self, x: &[f32], data: &GpuBuf, ggml_type: u32,
                in_dim: usize, out_dim: usize) -> Vec<f32>
    {
        unsafe {
            cuda_h2d(self.d_x.ptr, x.as_ptr() as *const c_void, x.len() * 4);
            cuda_gemv(
                ggml_type,
                data.ptr as *const u8,
                self.d_x.ptr as *const f32,
                self.d_y.ptr as *mut f32,
                in_dim as i32,
                out_dim as i32,
            );
            let mut out = vec![0.0f32; out_dim];
            cuda_d2h(out.as_mut_ptr() as *mut c_void,
                     self.d_y.ptr as *const c_void,
                     out_dim * 4);
            cuda_sync();
            out
        }
    }
}

// ── GPU weight store ──────────────────────────────────────────────────────────

pub struct GpuWeights {
    pub q_proj:    Vec<GpuBuf>, pub q_proj_type:    Vec<u32>,
    pub k_proj:    Vec<GpuBuf>, pub k_proj_type:    Vec<u32>,
    pub v_proj:    Vec<GpuBuf>, pub v_proj_type:    Vec<u32>,
    pub out_proj:  Vec<GpuBuf>, pub out_proj_type:  Vec<u32>,
    pub gate:      Vec<GpuBuf>, pub gate_type:      Vec<u32>,
    pub up:        Vec<GpuBuf>, pub up_type:        Vec<u32>,
    pub down:      Vec<GpuBuf>, pub down_type:      Vec<u32>,
    pub moe_gate:  Vec<GpuBuf>, pub moe_gate_type:  Vec<u32>,
    pub moe_up:    Vec<GpuBuf>, pub moe_up_type:    Vec<u32>,
    pub moe_down:  Vec<GpuBuf>, pub moe_down_type:  Vec<u32>,
    // lm_head on GPU — eliminates the 190 ms/token CPU BF16 GEMV bottleneck
    pub d_lm_head:    Option<GpuBuf>,
    pub lm_head_type: u32,
}

fn upload_ti(gguf: &GgufFile, ti: &TensorInfo) -> (GpuBuf, u32) {
    (GpuBuf::upload(gguf.tensor_data(ti)), ti.ggml_type)
}

impl GpuWeights {
    pub fn from_weights(w: &crate::Weights, gguf: &GgufFile) -> Self {
        eprint!("GPU: uploading weights... ");
        // embed stays on CPU. lm_head: try to upload — if cudaMalloc fails, fall back to CPU.
        let mut q_proj = Vec::new(); let mut q_proj_type = Vec::new();
        let mut k_proj = Vec::new(); let mut k_proj_type = Vec::new();
        let mut v_proj = Vec::new(); let mut v_proj_type = Vec::new();
        let mut out_proj = Vec::new(); let mut out_proj_type = Vec::new();
        for l in 0..w.q_proj.len() {
            let (b, t) = upload_ti(gguf, &w.q_proj[l]);   q_proj.push(b);  q_proj_type.push(t);
            let (b, t) = upload_ti(gguf, &w.k_proj[l]);   k_proj.push(b);  k_proj_type.push(t);
            let (b, t) = upload_ti(gguf, &w.v_proj[l]);   v_proj.push(b);  v_proj_type.push(t);
            let (b, t) = upload_ti(gguf, &w.out_proj[l]); out_proj.push(b); out_proj_type.push(t);
        }

        let mut gate = Vec::new(); let mut gate_type = Vec::new();
        let mut up   = Vec::new(); let mut up_type   = Vec::new();
        let mut down = Vec::new(); let mut down_type = Vec::new();
        for l in 0..w.gate.len() {
            let (b, t) = upload_ti(gguf, &w.gate[l]); gate.push(b); gate_type.push(t);
            let (b, t) = upload_ti(gguf, &w.up[l]);   up.push(b);   up_type.push(t);
            let (b, t) = upload_ti(gguf, &w.down[l]); down.push(b); down_type.push(t);
        }

        let mut moe_gate = Vec::new(); let mut moe_gate_type = Vec::new();
        let mut moe_up   = Vec::new(); let mut moe_up_type   = Vec::new();
        let mut moe_down = Vec::new(); let mut moe_down_type = Vec::new();
        for m in &w.moe {
            let (b, t) = upload_ti(gguf, &m.gate_info); moe_gate.push(b); moe_gate_type.push(t);
            let (b, t) = upload_ti(gguf, &m.up_info);   moe_up.push(b);   moe_up_type.push(t);
            let (b, t) = upload_ti(gguf, &m.down_info); moe_down.push(b); moe_down_type.push(t);
        }

        unsafe { cuda_sync(); }
        eprintln!("done.");
        Self {
            q_proj, q_proj_type, k_proj, k_proj_type,
            v_proj, v_proj_type, out_proj, out_proj_type,
            gate, gate_type, up, up_type, down, down_type,
            moe_gate, moe_gate_type, moe_up, moe_up_type, moe_down, moe_down_type,
            d_lm_head: None, lm_head_type: 0,
        }
    }

    /// Called after GpuForwardState is allocated so lm_head fits in remaining VRAM.
    pub fn try_upload_lm_head(&mut self, w: &crate::Weights, gguf: &GgufFile) {
        let data = gguf.tensor_data(&w.lm_head);
        self.lm_head_type = w.lm_head.ggml_type;
        unsafe {
            let p = cuda_alloc(data.len());
            if p.is_null() {
                eprintln!("GPU: lm_head too large ({} MB), using CPU fallback",
                    data.len() / (1024 * 1024));
            } else {
                cuda_h2d(p, data.as_ptr() as *const c_void, data.len());
                cuda_sync();
                eprintln!("GPU: lm_head uploaded ({} MB, type {})",
                    data.len() / (1024 * 1024), self.lm_head_type);
                self.d_lm_head = Some(GpuBuf { ptr: p, bytes: data.len() });
            }
        }
    }
}

// ── Expert slice on GPU ───────────────────────────────────────────────────────
// Returns a view into an already-uploaded 3D expert tensor.
// The pointer arithmetic happens on the host; the bytes are already in VRAM.

pub struct GpuSlice {
    pub ptr:       *const u8,
    pub ggml_type: u32,
}

unsafe impl Send for GpuSlice {}
unsafe impl Sync for GpuSlice {}

pub fn gpu_expert_slice(buf: &GpuBuf, ggml_type: u32,
                        expert: usize, n_elems_per_expert: usize) -> GpuSlice
{
    let bytes = crate::gguf::ggml_type_size(ggml_type, n_elems_per_expert);
    GpuSlice {
        ptr:       unsafe { (buf.ptr as *const u8).add(expert * bytes) },
        ggml_type,
    }
}

impl GpuScratch {
    pub fn gemv_slice(&self, x: &[f32], slice: &GpuSlice,
                      in_dim: usize, out_dim: usize) -> Vec<f32>
    {
        unsafe {
            cuda_h2d(self.d_x.ptr, x.as_ptr() as *const c_void, x.len() * 4);
            cuda_gemv(
                slice.ggml_type,
                slice.ptr,
                self.d_x.ptr as *const f32,
                self.d_y.ptr as *mut f32,
                in_dim as i32,
                out_dim as i32,
            );
            let mut out = vec![0.0f32; out_dim];
            cuda_d2h(out.as_mut_ptr() as *mut c_void,
                     self.d_y.ptr as *const c_void,
                     out_dim * 4);
            cuda_sync();
            out
        }
    }
}

// ── upload_f32 helper ─────────────────────────────────────────────────────────

impl GpuBuf {
    pub fn upload_f32(data: &[f32]) -> Self {
        let bytes = data.len() * 4;
        let ptr   = unsafe { cuda_alloc(bytes) };
        assert!(!ptr.is_null(), "cudaMalloc failed for {} f32s", data.len());
        unsafe { cuda_h2d(ptr, data.as_ptr() as *const c_void, bytes); }
        Self { ptr, bytes }
    }
}

// ── Full GPU forward state ─────────────────────────────────────────────────────
// Holds all GPU scratch buffers, pre-allocated KV cache, and norm/bias weights
// uploaded once at startup.  Hidden state stays in VRAM for the entire forward
// pass; only 2 small D2H transfers per token (MoE router + final lm_head input).

pub struct GpuForwardState {
    embed_dim:    usize,
    q_total_dim:  usize,
    kv_total_dim: usize,
    n_heads:      usize,
    n_kv_heads:   usize,
    head_dim:     usize,
    ffn_dim:      usize,
    n_layers:     usize,
    max_seq:      usize,
    // Scratch (reused every token)
    d_hidden:     GpuBuf,   // [embed_dim]
    d_norm_out:   GpuBuf,   // [embed_dim]
    d_q:          GpuBuf,   // [q_total_dim]
    d_k:          GpuBuf,   // [kv_total_dim]
    d_v:          GpuBuf,   // [kv_total_dim]
    d_attn_out:   GpuBuf,   // [q_total_dim]
    d_scores:     GpuBuf,   // [n_heads × max_seq]
    d_gate:       GpuBuf,   // [ffn_dim] — scratch (non-MoE or fallback)
    d_up:         GpuBuf,   // [ffn_dim]
    d_expert_out: GpuBuf,   // [ffn_dim] swiglu result (single expert, fallback)
    d_expert_acc: GpuBuf,   // [embed_dim] expert accumulator
    // Batched MoE buffers: all 4 experts in one allocation
    d_gate_all:       GpuBuf,   // [4 × ffn_dim] batched gate GEMV outputs
    d_up_all:         GpuBuf,   // [4 × ffn_dim] batched up GEMV outputs
    d_expert_out_all: GpuBuf,   // [4 × ffn_dim] batched swiglu outputs
    d_down_all:       GpuBuf,   // [4 × embed_dim] batched down GEMV outputs
    // KV cache — pre-allocated
    d_kv_k: Vec<GpuBuf>,   // [n_layers][max_seq × kv_total_dim]
    d_kv_v: Vec<GpuBuf>,
    // Norm weights (uploaded once)
    d_attn_norm:  Vec<GpuBuf>,
    d_ffn_norm:   Vec<GpuBuf>,
    d_q_norm:     Vec<Option<GpuBuf>>,   // Qwen3/OLMoE QK-norm (None if arch has none)
    d_k_norm:     Vec<Option<GpuBuf>>,
    d_final_norm: GpuBuf,
    // Attention biases (q/k/v always present; out optional)
    d_q_bias:   Vec<GpuBuf>,
    d_k_bias:   Vec<GpuBuf>,
    d_v_bias:   Vec<GpuBuf>,
    d_out_bias: Vec<Option<GpuBuf>>,
    // MoE FFN biases (optional per layer)
    d_moe_gate_bias: Vec<Option<GpuBuf>>,
    d_moe_up_bias:   Vec<Option<GpuBuf>>,
    d_moe_down_bias: Vec<Option<GpuBuf>>,
    // Per-layer attention sinks on GPU — [n_heads] floats (NEG_INF = no sink)
    d_attn_sinks: Vec<GpuBuf>,
    // MoE router weights on GPU — [n_experts × embed_dim] F32 per MoE layer
    d_moe_router_w: Vec<GpuBuf>,
    d_router_logits: GpuBuf,    // scratch [n_experts]
    n_experts: usize,
    // Scratch for GPU lm_head output — [vocab_size] floats
    d_logits:  GpuBuf,
    vocab_size: usize,
}

unsafe impl Send for GpuForwardState {}
unsafe impl Sync for GpuForwardState {}

impl GpuForwardState {
    pub fn new(w: &crate::Weights, cfg: &crate::Config) -> Self {
        eprint!("GPU: allocating forward state... ");
        let d    = cfg.embed_dim;
        let q_d  = cfg.q_total_dim();
        let kv_d = cfg.kv_total_dim();
        let fd   = cfg.ffn_dim;
        let nl   = cfg.n_layers;
        let mseq = cfg.yarn_orig_ctx.max(4096);

        let nh   = cfg.n_heads;
        // Cap KV cache at 2048 tokens so lm_head fits in VRAM on 16 GB GPUs.
        // yarn_orig_ctx can be 4096 but KV cache at 4096 = 384 MB leaving no room
        // for lm_head (1.1 GB BF16). At 2048 KV cache = 192 MB.
        let mseq = mseq.min(2048);
        let a = GpuBuf::alloc_f32;
        let d_hidden     = a(d);
        let d_norm_out   = a(d);
        let d_q          = a(q_d);
        let d_k          = a(kv_d);
        let d_v          = a(kv_d);
        let d_attn_out   = a(q_d);
        let d_scores     = a(nh * mseq);   // [n_heads × max_seq] for batched attention
        let d_gate       = a(fd);
        let d_up         = a(fd);
        let d_expert_out = a(fd);
        let d_expert_acc = a(d);
        let d_gate_all       = a(4 * fd);
        let d_up_all         = a(4 * fd);
        let d_expert_out_all = a(4 * fd);
        let d_down_all       = a(4 * d);
        let d_logits         = a(cfg.vocab_size);

        let mut d_kv_k = Vec::with_capacity(nl);
        let mut d_kv_v = Vec::with_capacity(nl);
        for _ in 0..nl {
            d_kv_k.push(a(mseq * kv_d));
            d_kv_v.push(a(mseq * kv_d));
        }

        let mut d_attn_norm = Vec::with_capacity(nl);
        let mut d_ffn_norm  = Vec::with_capacity(nl);
        for l in 0..nl {
            d_attn_norm.push(GpuBuf::upload_f32(&w.attn_norm[l]));
            d_ffn_norm .push(GpuBuf::upload_f32(&w.ffn_norm[l]));
        }
        let d_final_norm = GpuBuf::upload_f32(&w.final_norm);

        let mut d_q_norm = Vec::with_capacity(nl);
        let mut d_k_norm = Vec::with_capacity(nl);
        for l in 0..nl {
            d_q_norm.push(if w.q_norm[l].is_empty() { None } else { Some(GpuBuf::upload_f32(&w.q_norm[l])) });
            d_k_norm.push(if w.k_norm[l].is_empty() { None } else { Some(GpuBuf::upload_f32(&w.k_norm[l])) });
        }

        let mut d_q_bias   = Vec::with_capacity(nl);
        let mut d_k_bias   = Vec::with_capacity(nl);
        let mut d_v_bias   = Vec::with_capacity(nl);
        let mut d_out_bias = Vec::with_capacity(nl);
        for l in 0..nl {
            d_q_bias.push(GpuBuf::upload_f32(&w.q_bias[l]));
            d_k_bias.push(GpuBuf::upload_f32(&w.k_bias[l]));
            d_v_bias.push(GpuBuf::upload_f32(&w.v_bias[l]));
            d_out_bias.push(if w.out_bias[l].is_empty() { None }
                            else { Some(GpuBuf::upload_f32(&w.out_bias[l])) });
        }

        let (mut d_moe_gate_bias, mut d_moe_up_bias, mut d_moe_down_bias) =
            (Vec::new(), Vec::new(), Vec::new());
        for m in &w.moe {
            d_moe_gate_bias.push(if m.gate_bias.is_empty() { None }
                                 else { Some(GpuBuf::upload_f32(&m.gate_bias)) });
            d_moe_up_bias  .push(if m.up_bias.is_empty()   { None }
                                 else { Some(GpuBuf::upload_f32(&m.up_bias)) });
            d_moe_down_bias.push(if m.down_bias.is_empty() { None }
                                 else { Some(GpuBuf::upload_f32(&m.down_bias)) });
        }

        // Attn sinks — [n_heads] per layer; NEG_INF if model has none
        let no_sink = vec![f32::NEG_INFINITY; nh];
        let mut d_attn_sinks = Vec::with_capacity(nl);
        for l in 0..nl {
            d_attn_sinks.push(if w.attn_sinks[l].is_empty() {
                GpuBuf::upload_f32(&no_sink)
            } else {
                GpuBuf::upload_f32(&w.attn_sinks[l])
            });
        }

        // MoE router weights — [n_experts × embed_dim] F32 per MoE layer (tiny: ~9 MB total)
        let ne = cfg.n_experts;
        let mut d_moe_router_w = Vec::with_capacity(w.moe.len());
        for m in &w.moe {
            d_moe_router_w.push(GpuBuf::upload_f32(&m.router_w));
        }
        let d_router_logits = GpuBuf::alloc_f32(ne.max(1));

        unsafe { cuda_sync(); }
        eprintln!("done.");

        Self {
            embed_dim: d, q_total_dim: q_d, kv_total_dim: kv_d,
            n_heads: nh, n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim, ffn_dim: fd, n_layers: nl, max_seq: mseq,
            d_hidden, d_norm_out, d_q, d_k, d_v, d_attn_out,
            d_scores, d_gate, d_up, d_expert_out, d_expert_acc,
            d_gate_all, d_up_all, d_expert_out_all, d_down_all,
            d_kv_k, d_kv_v,
            d_attn_norm, d_ffn_norm, d_q_norm, d_k_norm, d_final_norm,
            d_q_bias, d_k_bias, d_v_bias, d_out_bias,
            d_moe_gate_bias, d_moe_up_bias, d_moe_down_bias,
            d_attn_sinks,
            d_moe_router_w, d_router_logits, n_experts: ne,
            d_logits, vocab_size: cfg.vocab_size,
        }
    }

    unsafe fn download_norm_out(&self) -> Vec<f32> {
        let n = self.embed_dim;
        let mut out = vec![0.0f32; n];
        unsafe {
            cuda_d2h(out.as_mut_ptr() as *mut c_void,
                     self.d_norm_out.ptr as *const c_void, n * 4);
            cuda_sync();
        }
        out
    }

    /// Full transformer forward pass — d_hidden stays in VRAM throughout.
    /// Pass need_logits=false for prefill body tokens (skips final norm + lm_head).
    pub fn forward_gpu_step(
        &self,
        token:       usize,
        pos:         usize,
        need_logits: bool,
        cfg:         &crate::Config,
        w:           &crate::Weights,
        gw:          &GpuWeights,
        gguf:        &GgufFile,
    ) -> Vec<f32> {
        if pos >= self.max_seq {
            eprintln!("KV cache full at pos {} (max_seq={}); clamping to avoid OOB", pos, self.max_seq);
        }
        let pos = pos.min(self.max_seq - 1);

        let d    = self.embed_dim;
        let h    = self.n_heads;
        let kv   = self.n_kv_heads;
        let hd   = self.head_dim;
        let q_d  = self.q_total_dim;
        let kv_d = self.kv_total_dim;
        let fd   = self.ffn_dim;
        let nl   = self.n_layers;
        let seq  = (pos + 1) as i32;

        let mscale2 = if cfg.yarn_scale > 1.0 && cfg.yarn_orig_ctx > 0 {
            let m = 1.0_f32 + 0.1 * cfg.yarn_scale.ln();
            m * m
        } else { 1.0_f32 };
        let attn_scale = (hd as f32).sqrt().recip() * mscale2;

        // CPU embed dequant (one row ~11 KB) → upload to d_hidden
        {
            let embed_data = gguf.tensor_data(&w.embed);
            let row_bytes  = ggml_type_size(w.embed.ggml_type, d);
            let row = &embed_data[token * row_bytes..(token + 1) * row_bytes];
            let xf  = crate::dequant::dequant(row, w.embed.ggml_type, d);
            unsafe { cuda_h2d(self.d_hidden.ptr, xf.as_ptr() as *const c_void, d * 4); }
        }

        unsafe {
            for l in 0..nl {
                // ── Attention ──────────────────────────────────────────────────

                cuda_rms_norm(self.d_norm_out.ptr as *mut f32,
                              self.d_hidden.ptr   as *const f32,
                              self.d_attn_norm[l].ptr as *const f32,
                              d as i32);

                cuda_gemv(gw.q_proj_type[l], gw.q_proj[l].ptr as *const u8,
                          self.d_norm_out.ptr as *const f32,
                          self.d_q.ptr as *mut f32, d as i32, q_d as i32);
                cuda_gemv(gw.k_proj_type[l], gw.k_proj[l].ptr as *const u8,
                          self.d_norm_out.ptr as *const f32,
                          self.d_k.ptr as *mut f32, d as i32, kv_d as i32);
                cuda_gemv(gw.v_proj_type[l], gw.v_proj[l].ptr as *const u8,
                          self.d_norm_out.ptr as *const f32,
                          self.d_v.ptr as *mut f32, d as i32, kv_d as i32);

                cuda_vec_add(self.d_q.ptr as *mut f32,
                             self.d_q_bias[l].ptr as *const f32, q_d as i32);
                cuda_vec_add(self.d_k.ptr as *mut f32,
                             self.d_k_bias[l].ptr as *const f32, kv_d as i32);
                cuda_vec_add(self.d_v.ptr as *mut f32,
                             self.d_v_bias[l].ptr as *const f32, kv_d as i32);

                // QK-norm (Qwen3 / OLMoE): RMSNorm Q and K BEFORE RoPE. Per-head when the weight
                // is head_dim-sized (Qwen3); over the full projection when it's q_dim-sized (OLMoE).
                // In-place is safe (the kernel finishes the reduction before writing).
                if let Some(qn) = &self.d_q_norm[l] {
                    if w.q_norm[l].len() == hd {
                        for hh in 0..h {
                            cuda_rms_norm((self.d_q.ptr as *mut f32).add(hh * hd),
                                          (self.d_q.ptr as *const f32).add(hh * hd),
                                          qn.ptr as *const f32, hd as i32);
                        }
                    } else {
                        cuda_rms_norm(self.d_q.ptr as *mut f32, self.d_q.ptr as *const f32,
                                      qn.ptr as *const f32, q_d as i32);
                    }
                }
                if let Some(kn) = &self.d_k_norm[l] {
                    if w.k_norm[l].len() == hd {
                        for hh in 0..kv {
                            cuda_rms_norm((self.d_k.ptr as *mut f32).add(hh * hd),
                                          (self.d_k.ptr as *const f32).add(hh * hd),
                                          kn.ptr as *const f32, hd as i32);
                        }
                    } else {
                        cuda_rms_norm(self.d_k.ptr as *mut f32, self.d_k.ptr as *const f32,
                                      kn.ptr as *const f32, kv_d as i32);
                    }
                }

                cuda_rope_yarn(self.d_q.ptr as *mut f32,
                               h as i32, hd as i32, pos as i32,
                               cfg.rope_theta, cfg.yarn_scale, cfg.yarn_orig_ctx as i32);
                cuda_rope_yarn(self.d_k.ptr as *mut f32,
                               kv as i32, hd as i32, pos as i32,
                               cfg.rope_theta, cfg.yarn_scale, cfg.yarn_orig_ctx as i32);

                cuda_kv_append(self.d_kv_k[l].ptr as *mut f32,
                               self.d_kv_v[l].ptr as *mut f32,
                               self.d_k.ptr as *const f32,
                               self.d_v.ptr as *const f32,
                               pos as i32, kv_d as i32);

                // Sliding-window attention (gpt-oss even layers): attend only to the
                // most recent `swa_window` keys. Implemented by sliding the K/V cache
                // base forward by `k_start` positions and shrinking the score length —
                // the per-head score stride (max_seq) is unchanged, so scores are
                // written at index 0..seq_a and the softmax/value kernels see the
                // windowed range. No kernel change needed.
                let k_start = cfg.attn_start(l, pos);
                let seq_a   = seq - k_start as i32;
                let k_cache = (self.d_kv_k[l].ptr as *const f32).add(k_start * kv_d);
                let v_cache = (self.d_kv_v[l].ptr as *const f32).add(k_start * kv_d);

                // All heads in 3 kernel calls instead of h×3=192
                cuda_attn_score_all(
                    self.d_scores.ptr as *mut f32,
                    self.d_q.ptr as *const f32,
                    k_cache,
                    h as i32, kv as i32, seq_a,
                    hd as i32, kv_d as i32, self.max_seq as i32, attn_scale,
                );
                cuda_softmax_sink_all(
                    self.d_scores.ptr as *mut f32,
                    h as i32, seq_a, self.max_seq as i32,
                    self.d_attn_sinks[l].ptr as *const f32,
                );
                cuda_attn_values_all(
                    self.d_attn_out.ptr as *mut f32,
                    self.d_scores.ptr as *const f32,
                    v_cache,
                    h as i32, kv as i32, seq_a,
                    hd as i32, kv_d as i32, self.max_seq as i32,
                );

                cuda_gemv(gw.out_proj_type[l], gw.out_proj[l].ptr as *const u8,
                          self.d_attn_out.ptr as *const f32,
                          self.d_norm_out.ptr as *mut f32,
                          q_d as i32, d as i32);

                if let Some(ob) = &self.d_out_bias[l] {
                    cuda_vec_add(self.d_norm_out.ptr as *mut f32,
                                 ob.ptr as *const f32, d as i32);
                }
                // Residual
                cuda_vec_add(self.d_hidden.ptr as *mut f32,
                             self.d_norm_out.ptr as *const f32, d as i32);

                // ── FFN (MoE) ──────────────────────────────────────────────────

                cuda_rms_norm(self.d_norm_out.ptr as *mut f32,
                              self.d_hidden.ptr   as *const f32,
                              self.d_ffn_norm[l].ptr as *const f32,
                              d as i32);

                if cfg.n_experts > 0 {
                    let moe = &w.moe[l];
                    let ne = self.n_experts;

                    // GPU router: GEMV(router_w, d_norm_out) → d_router_logits
                    // then D2H only 32 floats instead of the full 2880-float norm_out.
                    cuda_gemv(0, // F32
                              self.d_moe_router_w[l].ptr as *const u8,
                              self.d_norm_out.ptr as *const f32,
                              self.d_router_logits.ptr as *mut f32,
                              d as i32, ne as i32);
                    let mut logits = vec![0.0f32; ne];
                    cuda_d2h(logits.as_mut_ptr() as *mut c_void,
                             self.d_router_logits.ptr as *const c_void,
                             ne * 4);
                    cuda_sync();
                    if !moe.router_b.is_empty() {
                        logits.iter_mut().zip(&moe.router_b).for_each(|(v, b)| *v += b);
                    }
                    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate()
                        .map(|(i, &v)| (i, v)).collect();
                    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    let top   = &indexed[..cfg.n_experts_used];
                    let max_v = top.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = top.iter().map(|&(_, v)| (v - max_v).exp()).collect();
                    let sum_e: f32 = exps.iter().sum();

                    cuda_vec_zero(self.d_expert_acc.ptr as *mut f32, d as i32);

                    let n_act  = top.len(); // n_experts_used (4)
                    let n_io   = d * fd;
                    let n_down = fd * d;
                    let tg = gw.moe_gate_type[l];
                    let tu = gw.moe_up_type[l];
                    let td = gw.moe_down_type[l];

                    let xn = self.d_norm_out.ptr as *const f32;

                    if n_act == 4 {
                        // ── Fast path: top-4 (gpt-oss) — 4 experts in one batched kernel launch ──
                        let dummy_g = gw.moe_gate[l].ptr as *const u8;
                        let dummy_u = gw.moe_up[l].ptr   as *const u8;
                        let dummy_d = gw.moe_down[l].ptr  as *const u8;
                        let mut gptrs: [*const u8; 4] = [dummy_g; 4];
                        let mut uptrs: [*const u8; 4] = [dummy_u; 4];
                        let mut dptrs: [*const u8; 4] = [dummy_d; 4];
                        for (ki, &(eid, _)) in top.iter().enumerate() {
                            gptrs[ki] = (gw.moe_gate[l].ptr as *const u8).add(eid * ggml_type_size(tg, n_io));
                            uptrs[ki] = (gw.moe_up[l].ptr   as *const u8).add(eid * ggml_type_size(tu, n_io));
                            dptrs[ki] = (gw.moe_down[l].ptr as *const u8).add(eid * ggml_type_size(td, n_down));
                        }
                        let go  = self.d_gate_all.ptr as *mut f32;
                        let uo  = self.d_up_all.ptr   as *mut f32;
                        let eo  = self.d_expert_out_all.ptr as *mut f32;
                        let doo = self.d_down_all.ptr as *mut f32;
                        cuda_gemv_moe4(tg, gptrs[0], gptrs[1], gptrs[2], gptrs[3], xn, xn, xn, xn,
                            go, go.add(fd), go.add(2*fd), go.add(3*fd), n_act as i32, d as i32, fd as i32);
                        cuda_gemv_moe4(tu, uptrs[0], uptrs[1], uptrs[2], uptrs[3], xn, xn, xn, xn,
                            uo, uo.add(fd), uo.add(2*fd), uo.add(3*fd), n_act as i32, d as i32, fd as i32);
                        for (ki, &(eid, _)) in top.iter().enumerate() {
                            let ko = ki * fd;
                            if let Some(gb) = &self.d_moe_gate_bias[l] {
                                cuda_vec_add(go.add(ko), (gb.ptr as *const f32).add(eid * fd), fd as i32);
                            }
                            if let Some(ub) = &self.d_moe_up_bias[l] {
                                cuda_vec_add(uo.add(ko), (ub.ptr as *const f32).add(eid * fd), fd as i32);
                            }
                            // gpt-oss = clamped OAI SwiGLU; other MoE archs = standard SiLU SwiGLU.
                            if cfg.clamped_swiglu {
                                cuda_swiglu_oai(eo.add(ko), (go as *const f32).add(ko), (uo as *const f32).add(ko), fd as i32);
                            } else {
                                cuda_swiglu(eo.add(ko), (go as *const f32).add(ko), (uo as *const f32).add(ko), fd as i32);
                            }
                        }
                        cuda_gemv_moe4(td, dptrs[0], dptrs[1], dptrs[2], dptrs[3],
                            (eo as *const f32).add(0*fd), (eo as *const f32).add(1*fd),
                            (eo as *const f32).add(2*fd), (eo as *const f32).add(3*fd),
                            doo, doo.add(d), doo.add(2*d), doo.add(3*d), n_act as i32, fd as i32, d as i32);
                        for (ki, &(eid, _)) in top.iter().enumerate() {
                            let rw = exps[ki] / sum_e;
                            let ko = ki * d;
                            if let Some(db) = &self.d_moe_down_bias[l] {
                                cuda_vec_add(doo.add(ko), (db.ptr as *const f32).add(eid * d), d as i32);
                            }
                            cuda_vec_scale_add(self.d_expert_acc.ptr as *mut f32, (doo as *const f32).add(ko), rw, d as i32);
                        }
                    } else {
                        // ── General path: ANY active-expert count (Mixtral top-2, Qwen3/OLMoE top-8…) ──
                        // One expert at a time via single-expert GEMVs — correctness over the 4-wide
                        // throughput trick. Reuses the single-expert scratch buffers (no resize needed).
                        let g1 = self.d_gate.ptr as *mut f32;
                        let u1 = self.d_up.ptr as *mut f32;
                        let e1 = self.d_expert_out.ptr as *mut f32;
                        let dn = self.d_down_all.ptr as *mut f32; // scratch [d] (d_down_all is 4*d)
                        for (ki, &(eid, _)) in top.iter().enumerate() {
                            let rw = exps[ki] / sum_e;
                            let gp = (gw.moe_gate[l].ptr as *const u8).add(eid * ggml_type_size(tg, n_io));
                            cuda_gemv(tg, gp, xn, g1, d as i32, fd as i32);
                            if let Some(gb) = &self.d_moe_gate_bias[l] {
                                cuda_vec_add(g1, (gb.ptr as *const f32).add(eid * fd), fd as i32);
                            }
                            let up = (gw.moe_up[l].ptr as *const u8).add(eid * ggml_type_size(tu, n_io));
                            cuda_gemv(tu, up, xn, u1, d as i32, fd as i32);
                            if let Some(ub) = &self.d_moe_up_bias[l] {
                                cuda_vec_add(u1, (ub.ptr as *const f32).add(eid * fd), fd as i32);
                            }
                            if cfg.clamped_swiglu {
                                cuda_swiglu_oai(e1, g1 as *const f32, u1 as *const f32, fd as i32);
                            } else {
                                cuda_swiglu(e1, g1 as *const f32, u1 as *const f32, fd as i32);
                            }
                            let dp = (gw.moe_down[l].ptr as *const u8).add(eid * ggml_type_size(td, n_down));
                            cuda_gemv(td, dp, e1 as *const f32, dn, fd as i32, d as i32);
                            if let Some(db) = &self.d_moe_down_bias[l] {
                                cuda_vec_add(dn, (db.ptr as *const f32).add(eid * d), d as i32);
                            }
                            cuda_vec_scale_add(self.d_expert_acc.ptr as *mut f32, dn as *const f32, rw, d as i32);
                        }
                    }

                    // FFN residual
                    cuda_vec_add(self.d_hidden.ptr as *mut f32,
                                 self.d_expert_acc.ptr as *const f32, d as i32);
                } else {
                    if std::env::var_os("TINYQ4_DENSE_FFN_CPU").is_some() {
                        let norm_cpu = self.download_norm_out();
                        let ffn_out = crate::ffn_lazy(&norm_cpu, &w.gate[l], &w.up[l], &w.down[l], gguf, cfg);
                        cuda_h2d(self.d_norm_out.ptr, ffn_out.as_ptr() as *const c_void, d * 4);
                    } else {
                        // Dense FFN — gate/up/down stay resident on the GPU.
                        cuda_gemv(gw.gate_type[l], gw.gate[l].ptr as *const u8,
                                  self.d_norm_out.ptr as *const f32,
                                  self.d_gate.ptr as *mut f32, d as i32, fd as i32);
                        cuda_gemv(gw.up_type[l], gw.up[l].ptr as *const u8,
                                  self.d_norm_out.ptr as *const f32,
                                  self.d_up.ptr as *mut f32, d as i32, fd as i32);
                        cuda_swiglu(self.d_expert_out.ptr as *mut f32,
                                    self.d_gate.ptr as *const f32,
                                    self.d_up.ptr as *const f32,
                                    fd as i32);
                        cuda_gemv(gw.down_type[l], gw.down[l].ptr as *const u8,
                                  self.d_expert_out.ptr as *const f32,
                                  self.d_norm_out.ptr as *mut f32, fd as i32, d as i32);
                    }
                    cuda_vec_add(self.d_hidden.ptr as *mut f32,
                                 self.d_norm_out.ptr as *const f32, d as i32);
                }
            }

            if !need_logits { return Vec::new(); }

            cuda_rms_norm(self.d_norm_out.ptr as *mut f32,
                          self.d_hidden.ptr   as *const f32,
                          self.d_final_norm.ptr as *const f32,
                          d as i32);

            if let Some(ref lm) = gw.d_lm_head {
                // GPU lm_head — no D2H for norm_out; just GEMV + D2H of logits (~780 KB)
                cuda_gemv(gw.lm_head_type, lm.ptr as *const u8,
                          self.d_norm_out.ptr as *const f32,
                          self.d_logits.ptr as *mut f32,
                          d as i32, self.vocab_size as i32);
                let mut out = vec![0.0f32; self.vocab_size];
                cuda_d2h(out.as_mut_ptr() as *mut c_void,
                         self.d_logits.ptr as *const c_void,
                         self.vocab_size * 4);
                cuda_sync();
                out
            } else {
                // CPU fallback: D2H norm_out then parallel rayon gemv
                let norm_out = self.download_norm_out();
                crate::dequant::gemv(&norm_out, gguf.tensor_data(&w.lm_head),
                                     w.lm_head.ggml_type, d, cfg.vocab_size)
            }
        }
    }
}
