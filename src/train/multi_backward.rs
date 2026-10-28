//! Exact reverse-mode over each token and attention edge in the suffix.
//! No multi-agent / shortcut gradient: every causal key/value path is
//! aggregated before the previous trainable block receives its dX.

use super::multi_ops::gguf_transpose_batch;
use super::multi_qv::{MultiPair, Prepared, Tape, scale};
use super::ops::{attention_last_bwd, qk_norm_bwd, rms_bwd, rope_bwd, silu_derivative};
use super::qv::AdapterPairGrads;
use crate::gguf::GgufFile;
use crate::{Config, Weights};
use std::sync::Arc;

fn invert_rope(v: &mut [f32], pos: usize, heads: usize, cfg: &Config) {
    for h in 0..heads {
        let r = h * cfg.head_dim..(h + 1) * cfg.head_dim;
        rope_bwd(
            &mut v[r],
            pos,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.yarn_scale,
            cfg.yarn_orig_ctx,
            cfg.rope_neox,
        );
    }
}
/// dloss/dlogits enters through the frozen lm_head. Return all
/// trainable suffix Q/V LoRA gradients, in ascending layer order.
/// Partial gradients into the frozen prefix are deliberately dropped,
/// because there are no trainable parameters in that prefix.
pub fn backward(
    sample: &Prepared,
    model: &MultiPair,
    tape: &Tape,
    dlogits: &[f32],
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    grads: &mut [AdapterPairGrads],
) {
    let n = sample.prefix.len();
    let d = cfg.embed_dim;
    let qd = cfg.q_total_dim();
    let kd = cfg.kv_total_dim();
    assert_eq!(tape.layers.len(), model.blocks.len());
    assert_eq!(grads.len(), model.blocks.len());
    assert_eq!(tape.logits.len(), dlogits.len());

    let g_head = gguf_transpose_batch(gguf, &w.lm_head, &[dlogits.to_vec()], d, cfg.vocab_size);
    let mut g_next = vec![vec![0.0_f32; d]; n];
    let last = &tape.layers.last().expect("nonempty suffix").tokens[n - 1].output;
    g_next[n - 1] = rms_bwd(last, &w.final_norm, &g_head[0], cfg.norm_eps);

    for bi in (0..model.blocks.len()).rev() {
        let layer = &tape.layers[bi];
        let l = layer.layer;
        let pair = &model.blocks[bi];
        let tokens = &layer.tokens;

        // Frozen down projection and full SwiGLU backward for all tokens.
        let g_h = gguf_transpose_batch(gguf, &w.down[l], &g_next, cfg.ffn_dim, d);
        let mut g_gate = vec![vec![0.0_f32; cfg.ffn_dim]; n];
        let mut g_up = vec![vec![0.0_f32; cfg.ffn_dim]; n];
        for t in 0..n {
            for j in 0..cfg.ffn_dim {
                let gate = tokens[t].gate[j];
                let act = gate / (1.0 + (-gate).exp());
                g_gate[t][j] = g_h[t][j] * tokens[t].up[j] * silu_derivative(gate);
                g_up[t][j] = g_h[t][j] * act;
            }
        }
        let g_gatenorm = gguf_transpose_batch(gguf, &w.gate[l], &g_gate, d, cfg.ffn_dim);
        let g_upnorm = gguf_transpose_batch(gguf, &w.up[l], &g_up, d, cfg.ffn_dim);
        let mut g_z = vec![vec![0.0_f32; d]; n];
        for t in 0..n {
            let g_n2: Vec<f32> = (0..d).map(|i| g_gatenorm[t][i] + g_upnorm[t][i]).collect();
            let g_rms = rms_bwd(&tokens[t].z, &w.ffn_norm[l], &g_n2, cfg.norm_eps);
            for i in 0..d {
                g_z[t][i] = g_next[t][i] + g_rms[i];
            }
        }
        let g_att = gguf_transpose_batch(gguf, &w.out_proj[l], &g_z, qd, d);

        // FULL causal reverse pass: earlier K/V receive gradients
        // from every query that attended to them, including later tokens.
        let keys: Vec<Vec<f32>> = tokens.iter().map(|x| x.k_rot.clone()).collect();
        let values: Vec<Vec<f32>> = tokens.iter().map(|x| x.value.clone()).collect();
        let mut g_qrot = vec![vec![0.0_f32; qd]; n];
        let mut g_krot = vec![vec![0.0_f32; kd]; n];
        let mut g_v = vec![vec![0.0_f32; kd]; n];
        for t in 0..n {
            let att = attention_last_bwd(
                &tokens[t].q_rot,
                &keys[..=t],
                &values[..=t],
                &tokens[t].attention,
                &g_att[t],
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.attn_start(l, t),
                scale(cfg),
            );
            for i in 0..qd {
                g_qrot[t][i] += att.dq[i];
            }
            for past in 0..=t {
                for i in 0..kd {
                    g_krot[past][i] += att.dk[past][i];
                    g_v[past][i] += att.dv[past][i];
                }
            }
        }
        let mut g_q = vec![vec![0.0_f32; qd]; n];
        let mut g_k = vec![vec![0.0_f32; kd]; n];
        let mut g_adapt_dx = vec![vec![0.0_f32; d]; n];
        for t in 0..n {
            invert_rope(&mut g_qrot[t], t, cfg.n_heads, cfg);
            invert_rope(&mut g_krot[t], t, cfg.n_kv_heads, cfg);
            g_q[t] = qk_norm_bwd(
                &tokens[t].q_pre,
                &w.q_norm[l],
                &g_qrot[t],
                cfg.n_heads,
                cfg.head_dim,
                cfg.norm_eps,
            );
            g_k[t] = qk_norm_bwd(
                &tokens[t].k_pre,
                &w.k_norm[l],
                &g_krot[t],
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.norm_eps,
            );
            pair.q
                .backward_add(&tokens[t].n1, &g_q[t], &mut g_adapt_dx[t], &mut grads[bi].q);
            pair.v
                .backward_add(&tokens[t].n1, &g_v[t], &mut g_adapt_dx[t], &mut grads[bi].v);
        }
        if bi == 0 {
            break;
        } // Inputs are from frozen prefix, no earlier adapter.

        // Crucial frozen matrix transpose paths. LoRA-only dX is insufficient:
        // the earlier layer depends on the frozen Q/K/V projection outputs too.
        let g_qbase = gguf_transpose_batch(gguf, &w.q_proj[l], &g_q, d, qd);
        let g_kbase = gguf_transpose_batch(gguf, &w.k_proj[l], &g_k, d, kd);
        let g_vbase = gguf_transpose_batch(gguf, &w.v_proj[l], &g_v, d, kd);
        let mut g_input = vec![vec![0.0_f32; d]; n];
        for t in 0..n {
            let g_norm: Vec<f32> = (0..d)
                .map(|i| g_qbase[t][i] + g_kbase[t][i] + g_vbase[t][i] + g_adapt_dx[t][i])
                .collect();
            let g_rms = rms_bwd(&tokens[t].input, &w.attn_norm[l], &g_norm, cfg.norm_eps);
            for i in 0..d {
                g_input[t][i] = g_z[t][i] + g_rms[i];
            }
        }
        g_next = g_input;
    }
}
