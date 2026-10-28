//! Optional numerical end-to-end gradient check on a REAL frozen GGUF.
//! Uses nonzero A/B on both adapters and the production quantized forward
//! (including last FFN + lm_head), so omitting W^T*dY cannot pass.
use super::qv::{self, AdapterPair, AdapterPairGrads, Prepared};
use crate::gguf::GgufFile;
use crate::{Config, GpuState, Weights};
use anyhow::{Result, bail};
use std::sync::Arc;

fn get_param(p: &AdapterPair, which: usize, index: usize) -> f32 {
    match which {
        0 => p.q.a[index],
        1 => p.q.b[index],
        2 => p.v.a[index],
        3 => p.v.b[index],
        _ => panic!("bad parameter"),
    }
}
fn set_param(p: &mut AdapterPair, which: usize, index: usize, value: f32) {
    let s = match which {
        0 => &mut p.q.a,
        1 => &mut p.q.b,
        2 => &mut p.v.a,
        3 => &mut p.v.b,
        _ => panic!("bad parameter"),
    };
    s[index] = value;
}
pub fn check(
    sample: &Prepared,
    initial: &AdapterPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> Result<()> {
    let mut pair = initial.clone();
    for (i, v) in pair.q.a.iter_mut().enumerate() {
        *v = (((i * 7) % 13) as f32 - 6.0) * 0.012;
    }
    for (i, v) in pair.q.b.iter_mut().enumerate() {
        *v = (((i * 11) % 17) as f32 - 8.0) * 0.014;
    }
    for (i, v) in pair.v.a.iter_mut().enumerate() {
        *v = (((i * 5) % 11) as f32 - 5.0) * 0.018;
    }
    for (i, v) in pair.v.b.iter_mut().enumerate() {
        *v = (((i * 3) % 13) as f32 - 6.0) * 0.018;
    }
    let tape = qv::forward_last(sample, &pair, gguf, cfg, w, gpu);
    let (_, dlogits) = qv::token_ce(&tape.logits, sample.target);
    let mut g = AdapterPairGrads::zeros(&pair);
    qv::backward(sample, &pair, &tape, &dlogits, gguf, cfg, w, &mut g);
    let grads = [&g.q.a, &g.q.b, &g.v.a, &g.v.b];
    for (which, grads) in grads.iter().enumerate() {
        let (index, &analytic) = grads
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .ok_or_else(|| anyhow::anyhow!("empty gradient"))?;
        if analytic.abs() < 1e-6 || !analytic.is_finite() {
            bail!("Q/V numeric check: gradient group {which} was zero or invalid");
        }
        let original = get_param(&pair, which, index);
        let eps = 0.03_f32;
        set_param(&mut pair, which, index, original + eps);
        let hi = qv::loss(sample, &pair, gguf, cfg, w, gpu);
        set_param(&mut pair, which, index, original - eps);
        let lo = qv::loss(sample, &pair, gguf, cfg, w, gpu);
        set_param(&mut pair, which, index, original);
        let numeric = (hi - lo) / (2.0 * eps);
        let abs = (analytic - numeric).abs();
        let rel = abs / (numeric.abs() + analytic.abs() + 1e-6);
        println!(
            "QV_GRADCHECK: group={} index={} analytic={:.8} numeric={:.8} abs_err={:.8} rel_err={:.6}",
            ["q.A", "q.B", "v.A", "v.B"][which],
            index,
            analytic,
            numeric,
            abs,
            rel
        );
        if abs > 0.004 && rel > 0.1 {
            bail!("end-to-end numerical gradient mismatch for group {which}");
        }
    }
    println!("QV_GRADCHECK: PASS real-model four adapter tensors; nonzero A and B");
    Ok(())
}
