//! Experimental GGUF-backed head-only LoRA trainer.
//!
//! Stage 1 stepping-stone: run the *real frozen Quartz transformer* to produce
//! hidden states and base output-head scores, then learn only an independent
//! output-head LoRA. This is intentionally NOT transformer/backprop fine-tuning.
//! Each example has one target token; cached frozen scores are per example, so
//! there is no [seq_len x vocab] activation matrix.
//!
//! CLI: quartz train-head BASE.gguf DATA.jsonl ADAPTER.safetensors
//!         [--steps 80] [--rank 4] [--lr 0.01] [--max-tokens 16]

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use super::lora::{LoraGrads, LoraLinear};
use crate::GpuState;
use crate::gguf::GgufFile;
use crate::tokenizer::Tokenizer;
use crate::{Config, Weights, embed_and_layers, rms_norm};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Example {
    prompt: String,
    target: String,
    split: String,
}

struct Cached {
    x: Vec<f32>,
    base: Vec<f32>,
    target: usize,
    split: bool, // true=train; false=heldout
}

struct Adam {
    ma: Vec<f32>,
    va: Vec<f32>,
    mb: Vec<f32>,
    vb: Vec<f32>,
    step: usize,
}

impl Adam {
    fn new(l: &LoraLinear) -> Self {
        Self {
            ma: vec![0.; l.a.len()],
            va: vec![0.; l.a.len()],
            mb: vec![0.; l.b.len()],
            vb: vec![0.; l.b.len()],
            step: 0,
        }
    }
    fn update(&mut self, l: &mut LoraLinear, g: &LoraGrads, n: usize, lr: f32) {
        self.step += 1;
        let k = self.step as i32;
        let b1 = 0.9_f32;
        let b2 = 0.999_f32;
        let correction1 = 1.0 - b1.powi(k);
        let correction2 = 1.0 - b2.powi(k);
        let norm =
            (g.a.iter()
                .chain(g.b.iter())
                .map(|v| (v / n as f32).powi(2))
                .sum::<f32>())
            .sqrt();
        let clip = if norm > 1.0 { 1.0 / norm } else { 1.0 };
        fn one(
            weights: &mut [f32],
            grads: &[f32],
            m: &mut [f32],
            v: &mut [f32],
            denom: f32,
            lr: f32,
            b1: f32,
            b2: f32,
            c1: f32,
            c2: f32,
            clip: f32,
        ) {
            for i in 0..weights.len() {
                let g = grads[i] * clip / denom;
                m[i] = b1 * m[i] + (1. - b1) * g;
                v[i] = b2 * v[i] + (1. - b2) * g * g;
                weights[i] -= lr * (m[i] / c1) / ((v[i] / c2).sqrt() + 1.0e-8);
            }
        }
        one(
            &mut l.a,
            &g.a,
            &mut self.ma,
            &mut self.va,
            n as f32,
            lr,
            b1,
            b2,
            correction1,
            correction2,
            clip,
        );
        one(
            &mut l.b,
            &g.b,
            &mut self.mb,
            &mut self.vb,
            n as f32,
            lr,
            b1,
            b2,
            correction1,
            correction2,
            clip,
        );
    }
}

fn arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    if let Some(i) = args.iter().position(|s| s == key) {
        let value = args
            .get(i + 1)
            .with_context(|| format!("{key} requires a value"))?;
        return value.parse().map_err(|e| anyhow::anyhow!("{key}: {e}"));
    }
    Ok(default)
}

fn score(l: &LoraLinear, s: &Cached, grad: Option<&mut LoraGrads>) -> f32 {
    let mut logits = s.base.clone();
    l.forward_add(&s.x, &mut logits);
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits.iter().map(|v| ((v - m) as f64).exp()).sum::<f64>();
    let logz = m as f64 + sum.ln();
    let loss = (logz - logits[s.target] as f64) as f32;
    if let Some(g) = grad {
        for (j, v) in logits.iter_mut().enumerate() {
            let prob = ((*v as f64 - logz).exp()) as f32;
            *v = prob - if j == s.target { 1.0 } else { 0.0 };
        }
        // Adapter-only backward: W base is frozen and X already cached.
        let mut unused_dx = vec![0.0; l.input];
        l.backward_add(&s.x, &logits, &mut unused_dx, g);
    }
    loss
}

fn avg_loss(l: &LoraLinear, data: &[Cached], train: bool) -> Result<f32> {
    let mut n = 0;
    let mut sum = 0.0_f64;
    for s in data.iter().filter(|v| v.split == train) {
        let x = score(l, s, None);
        if !x.is_finite() {
            bail!("non-finite cross entropy");
        }
        sum += x as f64;
        n += 1;
    }
    if n == 0 {
        bail!("empty {} split", if train { "train" } else { "heldout" });
    }
    Ok((sum / n as f64) as f32)
}

fn sha256(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hash = Sha256::new();
    let mut chunk = [0_u8; 1024 * 256];
    loop {
        let n = f.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hash.update(&chunk[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn save(path: &Path, base_hash: &str, l: &LoraLinear, steps: usize) -> Result<()> {
    let ab = l.a.len() * 4;
    let bb = l.b.len() * 4;
    let header = serde_json::json!({
        "__metadata__": {
            "format":"quartz-lora-head-v0",
            "base_gguf_sha256":base_hash,
            "target":"lm_head",
            "rank":l.rank.to_string(),
            "alpha":l.alpha.to_string(),
            "steps":steps.to_string(),
            "warning":"head-only proof, not last-K transformer backprop"
        },
        "lora.lm_head.A":{"dtype":"F32","shape":[l.rank,l.input],"data_offsets":[0,ab]},
        "lora.lm_head.B":{"dtype":"F32","shape":[l.output,l.rank],"data_offsets":[ab,ab+bb]}
    });
    let h = serde_json::to_vec(&header)?;
    let tmp = path.with_extension("safetensors.tmp");
    if let Some(parent) = tmp.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = File::create(&tmp)?;
    f.write_all(&(h.len() as u64).to_le_bytes())?;
    f.write_all(&h)?;
    for v in l.a.iter().chain(l.b.iter()) {
        f.write_all(&v.to_le_bytes())?;
    }
    f.sync_all()?;
    fs::rename(&tmp, path)?;
    Ok(())
}

/// Simple strict loader for our own format. Rejects mismatched base model.
fn load(path: &Path, base_hash: &str, input: usize, output: usize) -> Result<LoraLinear> {
    let bytes = fs::read(path)?;
    if bytes.len() < 8 {
        bail!("short safetensors file");
    }
    let n = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    if n > 1_048_576 || bytes.len() < 8 + n {
        bail!("bad safetensors header");
    }
    let hdr: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n])?;
    let meta = &hdr["__metadata__"];
    if meta["format"].as_str() != Some("quartz-lora-head-v0") {
        bail!("wrong adapter format");
    }
    if meta["base_gguf_sha256"].as_str() != Some(base_hash) {
        bail!("base model hash mismatch");
    }
    let rank: usize = meta["rank"].as_str().context("no rank")?.parse()?;
    let alpha: f32 = meta["alpha"].as_str().context("no alpha")?.parse()?;
    if !(1..=128).contains(&rank) || input > 4096 || output > 1_000_000 {
        bail!("adapter dimensions rejected");
    }
    let mut l = LoraLinear::new(input, output, rank, alpha, 1);
    for (name, vec, shape) in [
        ("lora.lm_head.A", &mut l.a, vec![rank, input]),
        ("lora.lm_head.B", &mut l.b, vec![output, rank]),
    ] {
        let t = &hdr[name];
        let sh = t["shape"].as_array().context("missing shape")?;
        if sh.len() != 2
            || sh
                .iter()
                .zip(&shape)
                .any(|(x, y)| x.as_u64() != Some(*y as u64))
            || t["dtype"].as_str() != Some("F32")
        {
            bail!("adapter shape/dtype mismatch {name}");
        }
        let offsets = t["data_offsets"].as_array().context("missing offsets")?;
        if offsets.len() != 2 {
            bail!("missing data offsets");
        }
        let start = offsets[0].as_u64().context("invalid start")? as usize;
        let end = offsets[1].as_u64().context("invalid end")? as usize;
        if end.checked_sub(start) != Some(vec.len() * 4)
            || start > bytes.len() - (8 + n)
            || end > bytes.len() - (8 + n)
        {
            bail!("adapter data length mismatch {name}");
        }
        for (i, v) in vec.iter_mut().enumerate() {
            let at = 8 + n + start + i * 4;
            *v = f32::from_le_bytes(bytes[at..at + 4].try_into()?);
            if !v.is_finite() {
                bail!("nonfinite adapter weight");
            }
        }
    }
    Ok(l)
}

pub(crate) fn run(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        bail!(
            "usage: quartz train-head BASE.gguf DATA.jsonl ADAPTER.safetensors [--steps 80 --rank 4 --lr 0.01 --max-tokens 16]"
        );
    }
    let model_path = Path::new(&args[2]);
    let data_path = Path::new(&args[3]);
    let out_path = Path::new(&args[4]);
    let rank: usize = arg(args, "--rank", 4)?;
    let steps: usize = arg(args, "--steps", 80)?;
    let lr: f32 = arg(args, "--lr", 0.015)?;
    let max_tokens: usize = arg(args, "--max-tokens", 16)?;
    if !(1..=16).contains(&rank)
        || !(1..=500).contains(&steps)
        || !(0.000001..=0.1).contains(&lr)
        || !(1..=32).contains(&max_tokens)
    {
        bail!("unsafe training parameters");
    }
    if model_path == out_path || data_path == out_path {
        bail!("output must not overwrite input");
    }
    let raw = fs::read_to_string(data_path).context("cannot read training dataset")?;
    let mut rows: Vec<Example> = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Example = serde_json::from_str(line)
            .with_context(|| format!("bad dataset JSON line {}", idx + 1))?;
        if row.prompt.len() > 2048
            || row.target.len() > 128
            || row.target.is_empty()
            || !matches!(row.split.as_str(), "train" | "heldout")
        {
            bail!("invalid train/heldout example at line {}", idx + 1);
        }
        rows.push(row);
    }
    if rows.len() < 3 || rows.len() > 24 {
        bail!("need 3–24 examples total");
    }
    if rows.iter().filter(|x| x.split == "train").count() < 2
        || rows.iter().filter(|x| x.split == "heldout").count() < 1
    {
        bail!("need >=2 train and >=1 heldout examples");
    }

    println!("TRAIN_HEAD: opening GGUF {}", model_path.display());
    let gguf = Arc::new(GgufFile::open(model_path)?);
    let arch = gguf.architecture().unwrap_or("unknown");
    let cfg = Config::from_gguf(&gguf);
    if cfg.n_experts != 0 {
        bail!("v0 requires a dense model");
    }
    if cfg.embed_dim > 4096 || cfg.vocab_size > 1_000_000 {
        bail!("v0 model too large");
    }
    println!(
        "GGUF_REAL_MODEL: arch={} layers={} hidden={} vocab={} training=head-only",
        arch, cfg.n_layers, cfg.embed_dim, cfg.vocab_size
    );
    let tokenizer = Tokenizer::from_gguf(&gguf).context("GGUF missing tokenizer")?;
    let mut parsed = Vec::new();
    for row in &rows {
        let tok = tokenizer.encode(&row.prompt);
        let target = tokenizer.encode(&row.target);
        if tok.is_empty() || tok.len() > max_tokens || target.len() != 1 {
            bail!(
                "prompt requires 1..={max_tokens} tokens; target must encode as EXACTLY one token; prompt={:?} target={:?} prompt_tokens={} target_tokens={:?}",
                row.prompt,
                row.target,
                tok.len(),
                target
            );
        }
        if target[0] as usize >= cfg.vocab_size {
            bail!("target out of vocab");
        }
        parsed.push((tok, target[0] as usize, row.split == "train"));
    }
    let base_hash = sha256(model_path)?;
    println!("BASE_SHA256={base_hash}");
    let w = Weights::load_gguf(gguf.clone(), &cfg);
    // Always CPU for the correctness-first milestone.
    let gpu = GpuState::new_cpu();
    let started = Instant::now();
    let mut cached = Vec::new();
    for (idx, (tokens, target, split)) in parsed.iter().enumerate() {
        let mut kv: Vec<(Vec<f32>, Vec<f32>)> = (0..cfg.n_layers)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();
        let mut state = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            state = embed_and_layers(token as usize, pos, &cfg, &w, &mut kv, &gpu);
        }
        let x = rms_norm(&state, &w.final_norm, cfg.norm_eps);
        let base = gpu.gemv_lm_head(&x, &w.lm_head, &gguf, cfg.embed_dim, cfg.vocab_size);
        if base.len() != cfg.vocab_size || base.iter().any(|x| !x.is_finite()) {
            bail!("invalid GGUF forward logits");
        }
        cached.push(Cached {
            x,
            base,
            target: *target,
            split: *split,
        });
        println!(
            "CACHE: {}/{} split={} tokens={} elapsed_ms={}",
            idx + 1,
            rows.len(),
            if *split { "train" } else { "heldout" },
            tokens.len(),
            started.elapsed().as_millis()
        );
    }
    let mut adapter = LoraLinear::new(
        cfg.embed_dim,
        cfg.vocab_size,
        rank,
        rank as f32 * 2.,
        20261028,
    );
    let mut optimizer = Adam::new(&adapter);
    let train_count = cached.iter().filter(|x| x.split).count();
    let train0 = avg_loss(&adapter, &cached, true)?;
    let held0 = avg_loss(&adapter, &cached, false)?;
    println!("INITIAL: train={train0:.6} heldout={held0:.6} steps={steps} lr={lr}");
    for step in 1..=steps {
        let mut grads = adapter.zeros_like_grads();
        for s in cached.iter().filter(|x| x.split) {
            let loss = score(&adapter, s, Some(&mut grads));
            if !loss.is_finite() {
                bail!("nonfinite training loss at step {step}");
            }
        }
        optimizer.update(&mut adapter, &grads, train_count, lr);
        if step == 1 || step % 10 == 0 || step == steps {
            let tr = avg_loss(&adapter, &cached, true)?;
            let held = avg_loss(&adapter, &cached, false)?;
            println!(
                "STEP: {step} train={tr:.6} heldout={held:.6} elapsed_ms={}",
                started.elapsed().as_millis()
            );
        }
    }
    let train1 = avg_loss(&adapter, &cached, true)?;
    let held1 = avg_loss(&adapter, &cached, false)?;
    save(out_path, &base_hash, &adapter, steps)?;
    let loaded = load(out_path, &base_hash, cfg.embed_dim, cfg.vocab_size)?;
    let held_reload = avg_loss(&loaded, &cached, false)?;
    if (held_reload - held1).abs() > 1e-6 {
        bail!("adapter save/load parity failure");
    }
    let after = sha256(model_path)?;
    if after != base_hash {
        bail!("FROZEN BASE MODIFIED — SHA256 MISMATCH");
    }
    println!(
        "RESULT: {} base_unchanged=true saved_reload_parity=true train_loss={train0:.6}->{train1:.6} heldout_loss={held0:.6}->{held1:.6} elapsed_ms={} adapter={} base_sha256={}",
        if train1 < train0 && held1 < held0 {
            "PASS"
        } else {
            "NO_HELDOUT_GAIN"
        },
        started.elapsed().as_millis(),
        out_path.display(),
        base_hash
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_head_lora_gradient_matches_numeric() {
        let d = 4;
        let vocab = 5;
        let mut l = LoraLinear::new(d, vocab, 2, 4., 123);
        l.a = vec![0.2, -0.1, 0.3, 0.4, 0.1, 0.2, -0.2, 0.5];
        l.b = vec![0.2, -0.2, 0.3, 0.1, 0.2, 0.1, 0.3, 0.1, 0.1, -0.2];
        let c = Cached {
            x: vec![0.2, 0.9, -0.3, 0.8],
            base: vec![0.1, 0.7, -0.2, 0.3, 0.8],
            target: 3,
            split: true,
        };
        let mut g = l.zeros_like_grads();
        let _ = score(&l, &c, Some(&mut g));
        let eps = 1e-3;
        for i in 0..l.a.len() {
            l.a[i] += eps;
            let a = score(&l, &c, None);
            l.a[i] -= 2. * eps;
            let b = score(&l, &c, None);
            l.a[i] += eps;
            assert!(((a - b) / (2. * eps) - g.a[i]).abs() < 2e-4, "A[{i}]");
        }
        for i in 0..l.b.len() {
            l.b[i] += eps;
            let a = score(&l, &c, None);
            l.b[i] -= 2. * eps;
            let b = score(&l, &c, None);
            l.b[i] += eps;
            assert!(((a - b) / (2. * eps) - g.b[i]).abs() < 2e-4, "B[{i}]");
        }
    }

    #[test]
    fn adapter_safetensors_roundtrip_and_wrong_base_rejection() {
        let mut l = LoraLinear::new(2, 3, 2, 4., 42);
        l.a = vec![0.2, -0.3, 0.4, 0.5];
        l.b = vec![0.7, 0.1, 0.2, -0.2, 0.1, 0.4];
        let path = std::env::temp_dir().join(format!(
            "quartz-head-test-{}.safetensors",
            std::process::id()
        ));
        save(&path, "abc", &l, 10).unwrap();
        let loaded = load(&path, "abc", 2, 3).unwrap();
        assert_eq!(loaded.a, l.a);
        assert_eq!(loaded.b, l.b);
        assert!(load(&path, "wrong", 2, 3).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn optimizer_reduces_real_softmax_loss_on_cached_features() {
        let s = Cached {
            x: vec![1., -0.5],
            base: vec![0.5, -0.1, 0.2],
            target: 2,
            split: true,
        };
        let mut l = LoraLinear::new(2, 3, 2, 4., 13);
        let mut o = Adam::new(&l);
        let initial = score(&l, &s, None);
        for _ in 0..80 {
            let mut g = l.zeros_like_grads();
            score(&l, &s, Some(&mut g));
            o.update(&mut l, &g, 1, 0.03);
        }
        assert!(score(&l, &s, None) < initial * 0.1);
    }
}

/// Load a saved head adapter in a NEW process and evaluate novel prompts.
/// Does not update any weights, and enforces exact GGUF SHA256 binding.
pub(crate) fn evaluate(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        bail!("usage: quartz eval-head BASE.gguf EVAL.jsonl ADAPTER.safetensors [--max-tokens 16]");
    }
    let base_path = Path::new(&args[2]);
    let data_path = Path::new(&args[3]);
    let lora_path = Path::new(&args[4]);
    let max_tokens: usize = arg(args, "--max-tokens", 16)?;
    if !(1..=32).contains(&max_tokens) {
        bail!("bad max-tokens");
    }
    let raw = fs::read_to_string(data_path)?;
    let mut rows = Vec::new();
    for line in raw.lines().filter(|x| !x.trim().is_empty()) {
        let row: Example = serde_json::from_str(line)?;
        if row.split != "heldout" || row.prompt.len() > 2048 || row.target.len() > 128 {
            bail!("eval-head accepts heldout-only examples");
        }
        rows.push(row);
    }
    if rows.is_empty() || rows.len() > 24 {
        bail!("eval data must have 1..24 examples");
    }
    let gguf = Arc::new(GgufFile::open(base_path)?);
    let cfg = Config::from_gguf(&gguf);
    if cfg.n_experts != 0 || cfg.vocab_size > 1_000_000 {
        bail!("unsupported eval model");
    }
    let tokenizer = Tokenizer::from_gguf(&gguf).context("missing tokenizer")?;
    let base_hash = sha256(base_path)?;
    let adapter = load(lora_path, &base_hash, cfg.embed_dim, cfg.vocab_size)?;
    let unadapted = LoraLinear::new(
        cfg.embed_dim,
        cfg.vocab_size,
        adapter.rank,
        adapter.alpha,
        1,
    );
    let weights = Weights::load_gguf(gguf.clone(), &cfg);
    let gpu = GpuState::new_cpu();
    let start = Instant::now();
    let mut before = 0.0_f64;
    let mut after = 0.0_f64;
    println!(
        "EVAL_HEAD: arch={} layers={} examples={} rank={} adapter={} base_sha256={}",
        gguf.architecture().unwrap_or("unknown"),
        cfg.n_layers,
        rows.len(),
        adapter.rank,
        lora_path.display(),
        base_hash
    );
    for (idx, row) in rows.iter().enumerate() {
        let tokens = tokenizer.encode(&row.prompt);
        let target = tokenizer.encode(&row.target);
        if tokens.is_empty() || tokens.len() > max_tokens || target.len() != 1 {
            bail!("eval prompt/target token constraint failed on {}", idx + 1);
        }
        let mut kv: Vec<(Vec<f32>, Vec<f32>)> = (0..cfg.n_layers)
            .map(|_| (Vec::new(), Vec::new()))
            .collect();
        let mut state = Vec::new();
        for (pos, &id) in tokens.iter().enumerate() {
            state = embed_and_layers(id as usize, pos, &cfg, &weights, &mut kv, &gpu);
        }
        let x = rms_norm(&state, &weights.final_norm, cfg.norm_eps);
        let base = gpu.gemv_lm_head(&x, &weights.lm_head, &gguf, cfg.embed_dim, cfg.vocab_size);
        if base.iter().any(|v| !v.is_finite()) {
            bail!("nonfinite model logits");
        }
        let s = Cached {
            x,
            base,
            target: target[0] as usize,
            split: false,
        };
        let b = score(&unadapted, &s, None);
        let a = score(&adapter, &s, None);
        before += b as f64;
        after += a as f64;
        println!(
            "EVAL_SAMPLE: {} base={:.6} adapter={:.6} tokens={} elapsed_ms={}",
            idx + 1,
            b,
            a,
            tokens.len(),
            start.elapsed().as_millis()
        );
    }
    before /= rows.len() as f64;
    after /= rows.len() as f64;
    println!(
        "EVAL_RESULT: {} unseen={} base_loss={:.6} adapter_loss={:.6} elapsed_ms={} base_unchanged=true",
        if after < before {
            "IMPROVED"
        } else {
            "NOT_IMPROVED"
        },
        rows.len(),
        before,
        after,
        start.elapsed().as_millis()
    );
    Ok(())
}
