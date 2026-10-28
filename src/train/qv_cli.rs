//! CLI driver for last-layer Q/V LoRA over real frozen GGUF model activations.
//!
//! Usage:
//! quartz train-qv-last BASE.gguf DATA.jsonl OUT.safetensors
//!       --rank 2 --steps 12 --lr 0.01 --max-tokens 16
//! quartz eval-qv-last BASE.gguf UNSEEN.jsonl ADAPTER.safetensors
//!       --max-tokens 16

use super::qv::{self, AdapterPair, AdapterPairGrads, Prepared};
use super::qv_io::{self, Adam};
use crate::gguf::GgufFile;
use crate::tokenizer::Tokenizer;
use crate::{Config, GpuState, Weights};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetRow {
    prompt: String,
    target: String,
    split: String,
}
struct CachedRow {
    tokens: Vec<u32>,
    sample: Prepared,
    train: bool,
}
fn flag<T: std::str::FromStr>(args: &[String], k: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    if let Some(i) = args.iter().position(|v| v == k) {
        let val = args
            .get(i + 1)
            .with_context(|| format!("{k} requires a value"))?;
        val.parse().map_err(|e| anyhow::anyhow!("{k}: {e}"))
    } else {
        Ok(default)
    }
}
fn hash_file(path: &Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut sha = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    Ok(format!("{:x}", sha.finalize()))
}
fn read_rows(
    path: &Path,
    max_tokens: usize,
    tokenizer: &Tokenizer,
    only_heldout: bool,
) -> Result<Vec<(Vec<u32>, usize, bool)>> {
    let raw = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: DatasetRow =
            serde_json::from_str(line).with_context(|| format!("line {}", i + 1))?;
        if row.prompt.len() > 1024
            || row.target.is_empty()
            || row.target.len() > 64
            || !matches!(row.split.as_str(), "train" | "heldout")
            || (only_heldout && row.split != "heldout")
        {
            bail!(
                "invalid row {} (eval data must contain only heldout)",
                i + 1
            );
        }
        let toks = tokenizer.encode(&row.prompt);
        let target = tokenizer.encode(&row.target);
        if toks.is_empty() || toks.len() > max_tokens || target.len() != 1 {
            bail!(
                "row {}: prompt must have 1..={} tokens and target exactly one token; prompt={} target={:?}",
                i + 1,
                max_tokens,
                toks.len(),
                target
            );
        }
        rows.push((toks, target[0] as usize, row.split == "train"));
    }
    if rows.len() < 2 || rows.len() > 10 {
        bail!("use 2..10 small, auditable examples for this milestone");
    }
    if !only_heldout
        && (rows.iter().filter(|x| x.2).count() < 2 || rows.iter().filter(|x| !x.2).count() < 1)
    {
        bail!("train data requires >=2 training and >=1 held-out examples");
    }
    Ok(rows)
}
fn prep_rows(
    rows: Vec<(Vec<u32>, usize, bool)>,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
    start: Instant,
) -> Result<Vec<CachedRow>> {
    let total = rows.len();
    let mut out = Vec::with_capacity(total);
    for (index, (tokens, target, train)) in rows.into_iter().enumerate() {
        let p = qv::prepare(&tokens, target, gguf, cfg, w, gpu)?;
        println!(
            "QV_CACHE: {}/{} split={} tokens={} elapsed_ms={}",
            index + 1,
            total,
            if train { "train" } else { "heldout" },
            tokens.len(),
            start.elapsed().as_millis()
        );
        out.push(CachedRow {
            tokens,
            sample: p,
            train,
        });
    }
    Ok(out)
}
fn average(
    rows: &[CachedRow],
    pair: &AdapterPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
    train: bool,
) -> Result<f32> {
    let mut n = 0usize;
    let mut loss = 0.0_f64;
    for row in rows.iter().filter(|row| row.train == train) {
        let l = qv::loss(&row.sample, pair, gguf, cfg, w, gpu);
        if !l.is_finite() {
            bail!("non-finite model loss");
        }
        n += 1;
        loss += l as f64;
    }
    if n == 0 {
        bail!("no {} examples", if train { "train" } else { "heldout" });
    }
    Ok((loss / n as f64) as f32)
}
fn open_model(path: &Path) -> Result<(Arc<GgufFile>, Config, Weights, GpuState, Tokenizer)> {
    let gguf = Arc::new(GgufFile::open(path)?);
    let cfg = Config::from_gguf(&gguf);
    if cfg.n_layers == 0
        || cfg.n_experts != 0
        || cfg.embed_dim > 4096
        || cfg.vocab_size > 262144
        || cfg.ffn_dim > 16384
        || cfg.head_dim % 2 != 0
    {
        bail!("v0 Q/V trainer supports only dense models within device budgets");
    }
    let tokenizer = Tokenizer::from_gguf(&gguf).context("base GGUF missing tokenizer")?;
    let w = Weights::load_gguf(Arc::clone(&gguf), &cfg);
    let gpu = GpuState::new_cpu();
    Ok((gguf, cfg, w, gpu, tokenizer))
}
pub(crate) fn train(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        bail!(
            "usage: quartz train-qv-last BASE.gguf DATA.jsonl OUT.safetensors [--steps 12 --rank 2 --lr 0.01 --max-tokens 16]"
        );
    }
    let base_path = Path::new(&args[2]);
    let data_path = Path::new(&args[3]);
    let adapter_path = Path::new(&args[4]);
    if adapter_path == base_path || adapter_path == data_path || adapter_path.exists() {
        bail!("adapter output must not overwrite any existing input/output");
    }
    let rank: usize = flag(args, "--rank", 2)?;
    let steps: usize = flag(args, "--steps", 12)?;
    let lr: f32 = flag(args, "--lr", 0.01)?;
    let max_tokens: usize = flag(args, "--max-tokens", 16)?;
    if !(1..=8).contains(&rank)
        || !(1..=40).contains(&steps)
        || !(0.000001..=0.05).contains(&lr)
        || !(1..=16).contains(&max_tokens)
    {
        bail!("invalid phone-safe training hyperparameters");
    }
    println!(
        "QV_LAST: model={} data={} rank={} steps={} lr={}",
        base_path.display(),
        data_path.display(),
        rank,
        steps,
        lr
    );
    let start = Instant::now();
    let hash_before = hash_file(base_path)?;
    let (gguf, cfg, w, gpu, tokenizer) = open_model(base_path)?;
    let layer = cfg.n_layers - 1;
    println!(
        "QV_MODEL: arch={} layer={} layers={} embed={} q={} kv={} vocab={} hash={}",
        gguf.architecture().unwrap_or("unknown"),
        layer,
        cfg.n_layers,
        cfg.embed_dim,
        cfg.q_total_dim(),
        cfg.kv_total_dim(),
        cfg.vocab_size,
        hash_before
    );
    let input = read_rows(data_path, max_tokens, &tokenizer, false)?;
    let rows = prep_rows(input, &gguf, &cfg, &w, &gpu, start)?;
    let mut adapter = AdapterPair::new(&cfg, layer, rank, 20261028)?;
    // Zero-B LoRA must exactly match standard Quartz production inference.
    let rel = qv::frozen_forward_parity(
        &rows[0].tokens,
        &rows[0].sample,
        &adapter,
        &gguf,
        &cfg,
        &w,
        &gpu,
    )?;
    println!(
        "QV_FROZEN_PARITY: PASS relative_max_logit_error={rel:.9} elapsed_ms={}",
        start.elapsed().as_millis()
    );

    let before_train = average(&rows, &adapter, &gguf, &cfg, &w, &gpu, true)?;
    let before_held = average(&rows, &adapter, &gguf, &cfg, &w, &gpu, false)?;
    if args.iter().any(|v| v == "--gradient-check") {
        super::qv_gradient::check(&rows[0].sample, &adapter, &gguf, &cfg, &w, &gpu)?;
    }
    println!("QV_INITIAL: train={before_train:.7} heldout={before_held:.7}");
    let train_n = rows.iter().filter(|row| row.train).count();
    let mut optimizer = Adam::new(&adapter);
    for step in 1..=steps {
        let mut grad = AdapterPairGrads::zeros(&adapter);
        let mut running_loss = 0.0_f64;
        for row in rows.iter().filter(|row| row.train) {
            let tape = qv::forward_last(&row.sample, &adapter, &gguf, &cfg, &w, &gpu);
            let (loss, dlogits) = qv::token_ce(&tape.logits, row.sample.target);
            if !loss.is_finite() {
                bail!("non-finite loss step={step}");
            }
            running_loss += loss as f64;
            qv::backward(
                &row.sample,
                &adapter,
                &tape,
                &dlogits,
                &gguf,
                &cfg,
                &w,
                &mut grad,
            );
        }
        if grad
            .components()
            .iter()
            .flat_map(|c| c.iter())
            .any(|x| !x.is_finite())
        {
            bail!("non-finite gradient at step {step}");
        }
        optimizer.update(&mut adapter, &grad, train_n, lr);
        if step == 1 || step == steps || step % 5 == 0 {
            let held = average(&rows, &adapter, &gguf, &cfg, &w, &gpu, false)?;
            let tr = average(&rows, &adapter, &gguf, &cfg, &w, &gpu, true)?;
            println!(
                "QV_STEP: step={step} train={tr:.7} heldout={held:.7} pre_update_train={:.7} elapsed_ms={}",
                running_loss / train_n as f64,
                start.elapsed().as_millis()
            );
        }
    }
    let final_train = average(&rows, &adapter, &gguf, &cfg, &w, &gpu, true)?;
    let final_held = average(&rows, &adapter, &gguf, &cfg, &w, &gpu, false)?;
    let qnorm = adapter.q.b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let vnorm = adapter.v.b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !qnorm.is_finite() || !vnorm.is_finite() || qnorm == 0. || vnorm == 0. {
        bail!("Q and V must BOTH receive nonzero parameter updates");
    }
    qv_io::save(adapter_path, &hash_before, &adapter, steps)?;
    let reloaded = qv_io::load(
        adapter_path,
        &hash_before,
        cfg.embed_dim,
        cfg.q_total_dim(),
        cfg.kv_total_dim(),
        layer,
    )?;
    let reload_held = average(&rows, &reloaded, &gguf, &cfg, &w, &gpu, false)?;
    if (reload_held - final_held).abs() > 1e-5 {
        bail!("saved adapter failed forward parity");
    }
    let hash_after = hash_file(base_path)?;
    if hash_after != hash_before {
        bail!("frozen GGUF hash CHANGED");
    }
    println!(
        "QV_RESULT: {} base_unchanged=true adapter_roundtrip=true q_B_norm={qnorm:.7} v_B_norm={vnorm:.7} train_loss={before_train:.7}->{final_train:.7} heldout_loss={before_held:.7}->{final_held:.7} elapsed_ms={} adapter={} base_sha256={}",
        if final_train < before_train && final_held < before_held {
            "PASS"
        } else {
            "NEEDS_GENERALISATION_CHECK"
        },
        start.elapsed().as_millis(),
        adapter_path.display(),
        hash_before
    );
    Ok(())
}
pub(crate) fn eval(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        bail!(
            "usage: quartz eval-qv-last BASE.gguf NEW.jsonl ADAPTER.safetensors [--max-tokens 16]"
        );
    }
    let base_path = Path::new(&args[2]);
    let data_path = Path::new(&args[3]);
    let adapter_path = Path::new(&args[4]);
    let max_tokens: usize = flag(args, "--max-tokens", 16)?;
    if !(1..=16).contains(&max_tokens) {
        bail!("invalid evaluation token budget");
    }
    let start = Instant::now();
    let hash = hash_file(base_path)?;
    let (gguf, cfg, w, gpu, tokenizer) = open_model(base_path)?;
    let layer = cfg.n_layers - 1;
    let pair = qv_io::load(
        adapter_path,
        &hash,
        cfg.embed_dim,
        cfg.q_total_dim(),
        cfg.kv_total_dim(),
        layer,
    )?;
    let baseline = AdapterPair::new(&cfg, layer, pair.q.rank, 1)?;
    let input = read_rows(data_path, max_tokens, &tokenizer, true)?;
    let rows = prep_rows(input, &gguf, &cfg, &w, &gpu, start)?;
    let mut b = 0.0;
    let mut a = 0.0;
    for (i, row) in rows.iter().enumerate() {
        let before = qv::loss(&row.sample, &baseline, &gguf, &cfg, &w, &gpu);
        let after = qv::loss(&row.sample, &pair, &gguf, &cfg, &w, &gpu);
        b += before as f64;
        a += after as f64;
        println!(
            "QV_EVAL_SAMPLE: {} base={before:.7} adapted={after:.7} elapsed_ms={}",
            i + 1,
            start.elapsed().as_millis()
        );
    }
    b /= rows.len() as f64;
    a /= rows.len() as f64;
    if hash_file(base_path)? != hash {
        bail!("GGUF changed during evaluation");
    }
    println!(
        "QV_EVAL_RESULT: {} unseen={} base_loss={b:.7} adapted_loss={a:.7} elapsed_ms={} base_unchanged=true",
        if a < b { "IMPROVED" } else { "NOT_IMPROVED" },
        rows.len(),
        start.elapsed().as_millis()
    );
    Ok(())
}
