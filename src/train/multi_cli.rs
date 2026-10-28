//! CLI for real frozen GGUF training with 1..8 internal Q/V LoRA blocks.
//! Never mutates the base model. One target token, all causal prompt tokens.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use super::multi_backward;
use super::multi_io;
use super::multi_qv::{self, MultiPair, Prepared};
use super::qv::{AdapterPair, AdapterPairGrads};
use super::qv_io::Adam;
use crate::gguf::GgufFile;
use crate::tokenizer::Tokenizer;
use crate::{Config, GpuState, Weights};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    prompt: String,
    target: String,
    split: String,
}
struct Sample {
    ids: Vec<u32>,
    x: Prepared,
    train: bool,
}

fn option<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    if let Some(i) = args.iter().position(|v| v == key) {
        let val = args
            .get(i + 1)
            .with_context(|| format!("{key} requires value"))?;
        val.parse().map_err(|e| anyhow::anyhow!("{key}: {e}"))
    } else {
        Ok(default)
    }
}
fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut sha = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    Ok(format!("{:x}", sha.finalize()))
}
fn model(path: &Path) -> Result<(Arc<GgufFile>, Config, Weights, GpuState, Tokenizer)> {
    let gguf = Arc::new(GgufFile::open(path)?);
    let cfg = Config::from_gguf(&gguf);
    if cfg.n_layers < 2
        || cfg.n_experts != 0
        || cfg.embed_dim > 4096
        || cfg.ffn_dim > 16384
        || cfg.vocab_size > 262144
        || cfg.head_dim == 0
        || cfg.head_dim % 2 != 0
    {
        bail!("requires supported dense quantized transformer");
    }
    let tokenizer = Tokenizer::from_gguf(&gguf).context("missing GGUF tokenizer")?;
    let w = Weights::load_gguf(Arc::clone(&gguf), &cfg);
    let gpu = GpuState::new_cpu();
    Ok((gguf, cfg, w, gpu, tokenizer))
}
fn read_rows(
    path: &Path,
    tokenizer: &Tokenizer,
    max_tokens: usize,
    evaluation: bool,
) -> Result<Vec<(Vec<u32>, usize, bool)>> {
    let raw = fs::read_to_string(path)?;
    let mut rows = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Row =
            serde_json::from_str(line).with_context(|| format!("JSONL line {}", i + 1))?;
        if row.prompt.len() > 1024
            || row.target.is_empty()
            || row.target.len() > 128
            || !matches!(row.split.as_str(), "train" | "heldout")
            || (evaluation && row.split != "heldout")
        {
            bail!("invalid example line {}", i + 1);
        }
        let tokens = tokenizer.encode(&row.prompt);
        let labels = tokenizer.encode(&row.target);
        if tokens.is_empty() || tokens.len() > max_tokens || labels.len() != 1 {
            bail!(
                "sample {} must have <= {} input tokens and one label token; prompt_tokens={} label_tokens={:?}",
                i + 1,
                max_tokens,
                tokens.len(),
                labels
            );
        }
        rows.push((tokens, labels[0] as usize, row.split == "train"));
    }
    if rows.is_empty() || rows.len() > 6 {
        bail!("expected 1..6 controlled examples");
    }
    if !evaluation
        && (rows.len() < 3
            || rows.iter().filter(|x| x.2).count() < 2
            || rows.iter().filter(|x| !x.2).count() < 1)
    {
        bail!("train set requires >=2 training and >=1 held-out example");
    }
    Ok(rows)
}
fn prepare_data(
    raw: Vec<(Vec<u32>, usize, bool)>,
    first: usize,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
    start: Instant,
) -> Result<Vec<Sample>> {
    let total = raw.len();
    let mut samples = Vec::new();
    for (i, (ids, target, train)) in raw.into_iter().enumerate() {
        let x = multi_qv::prepare(&ids, target, first, gguf, cfg, w, gpu)?;
        println!(
            "MULTI_CACHE: {}/{} tokens={} split={} elapsed_ms={}",
            i + 1,
            total,
            ids.len(),
            if train { "train" } else { "heldout" },
            start.elapsed().as_millis()
        );
        samples.push(Sample { ids, x, train });
    }
    Ok(samples)
}
fn mean(
    samples: &[Sample],
    p: &MultiPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
    train: bool,
) -> Result<f32> {
    let mut loss = 0.0f64;
    let mut count = 0;
    for s in samples.iter().filter(|s| s.train == train) {
        let l = multi_qv::loss(&s.x, p, gguf, cfg, w, gpu);
        if !l.is_finite() {
            bail!("nonfinite loss");
        }
        loss += l as f64;
        count += 1;
    }
    if count == 0 {
        bail!("split empty");
    }
    Ok((loss / count as f64) as f32)
}
fn set_parameter(p: &mut AdapterPair, which: usize, index: usize, value: f32) {
    let weights: &mut Vec<f32> = match which {
        0 => &mut p.q.a,
        1 => &mut p.q.b,
        2 => &mut p.v.a,
        3 => &mut p.v.b,
        _ => panic!("bad group"),
    };
    weights[index] = value;
}
fn parameter(p: &AdapterPair, which: usize, index: usize) -> f32 {
    let weights: &[f32] = match which {
        0 => &p.q.a,
        1 => &p.q.b,
        2 => &p.v.a,
        3 => &p.v.b,
        _ => panic!("bad group"),
    };
    weights[index]
}

/// Real-model central differences for BOTH Q and V on the lowest and
/// highest trainable layers, with nonzero A and B. Detects omitted
/// interlayer and cross-token gradients; runs only on opt-in.
fn numeric_check(
    s: &Sample,
    base: &MultiPair,
    gguf: &Arc<GgufFile>,
    cfg: &Config,
    w: &Weights,
    gpu: &GpuState,
) -> Result<()> {
    let mut p = base.clone();
    for (bi, pair) in p.blocks.iter_mut().enumerate() {
        for (i, v) in pair.q.a.iter_mut().enumerate() {
            *v = (((i * 7 + bi) % 13) as f32 - 6.0) * 0.009;
        }
        for (i, v) in pair.q.b.iter_mut().enumerate() {
            *v = (((i * 11 + bi) % 17) as f32 - 8.0) * 0.008;
        }
        for (i, v) in pair.v.a.iter_mut().enumerate() {
            *v = (((i * 5 + bi) % 11) as f32 - 5.0) * 0.012;
        }
        for (i, v) in pair.v.b.iter_mut().enumerate() {
            *v = (((i * 3 + bi) % 13) as f32 - 6.0) * 0.012;
        }
    }
    let tape = multi_qv::forward(&s.x, &p, gguf, cfg, w, gpu);
    let (_, dl) = super::qv::token_ce(&tape.logits, s.x.target);
    let mut grads = p.grads();
    multi_backward::backward(&s.x, &p, &tape, &dl, gguf, cfg, w, &mut grads);
    let mut indices = vec![0];
    if p.blocks.len() > 1 {
        indices.push(p.blocks.len() - 1);
    }
    for bi in indices {
        let gs = [
            &grads[bi].q.a,
            &grads[bi].q.b,
            &grads[bi].v.a,
            &grads[bi].v.b,
        ];
        for (which, group) in gs.iter().enumerate() {
            let (index, &analytic) = group
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
                .context("empty numerical derivative group")?;
            if !analytic.is_finite() || analytic.abs() < 1e-7 {
                bail!(
                    "numerical derivative was zero for suffix layer {} group {which}",
                    p.blocks[bi].layer
                );
            }
            let original = parameter(&p.blocks[bi], which, index);
            let eps = 0.03_f32;
            set_parameter(&mut p.blocks[bi], which, index, original + eps);
            let high = multi_qv::loss(&s.x, &p, gguf, cfg, w, gpu);
            set_parameter(&mut p.blocks[bi], which, index, original - eps);
            let low = multi_qv::loss(&s.x, &p, gguf, cfg, w, gpu);
            set_parameter(&mut p.blocks[bi], which, index, original);
            let number = (high - low) / (2.0 * eps);
            let diff = (analytic - number).abs();
            let rel = diff / (analytic.abs() + number.abs() + 1e-6);
            println!(
                "MULTI_GRADCHECK: layer={} group={} index={} analytic={analytic:.8} numeric={number:.8} rel_err={rel:.7} abs_err={diff:.7}",
                p.blocks[bi].layer,
                ["Q.A", "Q.B", "V.A", "V.B"][which],
                index
            );
            if !number.is_finite() || (rel > 0.09 && diff > 0.005) {
                bail!(
                    "multi-layer model gradient check failed on layer {} tensor {}",
                    p.blocks[bi].layer,
                    which
                );
            }
        }
    }
    println!("MULTI_GRADCHECK: PASS first and last adapted layers");
    Ok(())
}
pub(crate) fn train(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        bail!(
            "usage: quartz train-qv-suffix BASE.gguf TRAIN.jsonl ADAPTER.safetensors --layers 2 --steps 1 --rank 2 --lr 0.01 --max-tokens 16 [--gradient-check]"
        );
    }
    let base_path = Path::new(&args[2]);
    let dataset = Path::new(&args[3]);
    let output = Path::new(&args[4]);
    let layers: usize = option(args, "--layers", 2)?;
    let steps: usize = option(args, "--steps", 1)?;
    let rank: usize = option(args, "--rank", 2)?;
    let lr: f32 = option(args, "--lr", 0.01)?;
    let max_tokens: usize = option(args, "--max-tokens", 16)?;
    if !(1..=8).contains(&layers)
        || !(1..=8).contains(&steps)
        || !(1..=4).contains(&rank)
        || !(0.000001..=0.03).contains(&lr)
        || !(1..=16).contains(&max_tokens)
    {
        bail!("unsafe or unsupported experimental hyperparameters");
    }
    if output == base_path || output == dataset || output.exists() {
        bail!("refusing overwrite of existing file");
    }
    let started = Instant::now();
    let before_hash = sha256_file(base_path)?;
    let (gguf, cfg, w, gpu, tokenizer) = model(base_path)?;
    if layers > cfg.n_layers {
        bail!("more layers than model");
    }
    let pair = MultiPair::new(&cfg, layers, rank, 20261028)?;
    println!(
        "MULTI_MODEL: arch={} layers={} adapted={} first={} rank={} params={} hidden={} vocab={} frozen_sha256={}",
        gguf.architecture().unwrap_or("unknown"),
        cfg.n_layers,
        layers,
        pair.first_layer,
        rank,
        pair.parameter_count(),
        cfg.embed_dim,
        cfg.vocab_size,
        before_hash
    );
    let input = read_rows(dataset, &tokenizer, max_tokens, false)?;
    let samples = prepare_data(input, pair.first_layer, &gguf, &cfg, &w, &gpu, started)?;
    let parity = multi_qv::parity(&samples[0].ids, &samples[0].x, &pair, &gguf, &cfg, &w, &gpu)?;
    println!(
        "MULTI_PARITY: PASS relative_max={parity:.9} elapsed_ms={}",
        started.elapsed().as_millis()
    );
    if args.iter().any(|s| s == "--gradient-check") {
        numeric_check(&samples[0], &pair, &gguf, &cfg, &w, &gpu)?;
    }
    let mut pair = pair;
    let train_0 = mean(&samples, &pair, &gguf, &cfg, &w, &gpu, true)?;
    let held_0 = mean(&samples, &pair, &gguf, &cfg, &w, &gpu, false)?;
    println!("MULTI_INITIAL: train={train_0:.7} heldout={held_0:.7}");
    let train_n = samples.iter().filter(|s| s.train).count();
    let mut opts: Vec<Adam> = pair.blocks.iter().map(Adam::new).collect();
    for step in 1..=steps {
        let mut grads: Vec<AdapterPairGrads> = pair.grads();
        let mut before = 0.0_f64;
        for s in samples.iter().filter(|s| s.train) {
            let tape = multi_qv::forward(&s.x, &pair, &gguf, &cfg, &w, &gpu);
            let (loss, dl) = super::qv::token_ce(&tape.logits, s.x.target);
            if !loss.is_finite() {
                bail!("loss became nonfinite");
            }
            before += loss as f64;
            multi_backward::backward(&s.x, &pair, &tape, &dl, &gguf, &cfg, &w, &mut grads);
        }
        for (i, g) in grads.iter().enumerate() {
            if g.components()
                .iter()
                .flat_map(|x| x.iter())
                .any(|v| !v.is_finite())
            {
                bail!("nonfinite gradient layer {}", pair.blocks[i].layer);
            }
        }
        for i in 0..pair.blocks.len() {
            opts[i].update(&mut pair.blocks[i], &grads[i], train_n, lr);
        }
        let train = mean(&samples, &pair, &gguf, &cfg, &w, &gpu, true)?;
        let held = mean(&samples, &pair, &gguf, &cfg, &w, &gpu, false)?;
        println!(
            "MULTI_STEP: step={step} before_train={:.7} train={train:.7} heldout={held:.7} elapsed_ms={}",
            before / train_n as f64,
            started.elapsed().as_millis()
        );
    }
    let train_end = mean(&samples, &pair, &gguf, &cfg, &w, &gpu, true)?;
    let held_end = mean(&samples, &pair, &gguf, &cfg, &w, &gpu, false)?;
    for p in &pair.blocks {
        let qnorm = p.q.b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let vnorm = p.v.b.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "MULTI_ADAPTER: layer={} q_B_norm={qnorm:.8} v_B_norm={vnorm:.8}",
            p.layer
        );
        if !qnorm.is_finite() || !vnorm.is_finite() || qnorm < 1e-8 || vnorm < 1e-8 {
            bail!("Q and V did not both update in layer {}", p.layer);
        }
    }
    multi_io::save(output, &before_hash, &pair, steps)?;
    let loaded = multi_io::load(output, &before_hash, &cfg, layers)?;
    let reload = mean(&samples, &loaded, &gguf, &cfg, &w, &gpu, false)?;
    if (reload - held_end).abs() > 1e-5 {
        bail!("saved multi-adapter reloading changed loss");
    }
    if sha256_file(base_path)? != before_hash {
        bail!("immutable GGUF changed!");
    }
    println!(
        "MULTI_RESULT: {} adapted_layers={} base_unchanged=true checkpoint_reload=true train_loss={train_0:.7}->{train_end:.7} heldout_loss={held_0:.7}->{held_end:.7} total_ms={} adapter={}",
        if train_end < train_0 && held_end < held_0 {
            "PASS"
        } else {
            "NO_HELDOUT_GAIN"
        },
        layers,
        started.elapsed().as_millis(),
        output.display()
    );
    Ok(())
}
pub(crate) fn eval(args: &[String]) -> Result<()> {
    if args.len() < 5 {
        bail!("usage: quartz eval-qv-suffix BASE.gguf UNSEEN.jsonl ADAPTER.safetensors --layers 8");
    }
    let base_path = Path::new(&args[2]);
    let dataset = Path::new(&args[3]);
    let adapter_path = Path::new(&args[4]);
    let layers: usize = option(args, "--layers", 8)?;
    let max_tokens: usize = option(args, "--max-tokens", 16)?;
    if !(1..=8).contains(&layers) || !(1..=16).contains(&max_tokens) {
        bail!("invalid evaluation settings");
    }
    let started = Instant::now();
    let hash = sha256_file(base_path)?;
    let (gguf, cfg, w, gpu, tokenizer) = model(base_path)?;
    let pair = multi_io::load(adapter_path, &hash, &cfg, layers)?;
    let baseline = MultiPair::new(&cfg, layers, pair.blocks[0].q.rank, 20261028)?;
    let rows = read_rows(dataset, &tokenizer, max_tokens, true)?;
    let samples = prepare_data(rows, pair.first_layer, &gguf, &cfg, &w, &gpu, started)?;
    let mut before = 0.0_f64;
    let mut after = 0.0_f64;
    for (i, s) in samples.iter().enumerate() {
        let original = multi_qv::loss(&s.x, &baseline, &gguf, &cfg, &w, &gpu);
        let adapted = multi_qv::loss(&s.x, &pair, &gguf, &cfg, &w, &gpu);
        before += original as f64;
        after += adapted as f64;
        println!(
            "MULTI_EVAL_SAMPLE: {} base={original:.7} adapted={adapted:.7} elapsed_ms={}",
            i + 1,
            started.elapsed().as_millis()
        );
    }
    before /= samples.len() as f64;
    after /= samples.len() as f64;
    if sha256_file(base_path)? != hash {
        bail!("base GGUF changed during evaluation");
    }
    println!(
        "MULTI_EVAL_RESULT: {} unseen={} layers={} base={before:.7} adapted={after:.7} total_ms={} frozen=true",
        if after < before {
            "IMPROVED"
        } else {
            "NOT_IMPROVED"
        },
        samples.len(),
        layers,
        started.elapsed().as_millis()
    );
    Ok(())
}
