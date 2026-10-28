//! Safetensors for multiple internal Q/V LoRA adapters, with strict
//! base-model SHA256, layer selection, rank and tensor-shape checks.

use super::lora::LoraLinear;
use super::multi_qv::MultiPair;
use super::qv::AdapterPair;
use crate::Config;
use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub fn save(path: &Path, base_sha: &str, model: &MultiPair, steps: usize) -> Result<()> {
    if model.blocks.is_empty() {
        bail!("empty adapter");
    }
    let mut hdr = Map::new();
    hdr.insert(
        "__metadata__".to_string(),
        json!({
            "format":"quartz-multi-qv-v0",
            "base_gguf_sha256":base_sha,
            "first_layer":model.first_layer.to_string(),
            "layer_count":model.blocks.len().to_string(),
            "rank":model.blocks[0].q.rank.to_string(),
            "alpha":model.blocks[0].q.alpha.to_string(),
            "steps":steps.to_string(),
            "target":"suffix_causal_qv",
            "training":"one_target_token_per_short_prompt"
        }),
    );
    let mut entries: Vec<(String, &[f32], Vec<usize>)> = Vec::new();
    for (index, p) in model.blocks.iter().enumerate() {
        if p.layer != model.first_layer + index {
            bail!("noncontiguous layers");
        }
        let l = p.layer;
        entries.push((format!("lora.{l}.q.A"), &p.q.a, vec![p.q.rank, p.q.input]));
        entries.push((format!("lora.{l}.q.B"), &p.q.b, vec![p.q.output, p.q.rank]));
        entries.push((format!("lora.{l}.v.A"), &p.v.a, vec![p.v.rank, p.v.input]));
        entries.push((format!("lora.{l}.v.B"), &p.v.b, vec![p.v.output, p.v.rank]));
    }
    let mut offset = 0usize;
    for (name, values, shape) in &entries {
        let next = offset
            .checked_add(values.len() * 4)
            .context("adapter size overflow")?;
        hdr.insert(
            name.clone(),
            json!({
                "dtype":"F32","shape":shape,"data_offsets":[offset,next]
            }),
        );
        offset = next;
    }
    let json = serde_json::to_vec(&hdr)?;
    if json.len() > 2_000_000 {
        bail!("adapter header too big");
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("safetensors.tmp");
    let mut f = File::create(&tmp)?;
    f.write_all(&(json.len() as u64).to_le_bytes())?;
    f.write_all(&json)?;
    for (_, weights, _) in &entries {
        for &v in *weights {
            if !v.is_finite() {
                bail!("non-finite LoRA checkpoint weight");
            }
            f.write_all(&v.to_le_bytes())?;
        }
    }
    f.sync_all()?;
    fs::rename(&tmp, path)?;
    Ok(())
}
pub fn load(path: &Path, hash: &str, cfg: &Config, expected_layers: usize) -> Result<MultiPair> {
    let b = fs::read(path)?;
    if b.len() < 8 {
        bail!("short safetensors");
    }
    let hlen = u64::from_le_bytes(b[..8].try_into()?) as usize;
    if hlen > 2_000_000 || hlen > b.len() - 8 {
        bail!("invalid adapter header length");
    }
    let header: Value = serde_json::from_slice(&b[8..8 + hlen])?;
    let meta = &header["__metadata__"];
    if meta["format"].as_str() != Some("quartz-multi-qv-v0")
        || meta["base_gguf_sha256"].as_str() != Some(hash)
        || meta["target"].as_str() != Some("suffix_causal_qv")
    {
        bail!("adapter model identity or format mismatch");
    }
    let count: usize = meta["layer_count"]
        .as_str()
        .context("no layer_count")?
        .parse()?;
    let first: usize = meta["first_layer"]
        .as_str()
        .context("no first_layer")?
        .parse()?;
    let rank: usize = meta["rank"].as_str().context("no rank")?.parse()?;
    let alpha: f32 = meta["alpha"].as_str().context("no alpha")?.parse()?;
    if count != expected_layers
        || count == 0
        || count > 8
        || first.checked_add(count) != Some(cfg.n_layers)
        || rank == 0
        || rank > 8
        || !alpha.is_finite()
        || alpha <= 0.0
    {
        bail!("adapter layers/rank incompatible with loaded GGUF");
    }
    let mut blocks = Vec::with_capacity(count);
    let mut offset = 0usize;
    for l in first..cfg.n_layers {
        let mut p = AdapterPair {
            layer: l,
            q: LoraLinear::new(cfg.embed_dim, cfg.q_total_dim(), rank, alpha, 1),
            v: LoraLinear::new(cfg.embed_dim, cfg.kv_total_dim(), rank, alpha, 2),
        };
        let mut entries = [
            (
                format!("lora.{l}.q.A"),
                &mut p.q.a,
                vec![rank, cfg.embed_dim],
            ),
            (
                format!("lora.{l}.q.B"),
                &mut p.q.b,
                vec![cfg.q_total_dim(), rank],
            ),
            (
                format!("lora.{l}.v.A"),
                &mut p.v.a,
                vec![rank, cfg.embed_dim],
            ),
            (
                format!("lora.{l}.v.B"),
                &mut p.v.b,
                vec![cfg.kv_total_dim(), rank],
            ),
        ];
        for (name, weights, shape) in &mut entries {
            let t = &header[name.as_str()];
            if t["dtype"].as_str() != Some("F32") {
                bail!("wrong dtype {name}");
            }
            let dims = t["shape"].as_array().context("missing tensor shape")?;
            if dims.len() != 2
                || dims
                    .iter()
                    .zip(shape.iter())
                    .any(|(a, b)| a.as_u64() != Some(*b as u64))
            {
                bail!("tensor dimensions mismatch {name}");
            }
            let off = t["data_offsets"].as_array().context("missing offsets")?;
            if off.len() != 2 {
                bail!("invalid tensor offsets");
            }
            let start = off[0].as_u64().context("bad start")? as usize;
            let end = off[1].as_u64().context("bad end")? as usize;
            if start != offset
                || end.checked_sub(start) != Some(weights.len() * 4)
                || end > b.len() - 8 - hlen
            {
                bail!("adapter tensor data invalid {name}");
            }
            for (i, v) in weights.iter_mut().enumerate() {
                let idx = 8 + hlen + start + 4 * i;
                *v = f32::from_le_bytes(b[idx..idx + 4].try_into()?);
                if !v.is_finite() {
                    bail!("nonfinite parameter");
                }
            }
            offset = end;
        }
        blocks.push(p);
    }
    if offset != b.len() - 8 - hlen {
        bail!("trailing unverified adapter data");
    }
    Ok(MultiPair {
        first_layer: first,
        blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multi_qv_save_reload_and_wrong_base_rejected() {
        let cfg = Config {
            vocab_size: 9,
            embed_dim: 4,
            n_heads: 2,
            n_kv_heads: 1,
            n_layers: 3,
            ffn_dim: 8,
            context_length: 16,
            norm_eps: 1e-5,
            rope_theta: 1e4,
            head_dim: 2,
            n_experts: 0,
            n_experts_used: 0,
            moe_renorm: false,
            yarn_scale: 1.0,
            yarn_orig_ctx: 0,
            clamped_swiglu: false,
            swa_window: 0,
            rope_neox: true,
        };
        let m = MultiPair::new(&cfg, 2, 2, 77).unwrap();
        let file = std::env::temp_dir().join(format!(
            "quartz-multi-test-{}.safetensors",
            std::process::id()
        ));
        save(&file, "frozen-base", &m, 1).unwrap();
        let v = load(&file, "frozen-base", &cfg, 2).unwrap();
        assert_eq!(v.blocks.len(), 2);
        assert_eq!(v.blocks[0].q.a, m.blocks[0].q.a);
        assert_eq!(v.blocks[1].v.b, m.blocks[1].v.b);
        assert!(load(&file, "different-model", &cfg, 2).is_err());
        assert!(load(&file, "frozen-base", &cfg, 1).is_err());
        fs::remove_file(file).unwrap();
    }
}
