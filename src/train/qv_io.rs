//! Portable safetensors adapter and Adam state for final-layer Q/V LoRA.
//! The actual model checkpoint is separate and remains SHA256-identical.

use super::lora::LoraLinear;
use super::qv::AdapterPair;
use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub fn save(path: &Path, base_sha: &str, a: &AdapterPair, steps: usize) -> Result<()> {
    let layer = a.layer;
    let entries = [
        (
            format!("lora.{layer}.q.A"),
            &a.q.a,
            vec![a.q.rank, a.q.input],
        ),
        (
            format!("lora.{layer}.q.B"),
            &a.q.b,
            vec![a.q.output, a.q.rank],
        ),
        (
            format!("lora.{layer}.v.A"),
            &a.v.a,
            vec![a.v.rank, a.v.input],
        ),
        (
            format!("lora.{layer}.v.B"),
            &a.v.b,
            vec![a.v.output, a.v.rank],
        ),
    ];
    let mut header = Map::new();
    header.insert(
        "__metadata__".to_owned(),
        json!({
            "format":"quartz-qv-last-v0",
            "base_gguf_sha256":base_sha,
            "layer":layer.to_string(),
            "rank":a.q.rank.to_string(),
            "alpha":a.q.alpha.to_string(),
            "steps":steps.to_string(),
            "target":"last_layer_qv",
            "warning":"stage1: last-block single-target training, not last-K SFT"
        }),
    );
    let mut offset = 0_usize;
    for (name, weights, shape) in &entries {
        let end = offset
            .checked_add(weights.len() * 4)
            .context("adapter size overflow")?;
        header.insert(
            name.clone(),
            json!({"dtype":"F32","shape":shape,"data_offsets":[offset,end]}),
        );
        offset = end;
    }
    let raw = serde_json::to_vec(&header)?;
    if raw.len() > 1_000_000 {
        bail!("adapter header too large");
    }
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("safetensors.tmp");
    let mut file = File::create(&tmp)?;
    file.write_all(&(raw.len() as u64).to_le_bytes())?;
    file.write_all(&raw)?;
    for (_, weights, _) in &entries {
        for &w in weights.iter() {
            if !w.is_finite() {
                bail!("cannot save nonfinite adapter parameter");
            }
            file.write_all(&w.to_le_bytes())?;
        }
    }
    file.sync_all()?;
    fs::rename(&tmp, path)?;
    Ok(())
}
pub fn load(
    path: &Path,
    base_sha: &str,
    input: usize,
    qout: usize,
    vout: usize,
    layer: usize,
) -> Result<AdapterPair> {
    if input == 0 || input > 8192 || qout == 0 || qout > 8192 || vout == 0 || vout > 8192 {
        bail!("unsupported model dimensions");
    }
    let b = fs::read(path)?;
    if b.len() < 8 {
        bail!("short adapter");
    }
    let hlen = u64::from_le_bytes(b[..8].try_into()?) as usize;
    if hlen > 1_000_000 || hlen > b.len() - 8 {
        bail!("bad safetensors header");
    }
    let head: Value = serde_json::from_slice(&b[8..8 + hlen])?;
    let m = &head["__metadata__"];
    if m["format"].as_str() != Some("quartz-qv-last-v0") {
        bail!("unknown Q/V format");
    }
    if m["base_gguf_sha256"].as_str() != Some(base_sha) {
        bail!("wrong base GGUF SHA256");
    }
    if m["target"].as_str() != Some("last_layer_qv") {
        bail!("wrong adapter target");
    }
    let r: usize = m["rank"].as_str().context("rank missing")?.parse()?;
    let l: usize = m["layer"].as_str().context("layer missing")?.parse()?;
    let alpha: f32 = m["alpha"].as_str().context("alpha missing")?.parse()?;
    if r == 0 || r > 16 || l != layer || !alpha.is_finite() || alpha <= 0.0 {
        bail!("wrong adapter rank/layer/scale");
    }
    let mut pair = AdapterPair {
        layer,
        q: LoraLinear::new(input, qout, r, alpha, 1),
        v: LoraLinear::new(input, vout, r, alpha, 1),
    };
    let specs = [
        (format!("lora.{layer}.q.A"), &mut pair.q.a, vec![r, input]),
        (format!("lora.{layer}.q.B"), &mut pair.q.b, vec![qout, r]),
        (format!("lora.{layer}.v.A"), &mut pair.v.a, vec![r, input]),
        (format!("lora.{layer}.v.B"), &mut pair.v.b, vec![vout, r]),
    ];
    let mut expected = 0usize;
    for (name, dest, shape) in specs {
        let t = &head[&name];
        if t["dtype"].as_str() != Some("F32") {
            bail!("wrong tensor dtype: {name}");
        }
        let dims = t["shape"].as_array().context("missing shape")?;
        if dims.len() != 2
            || dims
                .iter()
                .zip(&shape)
                .any(|(v, s)| v.as_u64() != Some(*s as u64))
        {
            bail!("wrong tensor shape: {name}");
        }
        let offsets = t["data_offsets"].as_array().context("no data offsets")?;
        if offsets.len() != 2 {
            bail!("malformed data offsets");
        }
        let st = offsets[0].as_u64().context("bad start")? as usize;
        let en = offsets[1].as_u64().context("bad end")? as usize;
        if st != expected || en.checked_sub(st) != Some(dest.len() * 4) || en > b.len() - 8 - hlen {
            bail!("malformed adapter tensor range: {name}");
        }
        for (i, slot) in dest.iter_mut().enumerate() {
            let at = 8 + hlen + st + i * 4;
            *slot = f32::from_le_bytes(b[at..at + 4].try_into()?);
            if !slot.is_finite() {
                bail!("nonfinite adapter");
            }
        }
        expected = en;
    }
    if expected != b.len() - 8 - hlen {
        bail!("trailing or missing adapter data");
    }
    Ok(pair)
}

struct Moments {
    m: Vec<f32>,
    v: Vec<f32>,
}
impl Moments {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.; n],
            v: vec![0.; n],
        }
    }
    fn update(&mut self, w: &mut [f32], g: &[f32], t: i32, lr: f32, scale: f32) {
        let b1 = 0.9_f32;
        let b2 = 0.999_f32;
        let c1 = 1.0 - b1.powi(t);
        let c2 = 1.0 - b2.powi(t);
        for i in 0..w.len() {
            let grad = g[i] * scale;
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * grad;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * grad * grad;
            w[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        }
    }
}
pub struct Adam {
    t: i32,
    qa: Moments,
    qb: Moments,
    va: Moments,
    vb: Moments,
}
impl Adam {
    pub fn new(p: &AdapterPair) -> Self {
        Self {
            t: 0,
            qa: Moments::new(p.q.a.len()),
            qb: Moments::new(p.q.b.len()),
            va: Moments::new(p.v.a.len()),
            vb: Moments::new(p.v.b.len()),
        }
    }
    pub fn update(
        &mut self,
        p: &mut AdapterPair,
        g: &super::qv::AdapterPairGrads,
        train_n: usize,
        lr: f32,
    ) {
        assert!(train_n > 0);
        self.t += 1;
        let norm = g
            .components()
            .iter()
            .flat_map(|v| v.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            / train_n as f32;
        let clip = if norm > 1.0 { 1.0 / norm } else { 1.0 };
        let scale = clip / train_n as f32;
        self.qa.update(&mut p.q.a, &g.q.a, self.t, lr, scale);
        self.qb.update(&mut p.q.b, &g.q.b, self.t, lr, scale);
        self.va.update(&mut p.v.a, &g.v.a, self.t, lr, scale);
        self.vb.update(&mut p.v.b, &g.v.b, self.t, lr, scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qv_safetensors_roundtrip_and_frozen_identity_check() {
        let p = AdapterPair {
            layer: 2,
            q: LoraLinear::new(4, 4, 2, 4.0, 12),
            v: LoraLinear::new(4, 2, 2, 4.0, 23),
        };
        let path = std::env::temp_dir().join(format!("qv-lora-{}.safetensors", std::process::id()));
        save(&path, "frozen-gguf", &p, 17).unwrap();
        let b = load(&path, "frozen-gguf", 4, 4, 2, 2).unwrap();
        assert_eq!(b.q.a, p.q.a);
        assert_eq!(b.v.b, p.v.b);
        assert!(load(&path, "wrong-base", 4, 4, 2, 2).is_err());
        assert!(load(&path, "frozen-gguf", 4, 4, 2, 1).is_err());
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn adam_updates_both_q_and_v_adapters() {
        let mut p = AdapterPair {
            layer: 0,
            q: LoraLinear::new(4, 4, 2, 4.0, 11),
            v: LoraLinear::new(4, 2, 2, 4.0, 12),
        };
        let mut g = super::super::qv::AdapterPairGrads::zeros(&p);
        g.q.b.fill(0.5);
        g.v.b.fill(-0.7);
        let mut adam = Adam::new(&p);
        adam.update(&mut p, &g, 1, 0.01);
        assert!(p.q.b.iter().any(|w| *w != 0.0));
        assert!(p.v.b.iter().any(|w| *w != 0.0));
    }
}
