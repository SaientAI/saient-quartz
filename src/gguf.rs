use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x46554747;

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8), Int8(i8), Uint16(u16), Int16(i16),
    Uint32(u32), Int32(i32), Float32(f32), Bool(bool),
    String(String), Array(Vec<GgufValue>),
    Uint64(u64), Int64(i64), Float64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Int32(v)  => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(s) = self { Some(s) } else { None }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,    // dims[0] = fastest-varying (in_features for weights)
    pub ggml_type: u32,
    pub offset: u64,       // relative to data_offset
}

impl TensorInfo {
    pub fn n_elems(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }
}

pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    _mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Cannot open {:?}", path))?;
        let mmap = unsafe { Mmap::map(&file) }.context("mmap failed")?;
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let data = &mmap[..];
        let mut pos = 0usize;

        let magic = read_u32(data, &mut pos)?;
        if magic != GGUF_MAGIC {
            bail!("Not a GGUF file (magic 0x{:08X})", magic);
        }
        let version = read_u32(data, &mut pos)?;
        if !(1..=3).contains(&version) {
            bail!("Unsupported GGUF version {}", version);
        }

        let tensor_count = if version == 1 { read_u32(data, &mut pos)? as u64 }
                           else { read_u64(data, &mut pos)? };
        let kv_count     = if version == 1 { read_u32(data, &mut pos)? as u64 }
                           else { read_u64(data, &mut pos)? };

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = read_string(data, &mut pos)?;
            let val = read_value(data, &mut pos, version)?;
            metadata.insert(key, val);
        }

        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name   = read_string(data, &mut pos)?;
            let n_dims = read_u32(data, &mut pos)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(if version == 1 { read_u32(data, &mut pos)? as u64 }
                          else { read_u64(data, &mut pos)? });
            }
            let ggml_type = read_u32(data, &mut pos)?;
            let offset    = read_u64(data, &mut pos)?;
            tensors.push(TensorInfo { name, n_dims, dims, ggml_type, offset });
        }

        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u32()).unwrap_or(32) as usize;
        if pos % alignment != 0 {
            pos += alignment - (pos % alignment);
        }
        let data_offset = pos as u64;

        Ok(Self { version, metadata, tensors, data_offset, _mmap: mmap })
    }

    // Raw bytes for a tensor's data block.
    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let n = info.n_elems();
        let bytes = ggml_type_size(info.ggml_type, n);
        let start = (self.data_offset + info.offset) as usize;
        &self._mmap[start..start + bytes]
    }

    pub fn tensor_map(&self) -> HashMap<&str, &TensorInfo> {
        self.tensors.iter().map(|t| (t.name.as_str(), t)).collect()
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(|v| v.as_u64())
    }

    // Read a u32/u64 metadata value under the arch-prefixed key.
    // Try arch-prefixed key, then fall back to alternative suffixes.
    // Handles e.g. "head_count" (LLaMA) vs "attention.head_count" (Qwen2).
    pub fn arch_u32(&self, suffix: &str) -> Option<u32> {
        let arch = self.architecture().unwrap_or("llama");
        self.meta_u32(&format!("{}.{}", arch, suffix))
    }

    pub fn arch_u32_any(&self, suffixes: &[&str]) -> Option<u32> {
        for s in suffixes {
            if let Some(v) = self.arch_u32(s) { return Some(v); }
        }
        None
    }

    pub fn arch_f32(&self, suffix: &str) -> Option<f32> {
        let arch = self.architecture().unwrap_or("llama");
        let key = format!("{}.{}", arch, suffix);
        match self.metadata.get(&key)? {
            GgufValue::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn arch_f32_any(&self, suffixes: &[&str]) -> Option<f32> {
        for s in suffixes { if let Some(v) = self.arch_f32(s) { return Some(v); } }
        None
    }
}

// Byte size of a GGML tensor blob.
pub fn ggml_type_size(ggml_type: u32, n_elems: usize) -> usize {
    match ggml_type {
        0  => n_elems * 4,                   // F32
        1  => n_elems * 2,                   // F16
        6  => (n_elems / 32)  * 22,          // Q5_0: 32 per block, 2+4+16 bytes
        7  => (n_elems / 32)  * 24,          // Q5_1: 32 per block, 2+2+4+16 bytes
        8  => (n_elems / 32)  * 34,          // Q8_0: 32 per block, 2+32 bytes
        10 => (n_elems / 256) * 84,          // Q2_K: 256 per block, 16+64+2+2 bytes
        12 => (n_elems / 256) * 144,         // Q4_K: 256 per block, 4+12+128 bytes
        13 => (n_elems / 256) * 176,         // Q5_K: 256 per block
        14 => (n_elems / 256) * 210,         // Q6_K: 256 per block
        20 => (n_elems / 32) * 18,           // IQ4_NL: 32 per block, 2+16 bytes
        30 => n_elems * 2,                   // BF16: 2 bytes each
        16 => (n_elems / 256) * 66,          // IQ2_XXS: 256/block, 2 + 32*2 bytes
        17 => (n_elems / 256) * 74,          // IQ2_XS:  256/block, 2 + 32*2 + 8 bytes
        22 => (n_elems / 256) * 82,          // IQ2_S:   256/block, 2 + 64 + 8 + 8 bytes
        21 => (n_elems / 256) * 110,         // IQ3_S:   256/block, 2 + 64 + 8 + 32 + 4 bytes
        t  => panic!("unsupported ggml_type: {}", t),
    }
}

// ── binary readers ────────────────────────────────────────────────────────────

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8> {
    let v = *data.get(*pos).context("EOF reading u8")?;
    *pos += 1; Ok(v)
}
fn read_i8(data: &[u8], pos: &mut usize) -> Result<i8>   { Ok(read_u8(data, pos)? as i8) }
fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16> {
    let b = data.get(*pos..*pos+2).context("EOF u16")?;
    *pos += 2; Ok(u16::from_le_bytes(b.try_into().unwrap()))
}
fn read_i16(data: &[u8], pos: &mut usize) -> Result<i16> { Ok(read_u16(data, pos)? as i16) }
fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32> {
    let b = data.get(*pos..*pos+4).context("EOF u32")?;
    *pos += 4; Ok(u32::from_le_bytes(b.try_into().unwrap()))
}
fn read_i32(data: &[u8], pos: &mut usize) -> Result<i32> { Ok(read_u32(data, pos)? as i32) }
fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64> {
    let b = data.get(*pos..*pos+8).context("EOF u64")?;
    *pos += 8; Ok(u64::from_le_bytes(b.try_into().unwrap()))
}
fn read_i64(data: &[u8], pos: &mut usize) -> Result<i64> { Ok(read_u64(data, pos)? as i64) }
fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32> { Ok(f32::from_bits(read_u32(data, pos)?)) }
fn read_f64(data: &[u8], pos: &mut usize) -> Result<f64> { Ok(f64::from_bits(read_u64(data, pos)?)) }

fn read_string(data: &[u8], pos: &mut usize) -> Result<String> {
    let len = read_u64(data, pos)? as usize;
    let bytes = data.get(*pos..*pos+len)
        .with_context(|| format!("EOF reading string len={}", len))?;
    *pos += len;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn read_value(data: &[u8], pos: &mut usize, version: u32) -> Result<GgufValue> {
    match read_u32(data, pos)? {
        0  => Ok(GgufValue::Uint8(read_u8(data, pos)?)),
        1  => Ok(GgufValue::Int8(read_i8(data, pos)?)),
        2  => Ok(GgufValue::Uint16(read_u16(data, pos)?)),
        3  => Ok(GgufValue::Int16(read_i16(data, pos)?)),
        4  => Ok(GgufValue::Uint32(read_u32(data, pos)?)),
        5  => Ok(GgufValue::Int32(read_i32(data, pos)?)),
        6  => Ok(GgufValue::Float32(read_f32(data, pos)?)),
        7  => Ok(GgufValue::Bool(read_u8(data, pos)? != 0)),
        8  => Ok(GgufValue::String(read_string(data, pos)?)),
        9  => {
            let elem_type = read_u32(data, pos)?;
            let count = if version == 1 { read_u32(data, pos)? as u64 }
                        else { read_u64(data, pos)? };
            let mut arr = Vec::with_capacity(count.min(65536) as usize);
            for _ in 0..count { arr.push(read_value_typed(data, pos, elem_type)?); }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64(data, pos)?)),
        11 => Ok(GgufValue::Int64(read_i64(data, pos)?)),
        12 => Ok(GgufValue::Float64(read_f64(data, pos)?)),
        t  => bail!("Unknown GGUF value type: {}", t),
    }
}

fn read_value_typed(data: &[u8], pos: &mut usize, type_id: u32) -> Result<GgufValue> {
    match type_id {
        0  => Ok(GgufValue::Uint8(read_u8(data, pos)?)),
        1  => Ok(GgufValue::Int8(read_i8(data, pos)?)),
        2  => Ok(GgufValue::Uint16(read_u16(data, pos)?)),
        3  => Ok(GgufValue::Int16(read_i16(data, pos)?)),
        4  => Ok(GgufValue::Uint32(read_u32(data, pos)?)),
        5  => Ok(GgufValue::Int32(read_i32(data, pos)?)),
        6  => Ok(GgufValue::Float32(read_f32(data, pos)?)),
        7  => Ok(GgufValue::Bool(read_u8(data, pos)? != 0)),
        8  => Ok(GgufValue::String(read_string(data, pos)?)),
        10 => Ok(GgufValue::Uint64(read_u64(data, pos)?)),
        11 => Ok(GgufValue::Int64(read_i64(data, pos)?)),
        12 => Ok(GgufValue::Float64(read_f64(data, pos)?)),
        t  => bail!("Unknown array elem type: {}", t),
    }
}
