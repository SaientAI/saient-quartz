//! Minimal, read-only SafeTensors parser owned by Quartz.
//!
//! SafeTensors is a container format, not an inference runtime. Quartz parses the
//! small JSON header itself and memory-maps tensor payloads so mobile inference
//! never needs a second full copy of multi-gigabyte model weights.

use std::{collections::BTreeMap, fs::File, path::Path, sync::Arc};

use anyhow::{Context, Result, bail};
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;

const HEADER_LENGTH_BYTES: usize = 8;
const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    F16,
    F32,
    BF16,
    I64,
}

impl DType {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "F16" => Ok(Self::F16),
            "F32" => Ok(Self::F32),
            "BF16" => Ok(Self::BF16),
            "I64" => Ok(Self::I64),
            other => bail!("unsupported SafeTensors dtype {other}"),
        }
    }

    pub fn byte_width(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    pub dtype: DType,
    pub shape: &'a [usize],
    bytes: &'a [u8],
}

#[derive(Clone)]
pub(crate) struct MappedTensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    mapping: Arc<Mmap>,
    tensor_count: usize,
    start: usize,
    end: usize,
}

impl MappedTensor {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mapping[self.start..self.end]
    }

    pub fn mapping(&self) -> &Arc<Mmap> {
        &self.mapping
    }

    pub fn offset(&self) -> usize {
        self.start
    }

    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }
}

impl<'a> TensorView<'a> {
    pub fn len(self) -> usize {
        self.shape.iter().product()
    }

    pub(crate) fn bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Read one element without materializing the full mapped tensor. The file
    /// parser has already proven that shape, dtype, and payload length agree.
    #[inline(always)]
    pub fn value(self, index: usize) -> f32 {
        debug_assert!(index < self.len());
        match self.dtype {
            DType::F16 => {
                let offset = index * 2;
                crate::dequant::f16_to_f32(u16::from_le_bytes([
                    self.bytes[offset],
                    self.bytes[offset + 1],
                ]))
            }
            DType::BF16 => {
                let offset = index * 2;
                let bits = u16::from_le_bytes([self.bytes[offset], self.bytes[offset + 1]]);
                f32::from_bits((bits as u32) << 16)
            }
            DType::F32 => {
                let offset = index * 4;
                f32::from_le_bytes(
                    self.bytes[offset..offset + 4]
                        .try_into()
                        .expect("validated four-byte element"),
                )
            }
            DType::I64 => {
                let offset = index * 8;
                i64::from_le_bytes(
                    self.bytes[offset..offset + 8]
                        .try_into()
                        .expect("validated eight-byte element"),
                ) as f32
            }
        }
    }
}

impl TensorInfo {
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.end - self.start
    }
}

#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct SafeTensorFile {
    mmap: Arc<Mmap>,
    tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensorFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("cannot open SafeTensors file {}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("cannot map SafeTensors file {}", path.display()))?;
        let tensors = parse_header(&mmap)
            .with_context(|| format!("invalid SafeTensors file {}", path.display()))?;
        Ok(Self {
            mmap: Arc::new(mmap),
            tensors,
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn mapped_len(&self) -> usize {
        self.mmap.len()
    }

    /// Stable identity for this file's underlying mmap, matching the key the
    /// Vulkan runtime uses for its per-file weight arena. Lets callers scope
    /// staged weight loading to one specific model component.
    pub fn mapping_key(&self) -> usize {
        self.mmap.as_ptr() as usize
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;
        Ok(&self.mmap[info.start..info.end])
    }

    pub fn view(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;
        Ok(TensorView {
            dtype: info.dtype,
            shape: &info.shape,
            bytes: &self.mmap[info.start..info.end],
        })
    }

    pub(crate) fn mapped(&self, name: &str) -> Result<MappedTensor> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;
        Ok(MappedTensor {
            dtype: info.dtype,
            shape: info.shape.clone(),
            mapping: Arc::clone(&self.mmap),
            tensor_count: self.tensors.len(),
            start: info.start,
            end: info.end,
        })
    }

    pub fn load_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;
        let bytes = &self.mmap[info.start..info.end];
        let mut output = Vec::with_capacity(info.element_count());
        match info.dtype {
            DType::F16 => {
                for chunk in bytes.chunks_exact(2) {
                    output.push(crate::dequant::f16_to_f32(u16::from_le_bytes([
                        chunk[0], chunk[1],
                    ])));
                }
            }
            DType::BF16 => {
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    output.push(f32::from_bits((bits as u32) << 16));
                }
            }
            DType::F32 => {
                for chunk in bytes.chunks_exact(4) {
                    output.push(f32::from_le_bytes(
                        chunk.try_into().expect("four-byte chunk"),
                    ));
                }
            }
            DType::I64 => bail!("tensor {name} is I64, not a floating-point weight"),
        }
        Ok(output)
    }
}

fn parse_header(file: &[u8]) -> Result<BTreeMap<String, TensorInfo>> {
    if file.len() < HEADER_LENGTH_BYTES {
        bail!("file is shorter than the SafeTensors header prefix");
    }
    let header_len = u64::from_le_bytes(file[..8].try_into().expect("eight-byte prefix"));
    let header_len =
        usize::try_from(header_len).context("header length does not fit this platform")?;
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        bail!("header length {header_len} is outside the accepted range");
    }
    let data_start = HEADER_LENGTH_BYTES
        .checked_add(header_len)
        .context("header length overflow")?;
    if data_start > file.len() {
        bail!("header extends past the end of the file");
    }

    let raw: serde_json::Value =
        serde_json::from_slice(&file[8..data_start]).context("header is not valid JSON")?;
    let object = raw
        .as_object()
        .context("header root must be a JSON object")?;
    let payload_len = file.len() - data_start;
    let mut tensors = BTreeMap::new();
    let mut ranges: Vec<(usize, usize, &str)> = Vec::new();

    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        if name.is_empty() {
            bail!("tensor name cannot be empty");
        }
        let raw_info: RawTensorInfo = serde_json::from_value(value.clone())
            .with_context(|| format!("invalid metadata for tensor {name}"))?;
        let dtype = DType::parse(&raw_info.dtype)
            .with_context(|| format!("invalid dtype for tensor {name}"))?;
        if raw_info.shape.is_empty() {
            bail!("tensor {name} has an empty shape");
        }
        let elements = raw_info.shape.iter().try_fold(1usize, |total, &dim| {
            if dim == 0 {
                bail!("tensor {name} has a zero-sized dimension");
            }
            total
                .checked_mul(dim)
                .context("tensor element count overflow")
        })?;
        let expected_bytes = elements
            .checked_mul(dtype.byte_width())
            .context("tensor byte length overflow")?;
        let [start, end] = raw_info.data_offsets;
        if start > end || end > payload_len {
            bail!("tensor {name} has out-of-range data offsets [{start}, {end}]");
        }
        if end - start != expected_bytes {
            bail!(
                "tensor {name} declares {} bytes but its shape and dtype require {expected_bytes}",
                end - start
            );
        }
        ranges.push((start, end, name));
        tensors.insert(
            name.clone(),
            TensorInfo {
                dtype,
                shape: raw_info.shape,
                start: data_start + start,
                end: data_start + end,
            },
        );
    }

    if tensors.is_empty() {
        bail!("file contains no tensors");
    }
    ranges.sort_unstable_by_key(|(start, _, _)| *start);
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            bail!("tensor payloads {} and {} overlap", pair[0].2, pair[1].2);
        }
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn fixture(header: &str, payload: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn accepts_a_valid_header_and_ignores_metadata() {
        let header = r#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let bytes = fixture(header, &[0; 8]);
        let tensors = parse_header(&bytes).unwrap();
        let info = tensors.get("weight").unwrap();
        assert_eq!(info.dtype, DType::F32);
        assert_eq!(info.shape, vec![2]);
        assert_eq!(info.byte_len(), 8);
    }

    #[test]
    fn rejects_overlapping_payloads() {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        let error = parse_header(&fixture(header, &[0; 12]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("overlap"), "{error}");
    }

    #[test]
    fn rejects_shape_length_mismatch() {
        let header = r#"{"weight":{"dtype":"F16","shape":[4],"data_offsets":[0,4]}}"#;
        let error = parse_header(&fixture(header, &[0; 4]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("require 8"), "{error}");
    }

    #[test]
    fn memory_maps_and_decodes_f16_and_f32_weights() {
        let header = r#"{"half":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"single":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let mut payload = Vec::new();
        payload.extend_from_slice(&0x3c00u16.to_le_bytes());
        payload.extend_from_slice(&0xc000u16.to_le_bytes());
        payload.extend_from_slice(&3.5f32.to_le_bytes());
        let bytes = fixture(header, &payload);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("quartz-safetensors-{nonce}.safetensors"));
        fs::write(&path, bytes).unwrap();

        let file = SafeTensorFile::open(&path).unwrap();
        assert_eq!(
            file.tensor_names().collect::<Vec<_>>(),
            vec!["half", "single"]
        );
        assert_eq!(file.info("half").unwrap().shape, vec![2]);
        assert_eq!(file.bytes("single").unwrap().len(), 4);
        assert_eq!(file.view("half").unwrap().value(1), -2.0);
        assert_eq!(file.load_f32("half").unwrap(), vec![1.0, -2.0]);
        assert_eq!(file.load_f32("single").unwrap(), vec![3.5]);

        fs::remove_file(path).unwrap();
    }
}
