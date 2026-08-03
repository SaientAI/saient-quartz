//! Quartz-owned Vulkan compute for the mobile diffusion path.
//!
//! The CPU tensor implementation remains the correctness oracle. This module
//! owns Vulkan setup, memory, descriptors, dispatch, and FP16 conversion; it is
//! not a wrapper around an inference runtime.

use std::{
    collections::HashMap,
    ffi::CStr,
    fmt,
    fs::File,
    io::Cursor,
    mem::size_of,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use anyhow::{Context, Result, anyhow, bail};
use ash::{Entry, vk};
use memmap2::MmapOptions;
use rayon::prelude::*;

use crate::{
    safetensors::{DType, MappedTensor, SafeTensorFile, TensorView},
    tensor::Tensor,
};

const GEMM_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp16_gemm.spv"));
const CONV2D_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp16_conv2d.spv"));
const ATTENTION_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp16_attention.spv"));
const IM2COL_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp16_im2col.spv"));
const GROUPNORM_SILU_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp16_groupnorm_silu.spv"));
const RESIDUAL_ADD_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fp16_residual_add.spv"));
const GEMM_HEADS_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp16_gemm_heads.spv"));
const MERGE_HEADS_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp16_merge_heads.spv"));
const GEGLU_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fp16_geglu.spv"));
const ELEMENTWISE_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_elementwise.spv"));
const CHANNEL_RMSNORM_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_channel_rmsnorm.spv"));
const RESIDENT_LINEAR_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_f16_linear.spv"));
const LAYERNORM_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_layernorm.spv"));
const RMSNORM_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_rmsnorm.spv"));
const ROPE_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_rope.spv"));
const F32_ATTENTION_SHADER: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_attention.spv"));
const PATCH_LAYOUT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_patch_layout.spv"));
const WAN_HEAD_MODULATE_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_wan_head_modulate.spv"));
const RESIDENT_CONV3D_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_f16_conv3d.spv"));
const RESIDENT_CONV2D_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_f16_conv2d.spv"));
const NCTHW_TEMPORAL_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_ncthw_temporal.spv"));
const VAE_SPATIAL_LAYOUT_SHADER: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/f32_vae_spatial_layout.spv"));
static SD_ACCELERATION: AtomicBool = AtomicBool::new(false);
static RUNTIME: OnceLock<Result<Mutex<VulkanRuntime>, String>> = OnceLock::new();
#[cfg(test)]
pub(crate) static PERSISTENCE_TEST_LOCK: Mutex<()> = Mutex::new(());
const REQUIRED_WORKGROUP_INVOCATIONS: u32 = 256;
const REQUIRED_SHARED_MEMORY: u32 = 4096 * 4 + 256 * 4;

/// Staged weight loading: when a model's mapping key (the mmap base pointer,
/// same identity used for `model_mappings`) has an entry here, that model's
/// weight arena is capped to `(budget_bytes, tensor_count_hint)` instead of
/// the whole model file, and `begin_weight_stage` evicts everything cached so
/// far so the next stage can reuse the same bounded arena. Keyed per-mapping
/// (not global) because a single process can hold several SafeTensorFiles
/// concurrently (e.g. SDXL's two text encoders + UNet + VAE) and only the one
/// actually staged with `begin_weight_stage` calls should pay the bounded-
/// arena cost; the others must keep their natural whole-file sizing or they
/// each redundantly allocate the staged budget's full capacity on top of
/// their own, which can multiply GPU memory use several times over. Empty by
/// default, which preserves the original whole-file, cache-forever behaviour
/// exactly for every mapping.
static STAGED_ARENA_BUDGET: Mutex<Option<HashMap<usize, (usize, usize)>>> = Mutex::new(None);

/// Opt one specific model (identified by `mapping_key`, from
/// `SafeTensorFile::mapping_key`) into staged weight loading: subsequent
/// Vulkan weight arenas created for that mapping are capped at
/// `budget_bytes` (plus per-tensor alignment padding sized by
/// `tensor_count_hint`) rather than sized to the whole mapped file. Other
/// mappings are unaffected. Callers must pair this with `begin_weight_stage`
/// calls at safe block boundaries and must size the budget to comfortably fit
/// the largest single stage's tensors, or later dispatches will fail with an
/// arena-too-small error.
pub fn enable_staged_weight_loading(
    mapping_key: usize,
    budget_bytes: usize,
    tensor_count_hint: usize,
) {
    if let Ok(mut guard) = STAGED_ARENA_BUDGET.lock() {
        guard
            .get_or_insert_with(HashMap::new)
            .insert(mapping_key, (budget_bytes, tensor_count_hint));
    }
}

/// Return every mapping to the default whole-file weight arena. Does not
/// affect arenas already created.
pub fn disable_staged_weight_loading() {
    if let Ok(mut guard) = STAGED_ARENA_BUDGET.lock() {
        *guard = None;
    }
}

pub fn staged_weight_loading_enabled() -> bool {
    STAGED_ARENA_BUDGET
        .lock()
        .map(|guard| guard.as_ref().is_some_and(|map| !map.is_empty()))
        .unwrap_or(false)
}

fn staged_arena_budget(mapping_key: usize) -> Option<(usize, usize)> {
    STAGED_ARENA_BUDGET.lock().ok().and_then(|guard| {
        guard
            .as_ref()
            .and_then(|map| map.get(&mapping_key).copied())
    })
}

/// Evict every weight cached in an uploaded (non-zero-copy) arena so the next
/// stage's tensors can reuse the same bounded buffer. A no-op unless staged
/// weight loading is enabled, so callers can invoke this unconditionally at
/// every block boundary without any cost or behaviour change in the default
/// (unstaged) path. Zero-copy `ImportedMapping`s are untouched: they bind the
/// mmap directly and hold no separate device-resident copy to evict.
pub fn begin_weight_stage() -> Result<()> {
    if !staged_weight_loading_enabled() {
        return Ok(());
    }
    let Some(runtime) = RUNTIME.get() else {
        return Ok(());
    };
    let runtime = runtime
        .as_ref()
        .map_err(|error| anyhow!("Vulkan initialization failed: {error}"))?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("Vulkan runtime lock was poisoned"))?;
    unsafe {
        runtime
            .device
            .queue_wait_idle(runtime.queue)
            .map_err(|error| anyhow!("queue wait before weight-stage reset failed: {error:?}"))?;
    }
    // Only reset mappings that actually opted into a bounded budget. Without this,
    // every begin_weight_stage() call (now several per UNet forward pass, once per
    // layer) would also evict the text encoders' and VAE's whole-file caches, which
    // hold no staging boundaries of their own and would just re-upload unchanged.
    let staged_keys: std::collections::HashSet<usize> = STAGED_ARENA_BUDGET
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|map| map.keys().copied().collect()))
        .unwrap_or_default();
    for (key, mapping) in runtime.model_mappings.iter_mut() {
        if staged_keys.contains(key) {
            mapping.reset();
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Default)]
struct KernelStats {
    calls: u64,
    dispatch_milliseconds: f64,
    wall_milliseconds: f64,
}

#[derive(Clone, Copy, Default)]
struct RuntimeStats {
    elementwise: KernelStats,
    norm: KernelStats,
    gemm: KernelStats,
    conv2d: KernelStats,
    attention: KernelStats,
    uploaded_bytes: u64,
    cached_weight_bytes: u64,
    peak_dispatch_bytes: u64,
    resident_weight_uploads: u64,
    resident_tensor_uploads: u64,
    resident_downloads: u64,
    resident_uploaded_bytes: u64,
    resident_downloaded_bytes: u64,
    resident_allocated_bytes: u64,
    peak_resident_allocated_bytes: u64,
    resident_device_local_bytes: u64,
    peak_resident_device_local_bytes: u64,
    resident_device_local_allocation_bytes: u64,
    peak_resident_device_local_allocation_bytes: u64,
}

#[derive(Clone, Copy)]
enum KernelKind {
    Elementwise,
    Norm,
    Gemm,
    Conv2d,
    Attention,
}

#[derive(Clone, Copy)]
enum DispatchInput<'a> {
    Upload(&'a [u8]),
    Mapped(&'a MappedTensor),
}

impl DispatchInput<'_> {
    fn len(self) -> usize {
        match self {
            Self::Upload(bytes) => bytes.len(),
            Self::Mapped(tensor) => tensor.bytes().len(),
        }
    }

    fn uploaded_len(self) -> usize {
        match self {
            Self::Upload(bytes) => bytes.len(),
            Self::Mapped(_) => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentElementType {
    F16,
    F32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ResidentClass {
    Activation,
    Weight,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BufferMemoryClass {
    HostVisible,
    DeviceLocal,
}

#[derive(Debug)]
struct ResidentLease {
    id: u64,
}

impl Drop for ResidentLease {
    fn drop(&mut self) {
        release_resident_buffer(self.id);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentTensor {
    lease: Arc<ResidentLease>,
    elements: usize,
    element_type: ResidentElementType,
}

impl ResidentTensor {
    fn id(&self) -> u64 {
        self.lease.id
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentLinearWeights {
    weight: ResidentTensor,
    bias: ResidentTensor,
    input_width: usize,
    output_width: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentConv3dWeights {
    weight: ResidentTensor,
    bias: ResidentTensor,
    input_channels: usize,
    output_channels: usize,
    kernel: [usize; 3],
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentConv2dWeights {
    weight: ResidentTensor,
    bias: ResidentTensor,
    input_channels: usize,
    output_channels: usize,
    kernel: [usize; 2],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PersistenceStats {
    pub resident_weight_uploads: u64,
    pub resident_tensor_uploads: u64,
    pub resident_downloads: u64,
    pub resident_uploaded_bytes: u64,
    pub resident_downloaded_bytes: u64,
    pub resident_allocated_bytes: u64,
    pub peak_resident_allocated_bytes: u64,
    pub resident_device_local_bytes: u64,
    pub peak_resident_device_local_bytes: u64,
    pub resident_device_local_allocation_bytes: u64,
    pub peak_resident_device_local_allocation_bytes: u64,
    pub scratch_buffer_bytes: u64,
    pub scratch_buffer_allocation_bytes: u64,
    pub cached_model_mappings: usize,
}

pub struct BenchmarkResult {
    device: String,
    rows: u32,
    outputs: u32,
    width: u32,
    dispatch_milliseconds: f64,
    wall_milliseconds: f64,
    max_error: f32,
}

#[derive(Clone, Debug)]
pub struct DeviceProfile {
    pub device_name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,
    pub api_version: u32,
    pub queue_family: u32,
    pub device_local_memory_bytes: u64,
    pub available_device_memory_bytes: u64,
    pub memory_budget_supported: bool,
    pub max_storage_buffer_bytes: u64,
    pub max_memory_allocation_bytes: u64,
    pub max_workgroup_invocations: u32,
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_count: [u32; 3],
    pub subgroup_size: u32,
    pub fp16_supported: bool,
    pub int8_supported: bool,
    pub integer_dot_product_supported: bool,
    pub cooperative_matrix_supported: bool,
    pub storage_buffer_alignment: u64,
    pub timestamp_supported: bool,
    pub timestamp_period_nanoseconds: f32,
    pub external_host_memory_supported: bool,
}

impl fmt::Display for DeviceProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Quartz Vulkan device: name={:?} vendor=0x{:04x} device=0x{:04x} driver={} api={}.{}.{} queue={} memory={} available={} budget_ext={} max_storage={} max_allocation={} workgroup_invocations={} workgroup_size={:?} workgroup_count={:?} subgroup={} fp16={} int8={} dot={} cooperative_matrix={} storage_alignment={} timestamps={} timestamp_period_ns={} external_host_memory={}",
            self.device_name,
            self.vendor_id,
            self.device_id,
            self.driver_version,
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            vk::api_version_patch(self.api_version),
            self.queue_family,
            self.device_local_memory_bytes,
            self.available_device_memory_bytes,
            self.memory_budget_supported,
            self.max_storage_buffer_bytes,
            self.max_memory_allocation_bytes,
            self.max_workgroup_invocations,
            self.max_workgroup_size,
            self.max_workgroup_count,
            self.subgroup_size,
            self.fp16_supported,
            self.int8_supported,
            self.integer_dot_product_supported,
            self.cooperative_matrix_supported,
            self.storage_buffer_alignment,
            self.timestamp_supported,
            self.timestamp_period_nanoseconds,
            self.external_host_memory_supported,
        )
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Vulkan FP16 GEMM: device={:?} shape={}x{}x{} dispatch_ms={:.3} wall_ms={:.3} max_error={:.6}",
            self.device,
            self.rows,
            self.outputs,
            self.width,
            self.dispatch_milliseconds,
            self.wall_milliseconds,
            self.max_error
        )
    }
}

/// Enable Quartz Vulkan operators for subsequent SD graph execution.
pub fn set_sd_acceleration(enabled: bool) {
    SD_ACCELERATION.store(enabled, Ordering::Release);
}

pub fn sd_acceleration_requested() -> bool {
    SD_ACCELERATION.load(Ordering::Acquire)
        || std::env::var("QUARTZ_SD_VULKAN").is_ok_and(|value| value != "0")
}

pub fn print_statistics() {
    let Some(Ok(runtime)) = RUNTIME.get() else {
        return;
    };
    let Ok(runtime) = runtime.lock() else {
        eprintln!("Quartz Vulkan statistics unavailable: runtime lock was poisoned");
        return;
    };
    let stats = runtime.stats;
    println!(
        "Vulkan profile: elementwise={} calls/{:.3} dispatch/{:.3} wall ms norm={} calls/{:.3} dispatch/{:.3} wall ms gemm={} calls/{:.3} dispatch/{:.3} wall ms conv={} calls/{:.3} dispatch/{:.3} wall ms attention={} calls/{:.3} dispatch/{:.3} wall ms uploaded={} bytes cached_weights={} bytes peak_dispatch={} bytes resident_weight_uploads={} resident_tensor_uploads={} resident_downloads={} resident_uploaded={} bytes resident_downloaded={} bytes resident_allocated={} bytes peak_resident={} bytes resident_device_local={} bytes peak_device_local={} bytes resident_device_local_allocation={} bytes peak_device_local_allocation={} bytes",
        stats.elementwise.calls,
        stats.elementwise.dispatch_milliseconds,
        stats.elementwise.wall_milliseconds,
        stats.norm.calls,
        stats.norm.dispatch_milliseconds,
        stats.norm.wall_milliseconds,
        stats.gemm.calls,
        stats.gemm.dispatch_milliseconds,
        stats.gemm.wall_milliseconds,
        stats.conv2d.calls,
        stats.conv2d.dispatch_milliseconds,
        stats.conv2d.wall_milliseconds,
        stats.attention.calls,
        stats.attention.dispatch_milliseconds,
        stats.attention.wall_milliseconds,
        stats.uploaded_bytes,
        stats.cached_weight_bytes,
        stats.peak_dispatch_bytes,
        stats.resident_weight_uploads,
        stats.resident_tensor_uploads,
        stats.resident_downloads,
        stats.resident_uploaded_bytes,
        stats.resident_downloaded_bytes,
        stats.resident_allocated_bytes,
        stats.peak_resident_allocated_bytes,
        stats.resident_device_local_bytes,
        stats.peak_resident_device_local_bytes,
        stats.resident_device_local_allocation_bytes,
        stats.peak_resident_device_local_allocation_bytes,
    );
}

pub fn device_profile() -> Result<DeviceProfile> {
    with_runtime(|runtime| Ok(runtime.profile.clone()))
}

pub(crate) fn persistence_stats() -> Result<PersistenceStats> {
    with_runtime(|runtime| {
        Ok(PersistenceStats {
            resident_weight_uploads: runtime.stats.resident_weight_uploads,
            resident_tensor_uploads: runtime.stats.resident_tensor_uploads,
            resident_downloads: runtime.stats.resident_downloads,
            resident_uploaded_bytes: runtime.stats.resident_uploaded_bytes,
            resident_downloaded_bytes: runtime.stats.resident_downloaded_bytes,
            resident_allocated_bytes: runtime.stats.resident_allocated_bytes,
            peak_resident_allocated_bytes: runtime.stats.peak_resident_allocated_bytes,
            resident_device_local_bytes: runtime.stats.resident_device_local_bytes,
            peak_resident_device_local_bytes: runtime.stats.peak_resident_device_local_bytes,
            resident_device_local_allocation_bytes: runtime
                .stats
                .resident_device_local_allocation_bytes,
            peak_resident_device_local_allocation_bytes: runtime
                .stats
                .peak_resident_device_local_allocation_bytes,
            scratch_buffer_bytes: runtime
                .buffers
                .iter()
                .flatten()
                .map(|buffer| buffer.bytes as u64)
                .sum(),
            scratch_buffer_allocation_bytes: runtime
                .buffers
                .iter()
                .flatten()
                .map(|buffer| buffer.allocation_bytes)
                .sum(),
            cached_model_mappings: runtime.model_mappings.len(),
        })
    })
}

/// Release temporary activation buffers after an SD request while retaining
/// the model-weight arenas for the next request.
pub fn trim_sd_scratch() -> Result<()> {
    let Some(runtime) = RUNTIME.get() else {
        return Ok(());
    };
    let runtime = runtime
        .as_ref()
        .map_err(|error| anyhow!("Vulkan initialization failed: {error}"))?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("Vulkan runtime lock was poisoned"))?;
    unsafe {
        runtime
            .device
            .queue_wait_idle(runtime.queue)
            .map_err(|error| anyhow!("queue wait before scratch trim failed: {error:?}"))?;
    }
    for buffer in &mut runtime.buffers {
        drop(buffer.take());
    }
    Ok(())
}

/// Release both temporary buffers and cached SD weight arenas. The app can use
/// this when its inference surface closes and rebuild the cache on demand.
pub fn release_sd_resources() -> Result<()> {
    trim_sd_scratch()?;
    let Some(runtime) = RUNTIME.get() else {
        return Ok(());
    };
    let runtime = runtime
        .as_ref()
        .map_err(|error| anyhow!("Vulkan initialization failed: {error}"))?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("Vulkan runtime lock was poisoned"))?;
    runtime.model_mappings.clear();
    runtime.stats.cached_weight_bytes = 0;
    Ok(())
}

pub fn probe_external_host_memory(path: &str) -> Result<()> {
    let file = File::open(path).with_context(|| format!("cannot open host-import probe {path}"))?;
    let mapping = unsafe { MmapOptions::new().map(&file) }
        .with_context(|| format!("cannot map host-import probe {path}"))?;
    with_runtime(|runtime| runtime.probe_external_host_pointer(mapping.as_ptr(), mapping.len()))
}

/// Elementwise FP32 addition. This path never falls back to the CPU: callers
/// receive a Vulkan initialization or dispatch error if it cannot execute.
pub fn add(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    if left.shape() != right.shape() {
        bail!(
            "Vulkan add shape mismatch: {:?} vs {:?}",
            left.shape(),
            right.shape()
        );
    }
    if left.len() == 0 {
        return Ok(left.clone());
    }
    elementwise(left, right, 0, 0.0, 0.0)
}

pub fn multiply(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    if left.shape() != right.shape() {
        bail!(
            "Vulkan multiply shape mismatch: {:?} vs {:?}",
            left.shape(),
            right.shape()
        );
    }
    elementwise(left, right, 1, 0.0, 0.0)
}

pub fn scale(input: &Tensor, value: f32) -> Result<Tensor> {
    elementwise(input, input, 2, value, 0.0)
}

pub fn silu(input: &Tensor) -> Result<Tensor> {
    elementwise(input, input, 3, 0.0, 0.0)
}

pub fn gelu_tanh(input: &Tensor) -> Result<Tensor> {
    elementwise(input, input, 4, 0.0, 0.0)
}

pub fn clamp(input: &Tensor, minimum: f32, maximum: f32) -> Result<Tensor> {
    if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
        bail!("Vulkan clamp bounds must be finite and ordered");
    }
    elementwise(input, input, 5, minimum, maximum)
}

fn elementwise(
    left: &Tensor,
    right: &Tensor,
    operation: u32,
    parameter0: f32,
    parameter1: f32,
) -> Result<Tensor> {
    if left.len() == 0 {
        return Ok(left.clone());
    }
    let elements = u32::try_from(left.len()).context("Vulkan element count exceeds u32")?;
    let wall_started = Instant::now();
    let (output, _) = with_runtime(|runtime| {
        let result = runtime.elementwise(
            bytes_of(left.data()),
            bytes_of(right.data()),
            elements,
            operation,
            parameter0,
            parameter1,
        );
        runtime.stats.elementwise.wall_milliseconds +=
            wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    })?;
    Tensor::new(left.shape().to_vec(), output)
}

pub fn channel_rms_norm_3d(input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
    let [batch, channels, time, height, width]: [usize; 5] = input
        .shape()
        .try_into()
        .context("Vulkan channel RMSNorm input must be NCTHW")?;
    if weight.shape() != [channels] {
        bail!(
            "Vulkan channel RMSNorm weight shape {:?} must be [{channels}]",
            weight.shape()
        );
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        bail!("Vulkan channel RMSNorm epsilon must be positive and finite");
    }
    let volume = time
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .context("Vulkan channel RMSNorm volume overflow")?;
    let dimensions = [
        u32::try_from(batch).context("Vulkan RMSNorm batch exceeds u32")?,
        u32::try_from(channels).context("Vulkan RMSNorm channels exceed u32")?,
        u32::try_from(volume).context("Vulkan RMSNorm volume exceeds u32")?,
        epsilon.to_bits(),
    ];
    let wall_started = Instant::now();
    let (output, _) = with_runtime(|runtime| {
        let result = runtime.channel_rms_norm(
            bytes_of(input.data()),
            bytes_of(weight.data()),
            dimensions,
            input.len(),
        );
        runtime.stats.norm.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    })?;
    Tensor::new(input.shape().to_vec(), output)
}

/// Dense model-facing linear layer executed by the FP16-storage Vulkan GEMM
/// kernel. Narrowing is explicit; GEMM and optional bias addition run on the
/// Vulkan queue, and the result returns as FP32.
pub fn linear_tensor(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    if weight.shape().len() != 2 {
        bail!("Vulkan linear weight must be rank two");
    }
    let outputs = weight.shape()[0];
    let width = weight.shape()[1];
    if width == 0 || width % 4 != 0 || input.shape().last().copied() != Some(width) {
        bail!(
            "Vulkan linear input width {:?} and weight width {width} must match and be divisible by four",
            input.shape().last()
        );
    }
    if let Some(bias) = bias
        && bias.shape() != [outputs]
    {
        bail!(
            "Vulkan linear bias shape {:?} must be [{outputs}]",
            bias.shape()
        );
    }
    let rows = input.len() / width;
    let rows_u32 = u32::try_from(rows).context("Vulkan linear row count exceeds u32")?;
    let outputs_u32 = u32::try_from(outputs).context("Vulkan linear output width exceeds u32")?;
    let width_u32 = u32::try_from(width).context("Vulkan linear input width exceeds u32")?;
    let input_f16 = input
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let weight_f16 = weight
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let (output, _) = with_runtime(|runtime| {
        let (mut output, dispatch_milliseconds) = runtime.matmul(
            bytes_of(&input_f16),
            DispatchInput::Upload(bytes_of(&weight_f16)),
            rows_u32,
            outputs_u32,
            width_u32,
        )?;
        if let Some(bias) = bias {
            let wall_started = Instant::now();
            (output, _) = runtime.bias_add(
                bytes_of(&output),
                bytes_of(bias.data()),
                rows_u32
                    .checked_mul(outputs_u32)
                    .context("Vulkan linear output element count overflow")?,
                outputs_u32,
            )?;
            runtime.stats.elementwise.wall_milliseconds +=
                wall_started.elapsed().as_secs_f64() * 1_000.0;
        }
        Ok((output, dispatch_milliseconds))
    })?;
    let mut shape = input.shape().to_vec();
    *shape
        .last_mut()
        .context("Vulkan linear input has no shape")? = outputs;
    Tensor::new(shape, output)
}

fn resident_tensor(id: u64, elements: usize, element_type: ResidentElementType) -> ResidentTensor {
    ResidentTensor {
        lease: Arc::new(ResidentLease { id }),
        elements,
        element_type,
    }
}

pub(crate) fn upload_resident_tensor(input: &Tensor) -> Result<ResidentTensor> {
    if input.len() == 0 {
        bail!("resident Vulkan tensors cannot be empty");
    }
    let id = with_runtime(|runtime| {
        let id = runtime.upload_resident(
            bytes_of(input.data()),
            input.len(),
            ResidentElementType::F32,
            ResidentClass::Activation,
        )?;
        runtime.stats.resident_tensor_uploads += 1;
        runtime.stats.resident_uploaded_bytes = runtime
            .stats
            .resident_uploaded_bytes
            .saturating_add((input.len() * size_of::<f32>()) as u64);
        Ok(id)
    })?;
    Ok(resident_tensor(id, input.len(), ResidentElementType::F32))
}

pub(crate) fn download_resident_tensor(input: &ResidentTensor, shape: &[usize]) -> Result<Tensor> {
    if input.element_type != ResidentElementType::F32 {
        bail!("only resident FP32 tensors can be downloaded as Quartz tensors");
    }
    let expected = shape.iter().try_fold(1usize, |elements, &dimension| {
        elements.checked_mul(dimension)
    });
    if expected != Some(input.elements) {
        bail!(
            "resident tensor shape {:?} has {:?} elements, expected {}",
            shape,
            expected,
            input.elements
        );
    }
    let values = with_runtime(|runtime| runtime.download_resident(input.id(), input.elements))?;
    Tensor::new(shape.to_vec(), values)
}

pub(crate) fn resident_tensor_is_device_local(input: &ResidentTensor) -> Result<bool> {
    with_runtime(|runtime| runtime.resident_is_device_local(input.id()))
}

pub(crate) fn prepare_resident_linear(
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<ResidentLinearWeights> {
    let [output_width, input_width]: [usize; 2] = weight
        .shape()
        .try_into()
        .context("resident Vulkan linear weight must be rank two")?;
    if input_width == 0 || input_width % 4 != 0 || output_width == 0 {
        bail!("resident Vulkan linear dimensions must be non-zero and width divisible by four");
    }
    if let Some(bias) = bias
        && bias.shape() != [output_width]
    {
        bail!(
            "resident Vulkan linear bias shape {:?} must be [{output_width}]",
            bias.shape()
        );
    }
    let weight_f16 = weight
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let zero_bias;
    let bias_values = if let Some(bias) = bias {
        bias.data()
    } else {
        zero_bias = vec![0.0; output_width];
        &zero_bias
    };
    let (weight_id, bias_id) = with_runtime(|runtime| {
        runtime.prepare_resident_linear(
            bytes_of(&weight_f16),
            weight.len(),
            bytes_of(bias_values),
            bias_values.len(),
        )
    })?;
    Ok(ResidentLinearWeights {
        weight: resident_tensor(weight_id, weight.len(), ResidentElementType::F16),
        bias: resident_tensor(bias_id, bias_values.len(), ResidentElementType::F32),
        input_width,
        output_width,
    })
}

pub(crate) fn prepare_resident_conv3d(
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<ResidentConv3dWeights> {
    let [
        output_channels,
        input_channels,
        kernel_time,
        kernel_height,
        kernel_width,
    ]: [usize; 5] = weight
        .shape()
        .try_into()
        .context("resident Vulkan Conv3D weight must be rank five")?;
    if output_channels == 0
        || input_channels == 0
        || kernel_time == 0
        || kernel_height == 0
        || kernel_width == 0
    {
        bail!("resident Vulkan Conv3D weight dimensions must be non-zero");
    }
    if let Some(bias) = bias
        && bias.shape() != [output_channels]
    {
        bail!(
            "resident Vulkan Conv3D bias shape {:?} must be [{output_channels}]",
            bias.shape()
        );
    }
    let weight_f16 = weight
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let zero_bias;
    let bias_values = if let Some(bias) = bias {
        bias.data()
    } else {
        zero_bias = vec![0.0; output_channels];
        &zero_bias
    };
    let (weight_id, bias_id) = with_runtime(|runtime| {
        runtime.prepare_resident_linear(
            bytes_of(&weight_f16),
            weight.len(),
            bytes_of(bias_values),
            bias_values.len(),
        )
    })?;
    Ok(ResidentConv3dWeights {
        weight: resident_tensor(weight_id, weight.len(), ResidentElementType::F16),
        bias: resident_tensor(bias_id, bias_values.len(), ResidentElementType::F32),
        input_channels,
        output_channels,
        kernel: [kernel_time, kernel_height, kernel_width],
    })
}

pub(crate) fn prepare_resident_conv2d(
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<ResidentConv2dWeights> {
    let [output_channels, input_channels, kernel_height, kernel_width]: [usize; 4] = weight
        .shape()
        .try_into()
        .context("resident Vulkan Conv2D weight must be rank four")?;
    if output_channels == 0 || input_channels == 0 || kernel_height == 0 || kernel_width == 0 {
        bail!("resident Vulkan Conv2D weight dimensions must be non-zero");
    }
    if let Some(bias) = bias
        && bias.shape() != [output_channels]
    {
        bail!(
            "resident Vulkan Conv2D bias shape {:?} must be [{output_channels}]",
            bias.shape()
        );
    }
    let weight_f16 = weight
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let zero_bias;
    let bias_values = if let Some(bias) = bias {
        bias.data()
    } else {
        zero_bias = vec![0.0; output_channels];
        &zero_bias
    };
    let (weight_id, bias_id) = with_runtime(|runtime| {
        runtime.prepare_resident_linear(
            bytes_of(&weight_f16),
            weight.len(),
            bytes_of(bias_values),
            bias_values.len(),
        )
    })?;
    Ok(ResidentConv2dWeights {
        weight: resident_tensor(weight_id, weight.len(), ResidentElementType::F16),
        bias: resident_tensor(bias_id, bias_values.len(), ResidentElementType::F32),
        input_channels,
        output_channels,
        kernel: [kernel_height, kernel_width],
    })
}

pub(crate) fn resident_linear(
    input: &ResidentTensor,
    weights: &ResidentLinearWeights,
) -> Result<ResidentTensor> {
    if input.element_type != ResidentElementType::F32 {
        bail!("resident Vulkan linear input must be FP32");
    }
    if input.elements % weights.input_width != 0 {
        bail!("resident Vulkan linear input is not divisible into complete rows");
    }
    let rows = input.elements / weights.input_width;
    let output_elements = rows
        .checked_mul(weights.output_width)
        .context("resident Vulkan linear output size overflow")?;
    let id = with_runtime(|runtime| {
        runtime.resident_linear(
            input.id(),
            weights.weight.id(),
            weights.bias.id(),
            u32::try_from(rows).context("resident Vulkan linear rows exceed u32")?,
            u32::try_from(weights.output_width)
                .context("resident Vulkan linear outputs exceed u32")?,
            u32::try_from(weights.input_width)
                .context("resident Vulkan linear width exceeds u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        output_elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_conv3d(
    input: &ResidentTensor,
    weights: &ResidentConv3dWeights,
    input_shape: [usize; 5],
    padding_before: [usize; 3],
    padding_after: [usize; 3],
) -> Result<ResidentTensor> {
    let [batch, input_channels, input_time, input_height, input_width] = input_shape;
    if input.element_type != ResidentElementType::F32
        || input_channels != weights.input_channels
        || input.elements != batch * input_channels * input_time * input_height * input_width
    {
        bail!("resident Vulkan Conv3D input storage does not match its dimensions");
    }
    let input_axes = [input_time, input_height, input_width];
    let mut output_axes = [0; 3];
    for axis in 0..3 {
        let padded = input_axes[axis]
            .checked_add(padding_before[axis])
            .and_then(|value| value.checked_add(padding_after[axis]))
            .context("resident Vulkan Conv3D padded dimension overflow")?;
        if padded < weights.kernel[axis] {
            bail!("resident Vulkan Conv3D kernel exceeds padded input");
        }
        output_axes[axis] = padded - weights.kernel[axis] + 1;
    }
    let output_elements = batch
        .checked_mul(weights.output_channels)
        .and_then(|value| value.checked_mul(output_axes[0]))
        .and_then(|value| value.checked_mul(output_axes[1]))
        .and_then(|value| value.checked_mul(output_axes[2]))
        .context("resident Vulkan Conv3D output size overflow")?;
    let dimensions = [
        batch,
        input_channels,
        input_time,
        input_height,
        input_width,
        weights.output_channels,
        weights.kernel[0],
        weights.kernel[1],
        weights.kernel[2],
        padding_before[0],
        padding_before[1],
        padding_before[2],
        output_axes[0],
        output_axes[1],
        output_axes[2],
    ]
    .map(u32::try_from)
    .into_iter()
    .collect::<std::result::Result<Vec<_>, _>>()
    .context("resident Vulkan Conv3D dimensions exceed u32")?;
    let id = with_runtime(|runtime| {
        runtime.resident_conv3d(
            input.id(),
            weights.weight.id(),
            weights.bias.id(),
            dimensions
                .as_slice()
                .try_into()
                .expect("15 Conv3D dimensions"),
            output_elements,
        )
    })?;
    Ok(resident_tensor(
        id,
        output_elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_conv2d(
    input: &ResidentTensor,
    weights: &ResidentConv2dWeights,
    input_shape: [usize; 4],
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<ResidentTensor> {
    let [batch, input_channels, input_height, input_width] = input_shape;
    if input.element_type != ResidentElementType::F32
        || batch == 0
        || input_channels != weights.input_channels
        || input_height == 0
        || input_width == 0
        || stride.contains(&0)
        || input.elements != batch * input_channels * input_height * input_width
    {
        bail!("resident Vulkan Conv2D input storage does not match its dimensions");
    }
    let input_axes = [input_height, input_width];
    let mut output_axes = [0; 2];
    for axis in 0..2 {
        let padded = input_axes[axis]
            .checked_add(padding[axis].saturating_mul(2))
            .context("resident Vulkan Conv2D padded dimension overflow")?;
        if padded < weights.kernel[axis] {
            bail!("resident Vulkan Conv2D kernel exceeds padded input");
        }
        output_axes[axis] = (padded - weights.kernel[axis]) / stride[axis] + 1;
    }
    let output_elements = batch
        .checked_mul(weights.output_channels)
        .and_then(|value| value.checked_mul(output_axes[0]))
        .and_then(|value| value.checked_mul(output_axes[1]))
        .context("resident Vulkan Conv2D output size overflow")?;
    let dimensions = [
        batch,
        input_channels,
        input_height,
        input_width,
        weights.output_channels,
        weights.kernel[0],
        weights.kernel[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        output_axes[0],
        output_axes[1],
    ]
    .map(u32::try_from)
    .into_iter()
    .collect::<std::result::Result<Vec<_>, _>>()
    .context("resident Vulkan Conv2D dimensions exceed u32")?;
    let id = with_runtime(|runtime| {
        runtime.resident_conv2d(
            input.id(),
            weights.weight.id(),
            weights.bias.id(),
            dimensions
                .as_slice()
                .try_into()
                .expect("13 Conv2D dimensions"),
            output_elements,
        )
    })?;
    Ok(resident_tensor(
        id,
        output_elements,
        ResidentElementType::F32,
    ))
}

fn resident_vae_spatial_layout(
    input: &ResidentTensor,
    shape: [usize; 5],
    operation: usize,
    chunk: usize,
) -> Result<ResidentTensor> {
    let [batch, channels, time, height, width] = shape;
    if input.element_type != ResidentElementType::F32
        || [batch, channels, time, height, width].contains(&0)
        || operation > 4
        || (operation == 4 && chunk == 0)
        || (operation != 4 && chunk > 2)
    {
        bail!("resident VAE spatial layout parameters are invalid");
    }
    let base_elements = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(time))
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .context("resident VAE spatial layout size overflow")?;
    let expected_input = if operation == 2 {
        base_elements
            .checked_mul(3)
            .context("resident VAE QKV layout size overflow")?
    } else {
        base_elements
    };
    let output_elements = if operation == 4 {
        base_elements
            .checked_mul(chunk)
            .and_then(|value| value.checked_mul(chunk))
            .context("resident VAE nearest upsample size overflow")?
    } else {
        base_elements
    };
    if input.elements != expected_input
        || (operation != 2 && operation != 4 && chunk != 0)
        || (operation == 2 && chunk > 2)
    {
        bail!("resident VAE spatial layout input size does not match its operation");
    }
    let parameters = [
        operation,
        batch,
        channels,
        time,
        height,
        width,
        chunk,
        output_elements,
    ]
    .map(u32::try_from)
    .into_iter()
    .collect::<std::result::Result<Vec<_>, _>>()
    .context("resident VAE spatial layout dimensions exceed u32")?;
    let id = with_runtime(|runtime| {
        runtime.resident_vae_spatial_layout(
            input.id(),
            input.elements,
            parameters
                .as_slice()
                .try_into()
                .expect("8 VAE spatial layout parameters"),
        )
    })?;
    Ok(resident_tensor(
        id,
        output_elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_ncthw_to_frames(
    input: &ResidentTensor,
    shape: [usize; 5],
) -> Result<ResidentTensor> {
    resident_vae_spatial_layout(input, shape, 0, 0)
}

pub(crate) fn resident_frames_to_ncthw(
    input: &ResidentTensor,
    shape: [usize; 5],
) -> Result<ResidentTensor> {
    resident_vae_spatial_layout(input, shape, 1, 0)
}

pub(crate) fn resident_vae_qkv_sequence(
    input: &ResidentTensor,
    shape: [usize; 5],
    chunk: usize,
) -> Result<ResidentTensor> {
    resident_vae_spatial_layout(input, shape, 2, chunk)
}

pub(crate) fn resident_vae_sequence_to_frames(
    input: &ResidentTensor,
    shape: [usize; 5],
) -> Result<ResidentTensor> {
    resident_vae_spatial_layout(input, shape, 3, 0)
}

pub(crate) fn resident_ncthw_upsample_nearest(
    input: &ResidentTensor,
    shape: [usize; 5],
    scale: usize,
) -> Result<ResidentTensor> {
    resident_vae_spatial_layout(input, shape, 4, scale)
}

fn resident_ncthw_temporal(
    input0: &ResidentTensor,
    input1: &ResidentTensor,
    input0_shape: [usize; 5],
    input1_time: usize,
    output_shape: [usize; 5],
    operation: usize,
    parameter0: usize,
) -> Result<ResidentTensor> {
    let [batch, input0_channels, input0_time, height, width] = input0_shape;
    let [
        output_batch,
        output_channels,
        output_time,
        output_height,
        output_width,
    ] = output_shape;
    if batch == 0
        || input0_channels == 0
        || input0_time == 0
        || height == 0
        || width == 0
        || output_batch != batch
        || output_channels == 0
        || output_time == 0
        || output_height != height
        || output_width != width
    {
        bail!("resident NCTHW temporal dimensions are invalid");
    }
    let input0_elements = batch
        .checked_mul(input0_channels)
        .and_then(|value| value.checked_mul(input0_time))
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .context("resident NCTHW temporal input size overflow")?;
    let input1_elements = input1.elements;
    if input0.element_type != ResidentElementType::F32
        || input1.element_type != ResidentElementType::F32
        || input0.elements != input0_elements
    {
        bail!("resident NCTHW temporal input storage does not match its dimensions");
    }
    let output_elements = batch
        .checked_mul(output_channels)
        .and_then(|value| value.checked_mul(output_time))
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .context("resident NCTHW temporal output size overflow")?;
    let parameters = [
        operation,
        output_elements,
        batch,
        output_channels,
        output_time,
        height,
        width,
        input0_channels,
        input0_time,
        input1_time,
        parameter0,
    ]
    .map(u32::try_from)
    .into_iter()
    .collect::<std::result::Result<Vec<_>, _>>()
    .context("resident NCTHW temporal dimensions exceed u32")?;
    let id = with_runtime(|runtime| {
        runtime.resident_ncthw_temporal(
            input0.id(),
            input1.id(),
            input0_elements,
            input1_elements,
            parameters
                .as_slice()
                .try_into()
                .expect("11 NCTHW temporal parameters"),
        )
    })?;
    Ok(resident_tensor(
        id,
        output_elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_ncthw_slice_time(
    input: &ResidentTensor,
    input_shape: [usize; 5],
    start: usize,
    count: usize,
) -> Result<ResidentTensor> {
    if count == 0 {
        bail!("resident NCTHW temporal slice cannot be empty");
    }
    let end = start
        .checked_add(count)
        .context("resident NCTHW temporal slice overflow")?;
    if end > input_shape[2] {
        bail!("resident NCTHW temporal slice is out of bounds");
    }
    let output_shape = [
        input_shape[0],
        input_shape[1],
        count,
        input_shape[3],
        input_shape[4],
    ];
    resident_ncthw_temporal(
        input,
        input,
        input_shape,
        input_shape[2],
        output_shape,
        0,
        start,
    )
}

pub(crate) fn resident_ncthw_concat_time(
    left: &ResidentTensor,
    right: &ResidentTensor,
    left_shape: [usize; 5],
    right_time: usize,
) -> Result<ResidentTensor> {
    let output_time = left_shape[2]
        .checked_add(right_time)
        .context("resident NCTHW temporal concat overflow")?;
    let output_shape = [
        left_shape[0],
        left_shape[1],
        output_time,
        left_shape[3],
        left_shape[4],
    ];
    resident_ncthw_temporal(left, right, left_shape, right_time, output_shape, 1, 0)
}

pub(crate) fn resident_ncthw_prepend_zero_time(
    input: &ResidentTensor,
    input_shape: [usize; 5],
    count: usize,
) -> Result<ResidentTensor> {
    if count == 0 {
        return Ok(input.clone());
    }
    let output_time = input_shape[2]
        .checked_add(count)
        .context("resident NCTHW zero-time prepend overflow")?;
    let output_shape = [
        input_shape[0],
        input_shape[1],
        output_time,
        input_shape[3],
        input_shape[4],
    ];
    resident_ncthw_temporal(
        input,
        input,
        input_shape,
        input_shape[2],
        output_shape,
        2,
        count,
    )
}

pub(crate) fn resident_ncthw_channels_to_time(
    input: &ResidentTensor,
    input_shape: [usize; 5],
) -> Result<ResidentTensor> {
    if input_shape[1] % 2 != 0 {
        bail!("resident Wan channel-to-time shuffle needs an even channel count");
    }
    let output_time = input_shape[2]
        .checked_mul(2)
        .context("resident Wan channel-to-time output length overflow")?;
    let output_shape = [
        input_shape[0],
        input_shape[1] / 2,
        output_time,
        input_shape[3],
        input_shape[4],
    ];
    resident_ncthw_temporal(
        input,
        input,
        input_shape,
        input_shape[2],
        output_shape,
        3,
        0,
    )
}

pub(crate) fn resident_silu(input: &ResidentTensor) -> Result<ResidentTensor> {
    if input.element_type != ResidentElementType::F32 {
        bail!("resident Vulkan SiLU input must be FP32");
    }
    let elements = u32::try_from(input.elements).context("resident Vulkan SiLU exceeds u32")?;
    let id = with_runtime(|runtime| runtime.resident_silu(input.id(), elements))?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_gelu_tanh(input: &ResidentTensor) -> Result<ResidentTensor> {
    if input.element_type != ResidentElementType::F32 {
        bail!("resident Vulkan GELU input must be FP32");
    }
    let elements = u32::try_from(input.elements).context("resident Vulkan GELU exceeds u32")?;
    let id = with_runtime(|runtime| runtime.resident_gelu_tanh(input.id(), elements))?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_scale(input: &ResidentTensor, value: f32) -> Result<ResidentTensor> {
    if input.element_type != ResidentElementType::F32 || !value.is_finite() {
        bail!("resident Vulkan scale requires FP32 input and a finite value");
    }
    let elements = u32::try_from(input.elements).context("resident Vulkan scale exceeds u32")?;
    let id = with_runtime(|runtime| runtime.resident_scale(input.id(), elements, value))?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_affine(
    input: &ResidentTensor,
    scale: f32,
    bias: f32,
) -> Result<ResidentTensor> {
    if input.element_type != ResidentElementType::F32 || !scale.is_finite() || !bias.is_finite() {
        bail!("resident Vulkan affine requires FP32 input and finite parameters");
    }
    let elements = u32::try_from(input.elements).context("resident Vulkan affine exceeds u32")?;
    let id = with_runtime(|runtime| runtime.resident_affine(input.id(), elements, scale, bias))?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_ncthw_channel_affine(
    input: &ResidentTensor,
    parameters: &ResidentTensor,
    channels: usize,
    channel_plane: usize,
) -> Result<ResidentTensor> {
    if input.element_type != ResidentElementType::F32
        || parameters.element_type != ResidentElementType::F32
        || channels == 0
        || channel_plane == 0
        || parameters.elements != 2 * channels
        || channels
            .checked_mul(channel_plane)
            .is_none_or(|sample_elements| input.elements % sample_elements != 0)
    {
        bail!("resident NCTHW channel affine storage or dimensions are invalid");
    }
    let id = with_runtime(|runtime| {
        runtime.resident_ncthw_channel_affine(
            input.id(),
            parameters.id(),
            u32::try_from(input.elements)
                .context("resident NCTHW channel affine elements exceed u32")?,
            u32::try_from(channels).context("resident NCTHW channel count exceeds u32")?,
            u32::try_from(channel_plane).context("resident NCTHW channel plane exceeds u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_clamp(
    input: &ResidentTensor,
    minimum: f32,
    maximum: f32,
) -> Result<ResidentTensor> {
    if input.element_type != ResidentElementType::F32
        || !minimum.is_finite()
        || !maximum.is_finite()
        || minimum > maximum
    {
        bail!("resident Vulkan clamp requires FP32 input and ordered finite bounds");
    }
    let elements = u32::try_from(input.elements).context("resident Vulkan clamp exceeds u32")?;
    let id =
        with_runtime(|runtime| runtime.resident_clamp(input.id(), elements, minimum, maximum))?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_patchify(
    input: &ResidentTensor,
    channels: usize,
    time: usize,
    height: usize,
    width: usize,
    patch: (usize, usize, usize),
) -> Result<ResidentTensor> {
    let (patch_time, patch_height, patch_width) = patch;
    let elements = channels
        .checked_mul(time)
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .context("resident patchify element count overflow")?;
    if input.element_type != ResidentElementType::F32 || input.elements != elements {
        bail!("resident patchify input storage does not match its dimensions");
    }
    let id = with_runtime(|runtime| {
        runtime.resident_patch_layout(
            input.id(),
            0,
            u32::try_from(elements).context("resident patchify elements exceed u32")?,
            u32::try_from(channels).context("resident patchify channels exceed u32")?,
            u32::try_from(time).context("resident patchify time exceeds u32")?,
            u32::try_from(height).context("resident patchify height exceeds u32")?,
            u32::try_from(width).context("resident patchify width exceeds u32")?,
            u32::try_from(patch_time).context("resident patchify patch time exceeds u32")?,
            u32::try_from(patch_height).context("resident patchify patch height exceeds u32")?,
            u32::try_from(patch_width).context("resident patchify patch width exceeds u32")?,
            0,
        )
    })?;
    Ok(resident_tensor(id, elements, ResidentElementType::F32))
}

pub(crate) fn resident_unpatchify(
    input: &ResidentTensor,
    output_channels: usize,
    time: usize,
    height: usize,
    width: usize,
    patch: (usize, usize, usize),
) -> Result<ResidentTensor> {
    let elements = output_channels
        .checked_mul(time)
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .context("resident unpatchify element count overflow")?;
    if input.element_type != ResidentElementType::F32 || input.elements != elements {
        bail!("resident unpatchify input storage does not match its output dimensions");
    }
    let (patch_time, patch_height, patch_width) = patch;
    let id = with_runtime(|runtime| {
        runtime.resident_patch_layout(
            input.id(),
            1,
            u32::try_from(elements).context("resident unpatchify elements exceed u32")?,
            0,
            u32::try_from(time).context("resident unpatchify time exceeds u32")?,
            u32::try_from(height).context("resident unpatchify height exceeds u32")?,
            u32::try_from(width).context("resident unpatchify width exceeds u32")?,
            u32::try_from(patch_time).context("resident unpatchify patch time exceeds u32")?,
            u32::try_from(patch_height).context("resident unpatchify patch height exceeds u32")?,
            u32::try_from(patch_width).context("resident unpatchify patch width exceeds u32")?,
            u32::try_from(output_channels)
                .context("resident unpatchify output channels exceed u32")?,
        )
    })?;
    Ok(resident_tensor(id, elements, ResidentElementType::F32))
}

pub(crate) fn resident_wan_head_modulate(
    input: &ResidentTensor,
    timestep: &ResidentTensor,
    modulation: &ResidentTensor,
    width: usize,
) -> Result<ResidentTensor> {
    if input.element_type != ResidentElementType::F32
        || timestep.element_type != ResidentElementType::F32
        || modulation.element_type != ResidentElementType::F32
        || width == 0
        || input.elements % width != 0
        || timestep.elements != width
        || modulation.elements != 2 * width
    {
        bail!("resident Wan head modulation storage or dimensions are invalid");
    }
    let id = with_runtime(|runtime| {
        runtime.resident_wan_head_modulate(
            input.id(),
            timestep.id(),
            modulation.id(),
            u32::try_from(input.elements).context("resident Wan head elements exceed u32")?,
            u32::try_from(width).context("resident Wan head width exceeds u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn prepare_resident_vector(vector: &Tensor) -> Result<ResidentTensor> {
    let [elements]: [usize; 1] = vector
        .shape()
        .try_into()
        .context("resident Vulkan vector must be rank one")?;
    if elements == 0 {
        bail!("resident Vulkan vector cannot be empty");
    }
    let id =
        with_runtime(|runtime| runtime.prepare_resident_vector(bytes_of(vector.data()), elements))?;
    Ok(resident_tensor(id, elements, ResidentElementType::F32))
}

pub(crate) fn resident_add_vector(
    input: &ResidentTensor,
    vector: &ResidentTensor,
) -> Result<ResidentTensor> {
    if input.elements != vector.elements {
        bail!(
            "resident Vulkan add-vector lengths differ: {} vs {}",
            input.elements,
            vector.elements
        );
    }
    let elements =
        u32::try_from(input.elements).context("resident add-vector length exceeds u32")?;
    let id =
        with_runtime(|runtime| runtime.resident_add_vector(input.id(), vector.id(), elements))?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_add(
    left: &ResidentTensor,
    right: &ResidentTensor,
) -> Result<ResidentTensor> {
    resident_add_vector(left, right)
}

pub(crate) fn resident_multiply(
    left: &ResidentTensor,
    right: &ResidentTensor,
) -> Result<ResidentTensor> {
    if left.elements != right.elements
        || left.element_type != ResidentElementType::F32
        || right.element_type != ResidentElementType::F32
    {
        bail!(
            "resident Vulkan multiply storage differs: {} {:?} vs {} {:?}",
            left.elements,
            left.element_type,
            right.elements,
            right.element_type
        );
    }
    let elements = u32::try_from(left.elements).context("resident multiply length exceeds u32")?;
    let id = with_runtime(|runtime| runtime.resident_multiply(left.id(), right.id(), elements))?;
    Ok(resident_tensor(id, left.elements, ResidentElementType::F32))
}

pub(crate) fn resident_layer_norm(
    input: &ResidentTensor,
    affine: Option<(&ResidentTensor, &ResidentTensor)>,
    width: usize,
    epsilon: f32,
) -> Result<ResidentTensor> {
    if width == 0 || input.elements % width != 0 {
        bail!("resident Vulkan LayerNorm input is not divisible into complete rows");
    }
    let rows = input.elements / width;
    let affine_ids = affine.map(|(weight, bias)| (weight.id(), bias.id()));
    let id = with_runtime(|runtime| {
        runtime.resident_layer_norm(
            input.id(),
            affine_ids,
            u32::try_from(rows).context("resident LayerNorm row count exceeds u32")?,
            u32::try_from(width).context("resident LayerNorm width exceeds u32")?,
            epsilon,
        )
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_rms_norm(
    input: &ResidentTensor,
    weight: &ResidentTensor,
    width: usize,
    epsilon: f32,
) -> Result<ResidentTensor> {
    if width == 0 || input.elements % width != 0 {
        bail!("resident Vulkan RMSNorm input is not divisible into complete rows");
    }
    if weight.elements != width {
        bail!("resident Vulkan RMSNorm weight does not match the final axis");
    }
    let rows = input.elements / width;
    let id = with_runtime(|runtime| {
        runtime.resident_rms_norm(
            input.id(),
            weight.id(),
            u32::try_from(rows).context("resident RMSNorm row count exceeds u32")?,
            u32::try_from(width).context("resident RMSNorm width exceeds u32")?,
            epsilon,
        )
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_channel_rms_norm_3d(
    input: &ResidentTensor,
    weight: &ResidentTensor,
    shape: [usize; 5],
    epsilon: f32,
) -> Result<ResidentTensor> {
    let [batch, channels, time, height, width] = shape;
    let volume = time
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .context("resident channel RMSNorm volume overflow")?;
    let expected = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(volume))
        .context("resident channel RMSNorm input size overflow")?;
    if batch == 0
        || channels == 0
        || volume == 0
        || input.elements != expected
        || weight.elements != channels
    {
        bail!("resident channel RMSNorm dimensions do not match its tensors");
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        bail!("resident channel RMSNorm epsilon must be finite and positive");
    }
    let dimensions = [
        u32::try_from(batch).context("resident channel RMSNorm batch exceeds u32")?,
        u32::try_from(channels).context("resident channel RMSNorm channels exceed u32")?,
        u32::try_from(volume).context("resident channel RMSNorm volume exceeds u32")?,
        epsilon.to_bits(),
    ];
    let id = with_runtime(|runtime| {
        runtime.resident_channel_rms_norm(input.id(), weight.id(), dimensions, input.elements)
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_rope(
    input: &ResidentTensor,
    positions: &ResidentTensor,
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Result<ResidentTensor> {
    let width = heads
        .checked_mul(head_dim)
        .context("resident RoPE width overflow")?;
    let expected_input = rows
        .checked_mul(width)
        .context("resident RoPE input size overflow")?;
    let expected_positions = rows
        .checked_mul(head_dim / 2)
        .and_then(|values| values.checked_mul(4))
        .context("resident RoPE position size overflow")?;
    if head_dim == 0 || head_dim % 2 != 0 || input.elements != expected_input {
        bail!("resident Vulkan RoPE dimensions do not match the input");
    }
    if positions.elements != expected_positions {
        bail!("resident Vulkan RoPE position tensor has the wrong size");
    }
    let id = with_runtime(|runtime| {
        runtime.resident_rope(
            input.id(),
            positions.id(),
            u32::try_from(rows).context("resident RoPE row count exceeds u32")?,
            u32::try_from(heads).context("resident RoPE head count exceeds u32")?,
            u32::try_from(head_dim).context("resident RoPE head dimension exceeds u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_attention_scores(
    query: &ResidentTensor,
    key: &ResidentTensor,
    queries: usize,
    keys: usize,
    heads: usize,
    head_dim: usize,
    scale: f32,
    batches: usize,
) -> Result<ResidentTensor> {
    let width = heads
        .checked_mul(head_dim)
        .context("resident attention score width overflow")?;
    if batches == 0
        || query.elements != batches.saturating_mul(queries).saturating_mul(width)
        || key.elements != batches.saturating_mul(keys).saturating_mul(width)
    {
        bail!("resident Vulkan attention score inputs have the wrong size");
    }
    let output_elements = batches
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(queries))
        .and_then(|values| values.checked_mul(keys))
        .context("resident attention score size overflow")?;
    let id = with_runtime(|runtime| {
        runtime.resident_attention_scores(
            query.id(),
            key.id(),
            u32::try_from(queries).context("resident attention queries exceed u32")?,
            u32::try_from(keys).context("resident attention keys exceed u32")?,
            u32::try_from(heads).context("resident attention heads exceed u32")?,
            u32::try_from(head_dim).context("resident attention head dimension exceeds u32")?,
            scale,
            u32::try_from(batches).context("resident attention batches exceed u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        output_elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_softmax(
    input: &ResidentTensor,
    rows: usize,
    width: usize,
) -> Result<ResidentTensor> {
    if rows == 0 || width == 0 || input.elements != rows.saturating_mul(width) {
        bail!("resident Vulkan softmax dimensions do not match the input");
    }
    let id = with_runtime(|runtime| {
        runtime.resident_softmax(
            input.id(),
            u32::try_from(rows).context("resident softmax rows exceed u32")?,
            u32::try_from(width).context("resident softmax width exceeds u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_attention_values(
    probabilities: &ResidentTensor,
    value: &ResidentTensor,
    queries: usize,
    keys: usize,
    heads: usize,
    head_dim: usize,
    batches: usize,
) -> Result<ResidentTensor> {
    let width = heads
        .checked_mul(head_dim)
        .context("resident attention value width overflow")?;
    let probability_elements = batches
        .checked_mul(heads)
        .and_then(|values| values.checked_mul(queries))
        .and_then(|values| values.checked_mul(keys))
        .context("resident attention probability size overflow")?;
    if probabilities.elements != probability_elements
        || value.elements != batches.saturating_mul(keys).saturating_mul(width)
    {
        bail!("resident Vulkan attention value inputs have the wrong size");
    }
    let output_elements = batches
        .checked_mul(queries)
        .and_then(|values| values.checked_mul(width))
        .context("resident attention output size overflow")?;
    let id = with_runtime(|runtime| {
        runtime.resident_attention_values(
            probabilities.id(),
            value.id(),
            u32::try_from(queries).context("resident attention queries exceed u32")?,
            u32::try_from(keys).context("resident attention keys exceed u32")?,
            u32::try_from(heads).context("resident attention heads exceed u32")?,
            u32::try_from(head_dim).context("resident attention head dimension exceeds u32")?,
            u32::try_from(batches).context("resident attention batches exceed u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        output_elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_wan_modulate(
    input: &ResidentTensor,
    modulation: &ResidentTensor,
    width: usize,
    shift_chunk: usize,
    scale_chunk: usize,
) -> Result<ResidentTensor> {
    if width == 0 || input.elements % width != 0 {
        bail!("resident Wan modulation input is not divisible into complete rows");
    }
    let required_chunks = shift_chunk.max(scale_chunk) + 1;
    if modulation.elements < required_chunks.saturating_mul(width) {
        bail!("resident Wan modulation does not contain every requested chunk");
    }
    let id = with_runtime(|runtime| {
        runtime.resident_wan_modulate(
            input.id(),
            modulation.id(),
            u32::try_from(input.elements).context("resident modulation length exceeds u32")?,
            u32::try_from(width).context("resident modulation width exceeds u32")?,
            u32::try_from(shift_chunk).context("resident shift chunk exceeds u32")?,
            u32::try_from(scale_chunk).context("resident scale chunk exceeds u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

pub(crate) fn resident_multiply_vector_chunk(
    input: &ResidentTensor,
    vector: &ResidentTensor,
    width: usize,
    chunk: usize,
) -> Result<ResidentTensor> {
    if width == 0 || input.elements % width != 0 {
        bail!("resident vector multiply input is not divisible into complete rows");
    }
    if vector.elements < (chunk + 1).saturating_mul(width) {
        bail!("resident vector multiply tensor does not contain the requested chunk");
    }
    let id = with_runtime(|runtime| {
        runtime.resident_multiply_vector_chunk(
            input.id(),
            vector.id(),
            u32::try_from(input.elements).context("resident vector multiply length exceeds u32")?,
            u32::try_from(width).context("resident vector multiply width exceeds u32")?,
            u32::try_from(chunk).context("resident vector multiply chunk exceeds u32")?,
        )
    })?;
    Ok(resident_tensor(
        id,
        input.elements,
        ResidentElementType::F32,
    ))
}

fn release_resident_buffer(id: u64) {
    let Some(Ok(runtime)) = RUNTIME.get() else {
        return;
    };
    let Ok(mut runtime) = runtime.lock() else {
        return;
    };
    runtime.release_resident(id);
}

/// Linear projection from an f32 CPU tensor through mapped FP16 weights.
/// Activations are narrowed for upload, the GPU performs packed FP16 products
/// with FP32 accumulation, and the result returns as f32 for the reference
/// graph's remaining operators.
pub fn linear(
    input: &Tensor,
    weight: MappedTensor,
    bias: Option<TensorView<'_>>,
) -> Result<Tensor> {
    if weight.dtype != DType::F16 || weight.shape.len() != 2 {
        bail!("Vulkan linear weight must be a rank-2 F16 tensor");
    }
    let outputs = weight.shape[0];
    let width = weight.shape[1];
    if width % 4 != 0 {
        bail!("Vulkan FP16 linear width {width} is not divisible by four");
    }
    if input.shape().last().copied() != Some(width) {
        bail!(
            "Vulkan linear input width {:?} does not match weight width {width}",
            input.shape().last()
        );
    }
    if let Some(bias) = bias {
        if bias.dtype != DType::F16 || bias.shape != [outputs] {
            bail!("Vulkan linear bias must be F16 with shape [{outputs}]");
        }
    }
    let rows = input.len() / width;
    let input_f16 = input
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let (mut output, _) = with_runtime(|runtime| {
        runtime.matmul(
            bytes_of(&input_f16),
            DispatchInput::Mapped(&weight),
            rows as u32,
            outputs as u32,
            width as u32,
        )
    })?;
    if let Some(bias) = bias {
        output.par_chunks_mut(outputs).for_each(|row| {
            for (column, value) in row.iter_mut().enumerate() {
                *value += bias.value(column);
            }
        });
    }
    let mut shape = input.shape().to_vec();
    *shape
        .last_mut()
        .context("Vulkan linear input has no shape")? = outputs;
    Tensor::new(shape, output)
}

pub fn conv2d(
    input: &Tensor,
    weight: MappedTensor,
    bias: TensorView<'_>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor> {
    let [batch, input_channels, input_height, input_width]: [usize; 4] =
        input
            .shape()
            .try_into()
            .context("Vulkan convolution input must be NCHW")?;
    if weight.dtype != DType::F16 || weight.shape.len() != 4 {
        bail!("Vulkan convolution weight must be rank-4 F16 OIHW");
    }
    let output_channels = weight.shape[0];
    if weight.shape[1] != input_channels {
        bail!("Vulkan convolution weight input channels do not match its input");
    }
    let kernel_height = weight.shape[2];
    let kernel_width = weight.shape[3];
    if bias.dtype != DType::F16 || bias.shape != [output_channels] {
        bail!("Vulkan convolution bias must be F16 with shape [{output_channels}]");
    }
    if stride.contains(&0) || kernel_height == 0 || kernel_width == 0 {
        bail!("Vulkan convolution stride and kernel dimensions must be non-zero");
    }
    let padded_height = input_height
        .checked_add(padding[0].checked_mul(2).context("padding overflow")?)
        .context("padded height overflow")?;
    let padded_width = input_width
        .checked_add(padding[1].checked_mul(2).context("padding overflow")?)
        .context("padded width overflow")?;
    if padded_height < kernel_height || padded_width < kernel_width {
        bail!("Vulkan convolution kernel is larger than the padded input");
    }
    let output_height = (padded_height - kernel_height) / stride[0] + 1;
    let output_width = (padded_width - kernel_width) / stride[1] + 1;
    let input_f16 = input
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let dimensions = [
        batch as u32,
        input_channels as u32,
        input_height as u32,
        input_width as u32,
        output_channels as u32,
        kernel_height as u32,
        kernel_width as u32,
        stride[0] as u32,
        stride[1] as u32,
        padding[0] as u32,
        padding[1] as u32,
        output_height as u32,
        output_width as u32,
    ];
    let output_len = batch
        .checked_mul(output_channels)
        .and_then(|value| value.checked_mul(output_height))
        .and_then(|value| value.checked_mul(output_width))
        .context("Vulkan convolution output size overflow")?;
    let (mut output, _) = with_runtime(|runtime| {
        runtime.conv2d(bytes_of(&input_f16), &weight, &dimensions, output_len)
    })?;
    let output_plane = output_height * output_width;
    output
        .par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(plane_index, plane)| {
            let value = bias.value(plane_index % output_channels);
            for element in plane {
                *element += value;
            }
        });
    Tensor::new(
        vec![batch, output_channels, output_height, output_width],
        output,
    )
}

/// Execute an SD1.5 ResNet block as one Vulkan submission. Quartz owns every
/// stage: group normalization, SiLU, both convolutions, optional shortcut,
/// timestep bias, and residual addition.
pub fn resnet(
    input: &Tensor,
    channel_bias: Option<&Tensor>,
    weights: &SafeTensorFile,
    prefix: &str,
    groups: usize,
    epsilon: f32,
) -> Result<Tensor> {
    let [batch, input_channels, height, width]: [usize; 4] = input
        .shape()
        .try_into()
        .context("Vulkan ResNet input must be NCHW")?;
    if groups == 0 || input_channels % groups != 0 || !epsilon.is_finite() || epsilon <= 0.0 {
        bail!("invalid Vulkan ResNet group-normalization parameters");
    }

    let conv1 = weights.mapped(&format!("{prefix}.conv1.weight"))?;
    let conv2 = weights.mapped(&format!("{prefix}.conv2.weight"))?;
    if conv1.dtype != DType::F16
        || conv2.dtype != DType::F16
        || conv1.shape.len() != 4
        || conv2.shape.len() != 4
        || conv1.shape[1] != input_channels
        || conv1.shape[2..] != [3, 3]
    {
        bail!("Vulkan ResNet convolution weights do not match the fixed SD1.5 block");
    }
    let output_channels = conv1.shape[0];
    if output_channels % groups != 0 || conv2.shape != [output_channels, output_channels, 3, 3] {
        bail!("Vulkan ResNet output shape is incompatible with its second convolution");
    }
    if let Some(channel_bias) = channel_bias {
        if channel_bias.shape() != [batch, output_channels] {
            bail!(
                "Vulkan ResNet channel bias shape {:?} must be [{batch}, {output_channels}]",
                channel_bias.shape()
            );
        }
    }

    let shortcut_name = format!("{prefix}.conv_shortcut.weight");
    let shortcut = weights
        .info(&shortcut_name)
        .map(|_| weights.mapped(&shortcut_name))
        .transpose()?;
    if let Some(shortcut) = shortcut.as_ref() {
        if shortcut.dtype != DType::F16 || shortcut.shape != [output_channels, input_channels, 1, 1]
        {
            bail!("Vulkan ResNet shortcut weight has an invalid shape");
        }
    } else if input_channels != output_channels {
        bail!("Vulkan ResNet changes channels without a shortcut convolution");
    }

    let norm1_weight = weights.view(&format!("{prefix}.norm1.weight"))?;
    let norm1_bias = weights.view(&format!("{prefix}.norm1.bias"))?;
    let norm2_weight = weights.view(&format!("{prefix}.norm2.weight"))?;
    let norm2_bias = weights.view(&format!("{prefix}.norm2.bias"))?;
    let conv1_bias = weights.view(&format!("{prefix}.conv1.bias"))?;
    let conv2_bias = weights.view(&format!("{prefix}.conv2.bias"))?;
    for (label, view, channels) in [
        ("norm1 weight", norm1_weight, input_channels),
        ("norm1 bias", norm1_bias, input_channels),
        ("norm2 weight", norm2_weight, output_channels),
        ("norm2 bias", norm2_bias, output_channels),
        ("conv1 bias", conv1_bias, output_channels),
        ("conv2 bias", conv2_bias, output_channels),
    ] {
        if view.dtype != DType::F16 || view.shape != [channels] {
            bail!("Vulkan ResNet {label} must be F16 with shape [{channels}]");
        }
    }

    let mut norm1_parameters = Vec::with_capacity(input_channels * 2);
    extend_view_f32(&mut norm1_parameters, norm1_weight);
    extend_view_f32(&mut norm1_parameters, norm1_bias);
    let mut norm2_parameters =
        Vec::with_capacity(output_channels * (3 + usize::from(channel_bias.is_some()) * batch));
    extend_view_f32(&mut norm2_parameters, norm2_weight);
    extend_view_f32(&mut norm2_parameters, norm2_bias);
    extend_view_f32(&mut norm2_parameters, conv1_bias);
    if let Some(channel_bias) = channel_bias {
        norm2_parameters.extend_from_slice(channel_bias.data());
    }
    let mut residual_biases = Vec::with_capacity(output_channels * 2);
    extend_view_f32(&mut residual_biases, conv2_bias);
    if shortcut.is_some() {
        let shortcut_bias = weights.view(&format!("{prefix}.conv_shortcut.bias"))?;
        if shortcut_bias.dtype != DType::F16 || shortcut_bias.shape != [output_channels] {
            bail!("Vulkan ResNet shortcut bias has an invalid shape");
        }
        extend_view_f32(&mut residual_biases, shortcut_bias);
    }

    let conv1_dimensions = [
        batch as u32,
        input_channels as u32,
        height as u32,
        width as u32,
        output_channels as u32,
        3,
        3,
        1,
        1,
        1,
        1,
        height as u32,
        width as u32,
    ];
    let conv2_dimensions = [
        batch as u32,
        output_channels as u32,
        height as u32,
        width as u32,
        output_channels as u32,
        3,
        3,
        1,
        1,
        1,
        1,
        height as u32,
        width as u32,
    ];
    let shortcut_dimensions = shortcut.as_ref().map(|_| {
        [
            batch as u32,
            input_channels as u32,
            height as u32,
            width as u32,
            output_channels as u32,
            1,
            1,
            1,
            1,
            0,
            0,
            height as u32,
            width as u32,
        ]
    });
    let input_f16 = shortcut.as_ref().map(|_| {
        input
            .data()
            .par_iter()
            .copied()
            .map(f32_to_f16)
            .collect::<Vec<_>>()
    });
    let output_len = batch
        .checked_mul(output_channels)
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(width))
        .context("Vulkan ResNet output size overflow")?;
    let output = with_runtime(|runtime| {
        runtime.resnet(
            bytes_of(input.data()),
            input_f16.as_deref().map(bytes_of),
            bytes_of(&norm1_parameters),
            bytes_of(&norm2_parameters),
            bytes_of(&residual_biases),
            &conv1,
            &conv2,
            shortcut.as_ref(),
            &conv1_dimensions,
            &conv2_dimensions,
            shortcut_dimensions.as_ref(),
            groups as u32,
            epsilon,
            channel_bias.is_some(),
            output_len,
        )
    })?;
    Tensor::new(vec![batch, output_channels, height, width], output)
}

fn extend_view_f32(output: &mut Vec<f32>, view: TensorView<'_>) {
    output.extend((0..view.len()).map(|index| view.value(index)));
}

pub fn projected_attention(
    query_input: &Tensor,
    key_value_input: &Tensor,
    weights: &SafeTensorFile,
    prefix: &str,
    heads: usize,
) -> Result<Tensor> {
    if query_input.shape().len() != 3 || key_value_input.shape().len() != 3 {
        bail!("Vulkan projected attention inputs must be rank three");
    }
    let batch = query_input.shape()[0];
    let queries = query_input.shape()[1];
    let query_width = query_input.shape()[2];
    let keys = key_value_input.shape()[1];
    let key_value_width = key_value_input.shape()[2];
    if key_value_input.shape()[0] != batch || heads == 0 || query_width % heads != 0 {
        bail!("Vulkan projected attention batch/head dimensions are invalid");
    }
    let query_weight = weights.mapped(&format!("{prefix}.to_q.weight"))?;
    let key_weight = weights.mapped(&format!("{prefix}.to_k.weight"))?;
    let value_weight = weights.mapped(&format!("{prefix}.to_v.weight"))?;
    let output_weight = weights.mapped(&format!("{prefix}.to_out.0.weight"))?;
    for (label, weight, expected) in [
        ("query", &query_weight, [query_width, query_width]),
        ("key", &key_weight, [query_width, key_value_width]),
        ("value", &value_weight, [query_width, key_value_width]),
        ("output", &output_weight, [query_width, query_width]),
    ] {
        if weight.dtype != DType::F16 || weight.shape != expected {
            bail!("Vulkan projected-attention {label} weight has an invalid shape");
        }
    }
    for projection in ["to_q", "to_k", "to_v"] {
        if weights
            .info(&format!("{prefix}.{projection}.bias"))
            .is_some()
        {
            bail!("Vulkan projected attention does not accept Q/K/V biases");
        }
    }
    let query_f16 = query_input
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let key_value_f16 = key_value_input
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let mut output = with_runtime(|runtime| {
        runtime.projected_attention(
            bytes_of(&query_f16),
            bytes_of(&key_value_f16),
            &query_weight,
            &key_weight,
            &value_weight,
            &output_weight,
            batch as u32,
            queries as u32,
            keys as u32,
            query_width as u32,
            key_value_width as u32,
            heads as u32,
        )
    })?;
    let output_bias_name = format!("{prefix}.to_out.0.bias");
    if let Some(output_bias) = weights
        .info(&output_bias_name)
        .map(|_| weights.view(&output_bias_name))
        .transpose()?
    {
        if output_bias.dtype != DType::F16 || output_bias.shape != [query_width] {
            bail!("Vulkan projected-attention output bias has an invalid shape");
        }
        output.par_chunks_mut(query_width).for_each(|row| {
            for (column, value) in row.iter_mut().enumerate() {
                *value += output_bias.value(column);
            }
        });
    }
    Tensor::new(vec![batch, queries, query_width], output)
}

pub fn feed_forward(input: &Tensor, weights: &SafeTensorFile, prefix: &str) -> Result<Tensor> {
    if input.shape().len() != 3 {
        bail!("Vulkan feed-forward input must be rank three");
    }
    let channels = input.shape()[2];
    let first = weights.mapped(&format!("{prefix}.net.0.proj.weight"))?;
    let second = weights.mapped(&format!("{prefix}.net.2.weight"))?;
    if first.dtype != DType::F16
        || second.dtype != DType::F16
        || first.shape.len() != 2
        || first.shape[1] != channels
        || first.shape[0] % 2 != 0
        || second.shape != [channels, first.shape[0] / 2]
    {
        bail!("Vulkan feed-forward weights have invalid shapes");
    }
    let hidden = first.shape[0] / 2;
    let first_bias = weights.view(&format!("{prefix}.net.0.proj.bias"))?;
    let second_bias = weights.view(&format!("{prefix}.net.2.bias"))?;
    if first_bias.dtype != DType::F16
        || first_bias.shape != [hidden * 2]
        || second_bias.dtype != DType::F16
        || second_bias.shape != [channels]
    {
        bail!("Vulkan feed-forward biases have invalid shapes");
    }
    let input_f16 = input
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let first_bias_f32 = (0..first_bias.len())
        .map(|index| first_bias.value(index))
        .collect::<Vec<_>>();
    let rows = input.len() / channels;
    let mut output = with_runtime(|runtime| {
        runtime.feed_forward(
            bytes_of(&input_f16),
            bytes_of(&first_bias_f32),
            &first,
            &second,
            rows as u32,
            channels as u32,
            hidden as u32,
        )
    })?;
    output.par_chunks_mut(channels).for_each(|row| {
        for (column, value) in row.iter_mut().enumerate() {
            *value += second_bias.value(column);
        }
    });
    Tensor::new(input.shape().to_vec(), output)
}

pub fn attention(query: &Tensor, key: &Tensor, value: &Tensor, causal: bool) -> Result<Tensor> {
    if query.shape().len() != 4 || key.shape().len() != 4 || value.shape().len() != 4 {
        bail!("Vulkan attention inputs must all be rank four");
    }
    let [batch, heads, queries, width]: [usize; 4] =
        query.shape().try_into().expect("validated attention rank");
    let keys = key.shape()[2];
    if key.shape() != [batch, heads, keys, width] || value.shape() != [batch, heads, keys, width] {
        bail!("Vulkan attention K/V shapes are incompatible with Q");
    }
    if keys > 4096 {
        bail!("Vulkan attention supports at most 4096 keys, got {keys}");
    }
    if width % 4 != 0 {
        bail!("Vulkan attention width {width} is not divisible by four");
    }
    if causal && queries != keys {
        bail!("Vulkan causal attention requires equal query and key lengths");
    }
    let query_f16 = query
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let key_f16 = key
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let value_f16 = value
        .data()
        .par_iter()
        .copied()
        .map(f32_to_f16)
        .collect::<Vec<_>>();
    let dimensions = [
        (batch * heads) as u32,
        queries as u32,
        keys as u32,
        width as u32,
        (1.0 / (width as f32).sqrt()).to_bits(),
        u32::from(causal),
        0,
    ];
    let (output, _) = with_runtime(|runtime| {
        runtime.attention(
            bytes_of(&query_f16),
            bytes_of(&key_f16),
            bytes_of(&value_f16),
            &dimensions,
            query.len(),
        )
    })?;
    Tensor::new(query.shape().to_vec(), output)
}

pub fn benchmark_fp16_gemm() -> Result<BenchmarkResult> {
    const ROWS: u32 = 4096;
    const OUTPUTS: u32 = 320;
    const WIDTH: u32 = 320;
    let input = (0..ROWS as usize * WIDTH as usize)
        .map(|index| f32_to_f16(((index % 31) as f32 - 15.0) / 16.0))
        .collect::<Vec<_>>();
    let weights = (0..OUTPUTS as usize * WIDTH as usize)
        .map(|index| f32_to_f16(((index % 29) as f32 - 14.0) / 32.0))
        .collect::<Vec<_>>();

    let wall_started = Instant::now();
    let (output, dispatch_milliseconds, device) = with_runtime(|runtime| {
        let device = runtime.device_name.clone();
        let (output, elapsed) = runtime.matmul(
            bytes_of(&input),
            DispatchInput::Upload(bytes_of(&weights)),
            ROWS,
            OUTPUTS,
            WIDTH,
        )?;
        Ok((output, elapsed, device))
    })?;
    let wall_milliseconds = wall_started.elapsed().as_secs_f64() * 1_000.0;

    let mut max_error = 0.0f32;
    for row in [0usize, 7, ROWS as usize - 1] {
        for column in [0usize, 11, OUTPUTS as usize - 1] {
            let mut expected = 0.0;
            for inner in 0..WIDTH as usize {
                expected += crate::dequant::f16_to_f32(input[row * WIDTH as usize + inner])
                    * crate::dequant::f16_to_f32(weights[column * WIDTH as usize + inner]);
            }
            max_error = max_error.max((output[row * OUTPUTS as usize + column] - expected).abs());
        }
    }

    Ok(BenchmarkResult {
        device,
        rows: ROWS,
        outputs: OUTPUTS,
        width: WIDTH,
        dispatch_milliseconds,
        wall_milliseconds,
        max_error,
    })
}

fn with_runtime<T>(operation: impl FnOnce(&mut VulkanRuntime) -> Result<T>) -> Result<T> {
    let runtime = RUNTIME.get_or_init(|| {
        VulkanRuntime::new()
            .map(Mutex::new)
            .map_err(|error| format!("{error:#}"))
    });
    let runtime = runtime
        .as_ref()
        .map_err(|error| anyhow!("Vulkan initialization failed: {error}"))?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("Vulkan runtime lock was poisoned"))?;
    operation(&mut runtime)
}

fn select_physical_device(
    instance: &ash::Instance,
) -> Result<(vk::PhysicalDevice, vk::PhysicalDeviceProperties, u32)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|error| anyhow!("cannot enumerate Vulkan devices: {error:?}"))?;
    if physical_devices.is_empty() {
        bail!("no Vulkan physical device");
    }
    let requested = std::env::var("QUARTZ_VULKAN_DEVICE").ok();
    let requested_index = requested
        .as_deref()
        .and_then(|value| value.parse::<usize>().ok());
    let requested_name = requested.as_deref().map(str::to_ascii_lowercase);
    let mut candidates = Vec::new();
    let mut diagnostics = Vec::new();

    for (index, physical) in physical_devices.into_iter().enumerate() {
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let selected_by_override = requested.is_none()
            || requested_index == Some(index)
            || requested_name
                .as_ref()
                .is_some_and(|wanted| name.to_ascii_lowercase().contains(wanted));
        if !selected_by_override {
            diagnostics.push(format!("[{index}] {name:?}: not selected by override"));
            continue;
        }

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = queue_families
            .iter()
            .enumerate()
            .filter(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .min_by_key(|(_, family)| {
                u8::from(family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            })
            .map(|(index, _)| index as u32);
        let Some(queue_family) = queue_family else {
            diagnostics.push(format!("[{index}] {name:?}: no compute queue"));
            continue;
        };

        let mut storage_support = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut float_support = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut storage_support)
            .push_next(&mut float_support)
            .build();
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
        if storage_support.storage_buffer16_bit_access == 0 || float_support.shader_float16 == 0 {
            diagnostics.push(format!("[{index}] {name:?}: no FP16 arithmetic/storage"));
            continue;
        }
        if properties.limits.max_compute_work_group_invocations < REQUIRED_WORKGROUP_INVOCATIONS
            || properties.limits.max_compute_work_group_size[0] < REQUIRED_WORKGROUP_INVOCATIONS
            || properties.limits.max_compute_shared_memory_size < REQUIRED_SHARED_MEMORY
        {
            diagnostics.push(format!("[{index}] {name:?}: insufficient compute limits"));
            continue;
        }

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let device_local_bytes = memory_properties.memory_heaps
            [..memory_properties.memory_heap_count as usize]
            .iter()
            .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum::<u64>();
        let device_type_priority = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 4u8,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 1,
            _ => 0,
        };
        diagnostics.push(format!(
            "[{index}] {name:?}: usable type={} local_memory={device_local_bytes}",
            properties.device_type.as_raw()
        ));
        candidates.push((
            (device_type_priority, device_local_bytes, usize::MAX - index),
            physical,
            properties,
            queue_family,
        ));
    }

    let Some((_, physical, properties, queue_family)) =
        candidates.into_iter().max_by_key(|candidate| candidate.0)
    else {
        let override_description = requested
            .map(|value| format!(" matching QUARTZ_VULKAN_DEVICE={value:?}"))
            .unwrap_or_default();
        bail!(
            "no usable Vulkan compute device{override_description}; candidates: {}",
            diagnostics.join("; ")
        );
    };
    Ok((physical, properties, queue_family))
}

struct ResidentBuffer {
    buffer: Buffer,
    logical_bytes: usize,
    elements: usize,
    element_type: ResidentElementType,
    class: ResidentClass,
}

struct VulkanRuntime {
    _entry: Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue_family: u32,
    queue: vk::Queue,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    gemm_pipeline: vk::Pipeline,
    conv2d_pipeline: vk::Pipeline,
    attention_pipeline: vk::Pipeline,
    im2col_pipeline: vk::Pipeline,
    groupnorm_silu_pipeline: vk::Pipeline,
    residual_add_pipeline: vk::Pipeline,
    gemm_heads_pipeline: vk::Pipeline,
    merge_heads_pipeline: vk::Pipeline,
    geglu_pipeline: vk::Pipeline,
    elementwise_pipeline: vk::Pipeline,
    channel_rmsnorm_pipeline: vk::Pipeline,
    resident_linear_pipeline: vk::Pipeline,
    layernorm_pipeline: vk::Pipeline,
    rmsnorm_pipeline: vk::Pipeline,
    rope_pipeline: vk::Pipeline,
    f32_attention_pipeline: vk::Pipeline,
    patch_layout_pipeline: vk::Pipeline,
    wan_head_modulate_pipeline: vk::Pipeline,
    resident_conv3d_pipeline: vk::Pipeline,
    resident_conv2d_pipeline: vk::Pipeline,
    ncthw_temporal_pipeline: vk::Pipeline,
    vae_spatial_layout_pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    device_name: String,
    profile: DeviceProfile,
    stats: RuntimeStats,
    buffers: [Option<Buffer>; 8],
    resident_buffers: HashMap<u64, ResidentBuffer>,
    next_resident_id: u64,
    external_host: Option<vk::ExtExternalMemoryHostFn>,
    external_host_alignment: u64,
    storage_buffer_alignment: u64,
    model_mappings: HashMap<usize, ModelMapping>,
}

impl VulkanRuntime {
    fn new() -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("cannot load the Vulkan loader")?;
        let app_name = c"Quartz";
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .application_version(1)
            .engine_name(app_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_2);
        let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .map_err(|error| anyhow!("vkCreateInstance failed: {error:?}"))?;

        let setup = (|| -> Result<_> {
            let (physical, properties, queue_family) = select_physical_device(&instance)?;
            let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical) };

            let mut storage_support = vk::PhysicalDevice16BitStorageFeatures::default();
            let mut float_support = vk::PhysicalDeviceShaderFloat16Int8Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::builder()
                .push_next(&mut storage_support)
                .push_next(&mut float_support)
                .build();
            unsafe { instance.get_physical_device_features2(physical, &mut features) };
            if storage_support.storage_buffer16_bit_access == 0 || float_support.shader_float16 == 0
            {
                bail!("Vulkan device lacks FP16 arithmetic or storage-buffer support");
            }

            let priorities = [1.0f32];
            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)
                .build()];
            let mut enable_storage = vk::PhysicalDevice16BitStorageFeatures::builder()
                .storage_buffer16_bit_access(true)
                .build();
            let mut enable_float = vk::PhysicalDeviceShaderFloat16Int8Features::builder()
                .shader_float16(true)
                .build();
            let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
                .map_err(|error| anyhow!("cannot enumerate Vulkan device extensions: {error:?}"))?;
            let has_extension = |wanted: &[u8]| {
                extensions.iter().any(|extension| {
                    let name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                    name.to_bytes() == wanted
                })
            };
            let external_host_supported = extensions.iter().any(|extension| {
                let name = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
                name == vk::ExtExternalMemoryHostFn::name()
            });
            let mut external_host_properties =
                vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
            if external_host_supported {
                let mut properties2 = vk::PhysicalDeviceProperties2::builder()
                    .push_next(&mut external_host_properties)
                    .build();
                unsafe { instance.get_physical_device_properties2(physical, &mut properties2) };
            }

            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::builder()
                .push_next(&mut subgroup)
                .push_next(&mut maintenance3)
                .build();
            unsafe { instance.get_physical_device_properties2(physical, &mut properties2) };

            let dot_extension = has_extension(b"VK_KHR_shader_integer_dot_product");
            let dot_core = properties.api_version >= vk::API_VERSION_1_3;
            let integer_dot_product_supported = if dot_core || dot_extension {
                let mut dot_support = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
                let mut dot_features = vk::PhysicalDeviceFeatures2::builder()
                    .push_next(&mut dot_support)
                    .build();
                unsafe { instance.get_physical_device_features2(physical, &mut dot_features) };
                dot_support.shader_integer_dot_product != 0
            } else {
                false
            };
            let cooperative_matrix_supported = has_extension(b"VK_KHR_cooperative_matrix")
                || has_extension(b"VK_NV_cooperative_matrix");

            let memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical) };
            let memory_budget_supported = has_extension(b"VK_EXT_memory_budget");
            let mut memory_budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            if memory_budget_supported {
                let mut memory_properties2 = vk::PhysicalDeviceMemoryProperties2::builder()
                    .push_next(&mut memory_budget)
                    .build();
                unsafe {
                    instance
                        .get_physical_device_memory_properties2(physical, &mut memory_properties2)
                };
            }
            let mut device_local_memory_bytes = 0u64;
            let mut available_device_memory_bytes = 0u64;
            for index in 0..memory_properties.memory_heap_count as usize {
                let heap = memory_properties.memory_heaps[index];
                if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
                    device_local_memory_bytes = device_local_memory_bytes.saturating_add(heap.size);
                    let available = if memory_budget_supported {
                        memory_budget.heap_budget[index]
                            .saturating_sub(memory_budget.heap_usage[index])
                    } else {
                        heap.size
                    };
                    available_device_memory_bytes =
                        available_device_memory_bytes.saturating_add(available);
                }
            }
            let timestamp_supported =
                queue_families[queue_family as usize].timestamp_valid_bits > 0;
            let profile = DeviceProfile {
                device_name: device_name.clone(),
                vendor_id: properties.vendor_id,
                device_id: properties.device_id,
                driver_version: properties.driver_version,
                api_version: properties.api_version,
                queue_family,
                device_local_memory_bytes,
                available_device_memory_bytes,
                memory_budget_supported,
                max_storage_buffer_bytes: properties.limits.max_storage_buffer_range as u64,
                max_memory_allocation_bytes: maintenance3.max_memory_allocation_size,
                max_workgroup_invocations: properties.limits.max_compute_work_group_invocations,
                max_workgroup_size: properties.limits.max_compute_work_group_size,
                max_workgroup_count: properties.limits.max_compute_work_group_count,
                subgroup_size: subgroup.subgroup_size,
                fp16_supported: float_support.shader_float16 != 0
                    && storage_support.storage_buffer16_bit_access != 0,
                int8_supported: float_support.shader_int8 != 0,
                integer_dot_product_supported,
                cooperative_matrix_supported,
                storage_buffer_alignment: properties.limits.min_storage_buffer_offset_alignment,
                timestamp_supported,
                timestamp_period_nanoseconds: properties.limits.timestamp_period,
                external_host_memory_supported: external_host_supported,
            };
            if std::env::var("QUARTZ_DEBUG_ARENA_SIZE").is_ok() {
                eprintln!(
                    "Quartz Vulkan: maxMemoryAllocationSize={} bytes",
                    maintenance3.max_memory_allocation_size
                );
                for index in 0..memory_properties.memory_heap_count {
                    let heap = memory_properties.memory_heaps[index as usize];
                    eprintln!(
                        "Quartz Vulkan: heap[{index}] size={} bytes flags={:#x}",
                        heap.size,
                        heap.flags.as_raw()
                    );
                }
            }

            let enabled_extensions = external_host_supported
                .then(|| vk::ExtExternalMemoryHostFn::name().as_ptr())
                .into_iter()
                .collect::<Vec<_>>();
            let device_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&enabled_extensions)
                .push_next(&mut enable_storage)
                .push_next(&mut enable_float);
            let device = unsafe { instance.create_device(physical, &device_info, None) }
                .map_err(|error| anyhow!("vkCreateDevice failed: {error:?}"))?;
            Ok((
                physical,
                device,
                queue_family,
                profile,
                external_host_supported,
                external_host_properties.min_imported_host_pointer_alignment,
                properties.limits.min_storage_buffer_offset_alignment,
            ))
        })();
        let (
            physical,
            device,
            queue_family,
            mut profile,
            external_host_supported,
            external_host_alignment,
            storage_buffer_alignment,
        ) = match setup {
            Ok(value) => value,
            Err(error) => {
                unsafe { instance.destroy_instance(None) };
                return Err(error);
            }
        };
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let external_host_supported =
            external_host_supported && std::env::var("QUARTZ_VULKAN_NO_HOST_IMPORT").is_err();
        profile.external_host_memory_supported = external_host_supported;
        let device_name = profile.device_name.clone();
        let external_host = external_host_supported.then(|| {
            vk::ExtExternalMemoryHostFn::load(|name| {
                unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) }
                    .map_or(std::ptr::null(), |function| function as *const _)
            })
        });
        let mut runtime = Self {
            _entry: entry,
            instance,
            physical,
            device,
            queue_family,
            queue,
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            gemm_pipeline: vk::Pipeline::null(),
            conv2d_pipeline: vk::Pipeline::null(),
            attention_pipeline: vk::Pipeline::null(),
            im2col_pipeline: vk::Pipeline::null(),
            groupnorm_silu_pipeline: vk::Pipeline::null(),
            residual_add_pipeline: vk::Pipeline::null(),
            gemm_heads_pipeline: vk::Pipeline::null(),
            merge_heads_pipeline: vk::Pipeline::null(),
            geglu_pipeline: vk::Pipeline::null(),
            elementwise_pipeline: vk::Pipeline::null(),
            channel_rmsnorm_pipeline: vk::Pipeline::null(),
            resident_linear_pipeline: vk::Pipeline::null(),
            layernorm_pipeline: vk::Pipeline::null(),
            rmsnorm_pipeline: vk::Pipeline::null(),
            rope_pipeline: vk::Pipeline::null(),
            f32_attention_pipeline: vk::Pipeline::null(),
            patch_layout_pipeline: vk::Pipeline::null(),
            wan_head_modulate_pipeline: vk::Pipeline::null(),
            resident_conv3d_pipeline: vk::Pipeline::null(),
            resident_conv2d_pipeline: vk::Pipeline::null(),
            ncthw_temporal_pipeline: vk::Pipeline::null(),
            vae_spatial_layout_pipeline: vk::Pipeline::null(),
            descriptor_pool: vk::DescriptorPool::null(),
            command_pool: vk::CommandPool::null(),
            device_name,
            profile,
            stats: RuntimeStats::default(),
            buffers: std::array::from_fn(|_| None),
            resident_buffers: HashMap::new(),
            next_resident_id: 1,
            external_host,
            external_host_alignment,
            storage_buffer_alignment,
            model_mappings: HashMap::new(),
        };
        runtime.initialize_resources()?;
        eprintln!("{}", runtime.profile);
        Ok(runtime)
    }

    fn probe_external_host_pointer(&self, pointer: *const u8, bytes: usize) -> Result<()> {
        let external_host = self
            .external_host
            .as_ref()
            .context("Vulkan device does not expose VK_EXT_external_memory_host")?;
        let alignment = usize::try_from(self.external_host_alignment)
            .context("external-host alignment does not fit this platform")?;
        if alignment == 0 || !alignment.is_power_of_two() {
            bail!("Vulkan reported invalid external-host alignment {alignment}");
        }
        println!(
            "Vulkan host import: alignment={} pointer_mod={} bytes={} size_mod={}",
            alignment,
            pointer as usize % alignment,
            bytes,
            bytes % alignment,
        );
        if pointer as usize % alignment != 0 {
            bail!("mapped file base is not aligned for Vulkan host import");
        }
        for (label, handle_type) in [
            (
                "HOST_ALLOCATION",
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            ),
            (
                "HOST_MAPPED_FOREIGN_MEMORY",
                vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT,
            ),
        ] {
            let buffer_info = vk::PhysicalDeviceExternalBufferInfo::builder()
                .flags(vk::BufferCreateFlags::empty())
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .handle_type(handle_type);
            let mut buffer_properties = vk::ExternalBufferProperties::default();
            unsafe {
                self.instance
                    .get_physical_device_external_buffer_properties(
                        self.physical,
                        &buffer_info,
                        &mut buffer_properties,
                    )
            };
            let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
            let result = unsafe {
                (external_host.get_memory_host_pointer_properties_ext)(
                    self.device.handle(),
                    handle_type,
                    pointer.cast(),
                    &mut properties,
                )
            };
            println!(
                "Vulkan host import {label}: result={result:?} memory_type_bits=0x{:08x} external_features=0x{:08x} compatible_handles=0x{:08x}",
                properties.memory_type_bits,
                buffer_properties
                    .external_memory_properties
                    .external_memory_features
                    .as_raw(),
                buffer_properties
                    .external_memory_properties
                    .compatible_handle_types
                    .as_raw(),
            );
        }
        Ok(())
    }

    fn initialize_resources(&mut self) -> Result<()> {
        let bindings = [
            descriptor_binding(0),
            descriptor_binding(1),
            descriptor_binding(2),
            descriptor_binding(3),
        ];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        self.set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&set_layout_info, None)
        }
        .map_err(|error| anyhow!("descriptor layout failed: {error:?}"))?;

        let push_range = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(64)
            .build()];
        let layouts = [self.set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_range);
        self.pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
        }
        .map_err(|error| anyhow!("pipeline layout failed: {error:?}"))?;

        self.gemm_pipeline = self.create_pipeline(GEMM_SHADER)?;
        self.conv2d_pipeline = self.create_pipeline(CONV2D_SHADER)?;
        self.attention_pipeline = self.create_pipeline(ATTENTION_SHADER)?;
        self.im2col_pipeline = self.create_pipeline(IM2COL_SHADER)?;
        self.groupnorm_silu_pipeline = self.create_pipeline(GROUPNORM_SILU_SHADER)?;
        self.residual_add_pipeline = self.create_pipeline(RESIDUAL_ADD_SHADER)?;
        self.gemm_heads_pipeline = self.create_pipeline(GEMM_HEADS_SHADER)?;
        self.merge_heads_pipeline = self.create_pipeline(MERGE_HEADS_SHADER)?;
        self.geglu_pipeline = self.create_pipeline(GEGLU_SHADER)?;
        self.elementwise_pipeline = self.create_pipeline(ELEMENTWISE_SHADER)?;
        self.channel_rmsnorm_pipeline = self.create_pipeline(CHANNEL_RMSNORM_SHADER)?;
        self.resident_linear_pipeline = self.create_pipeline(RESIDENT_LINEAR_SHADER)?;
        self.layernorm_pipeline = self.create_pipeline(LAYERNORM_SHADER)?;
        self.rmsnorm_pipeline = self.create_pipeline(RMSNORM_SHADER)?;
        self.rope_pipeline = self.create_pipeline(ROPE_SHADER)?;
        self.f32_attention_pipeline = self.create_pipeline(F32_ATTENTION_SHADER)?;
        self.patch_layout_pipeline = self.create_pipeline(PATCH_LAYOUT_SHADER)?;
        self.wan_head_modulate_pipeline = self.create_pipeline(WAN_HEAD_MODULATE_SHADER)?;
        self.resident_conv3d_pipeline = self.create_pipeline(RESIDENT_CONV3D_SHADER)?;
        self.resident_conv2d_pipeline = self.create_pipeline(RESIDENT_CONV2D_SHADER)?;
        self.ncthw_temporal_pipeline = self.create_pipeline(NCTHW_TEMPORAL_SHADER)?;
        self.vae_spatial_layout_pipeline = self.create_pipeline(VAE_SPATIAL_LAYOUT_SHADER)?;

        let pool_size = [vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(128)
            .build()];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(32)
            .pool_sizes(&pool_size);
        self.descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }
            .map_err(|error| anyhow!("descriptor pool failed: {error:?}"))?;
        let command_pool_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(self.queue_family);
        self.command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None) }
            .map_err(|error| anyhow!("command pool failed: {error:?}"))?;
        Ok(())
    }

    fn create_pipeline(&self, spirv: &[u8]) -> Result<vk::Pipeline> {
        let mut cursor = Cursor::new(spirv);
        let shader_code = ash::util::read_spv(&mut cursor).context("invalid embedded SPIR-V")?;
        let shader_info = vk::ShaderModuleCreateInfo::builder().code(&shader_code);
        let shader = unsafe { self.device.create_shader_module(&shader_info, None) }
            .map_err(|error| anyhow!("shader module failed: {error:?}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(c"main")
            .build();
        let pipeline_info = [vk::ComputePipelineCreateInfo::builder()
            .stage(stage)
            .layout(self.pipeline_layout)
            .build()];
        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
        };
        unsafe { self.device.destroy_shader_module(shader, None) };
        pipeline
            .map_err(|(_, error)| anyhow!("compute pipeline failed: {error:?}"))
            .map(|pipelines| pipelines[0])
    }

    fn matmul(
        &mut self,
        input: &[u8],
        weights: DispatchInput<'_>,
        rows: u32,
        outputs: u32,
        width: u32,
    ) -> Result<(Vec<f32>, f64)> {
        let wall_started = Instant::now();
        if rows == 0 || outputs == 0 || width == 0 || width % 4 != 0 {
            bail!("Vulkan GEMM dimensions must be non-zero and width divisible by four");
        }
        let expected_input = rows as usize * width as usize * size_of::<u16>();
        let expected_weights = outputs as usize * width as usize * size_of::<u16>();
        if input.len() != expected_input || weights.len() != expected_weights {
            bail!("Vulkan GEMM payload length does not match its dimensions");
        }

        let result = self.dispatch(
            &[DispatchInput::Upload(input), weights],
            rows as usize * outputs as usize,
            self.gemm_pipeline,
            bytes_of(&[rows, outputs, width, 0, 0]),
            [outputs.div_ceil(32), rows.div_ceil(8), 1],
            KernelKind::Gemm,
            None,
        );
        self.stats.gemm.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn elementwise(
        &mut self,
        left: &[u8],
        right: &[u8],
        elements: u32,
        operation: u32,
        parameter0: f32,
        parameter1: f32,
    ) -> Result<(Vec<f32>, f64)> {
        let expected_bytes = (elements as usize)
            .checked_mul(size_of::<f32>())
            .context("Vulkan elementwise byte length overflow")?;
        if elements == 0 || left.len() != expected_bytes || right.len() != expected_bytes {
            bail!("Vulkan elementwise payload length does not match its dimensions");
        }
        self.dispatch(
            &[DispatchInput::Upload(left), DispatchInput::Upload(right)],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[
                elements,
                operation,
                parameter0.to_bits(),
                parameter1.to_bits(),
            ]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
            None,
        )
    }

    fn channel_rms_norm(
        &mut self,
        input: &[u8],
        weight: &[u8],
        dimensions: [u32; 4],
        output_len: usize,
    ) -> Result<(Vec<f32>, f64)> {
        let locations = dimensions[0]
            .checked_mul(dimensions[2])
            .context("Vulkan RMSNorm location count overflow")?;
        self.dispatch(
            &[DispatchInput::Upload(input), DispatchInput::Upload(weight)],
            output_len,
            self.channel_rmsnorm_pipeline,
            bytes_of(&dimensions),
            [locations, 1, 1],
            KernelKind::Norm,
            None,
        )
    }

    fn bias_add(
        &mut self,
        input: &[u8],
        bias: &[u8],
        elements: u32,
        width: u32,
    ) -> Result<(Vec<f32>, f64)> {
        let input_bytes = elements as usize * size_of::<f32>();
        let bias_bytes = width as usize * size_of::<f32>();
        if elements == 0 || width == 0 || input.len() != input_bytes || bias.len() != bias_bytes {
            bail!("Vulkan bias-add payload length does not match its dimensions");
        }
        self.dispatch(
            &[DispatchInput::Upload(input), DispatchInput::Upload(bias)],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 6, width, 0]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
            None,
        )
    }

    fn upload_resident(
        &mut self,
        bytes: &[u8],
        elements: usize,
        element_type: ResidentElementType,
        class: ResidentClass,
    ) -> Result<u64> {
        let expected_bytes = elements
            .checked_mul(match element_type {
                ResidentElementType::F16 => size_of::<u16>(),
                ResidentElementType::F32 => size_of::<f32>(),
            })
            .context("resident Vulkan upload size overflow")?;
        if elements == 0 || bytes.len() != expected_bytes {
            bail!("resident Vulkan upload payload does not match its element type");
        }
        let buffer = Buffer::new(&self.instance, self.physical, &self.device, bytes.len())?;
        buffer.write_bytes(bytes)?;
        self.stats.uploaded_bytes = self.stats.uploaded_bytes.saturating_add(bytes.len() as u64);
        Ok(self.insert_resident(buffer, bytes.len(), elements, element_type, class))
    }

    fn stage_device_local_buffers(&mut self, payloads: &[&[u8]]) -> Result<Vec<Buffer>> {
        if payloads.is_empty() || payloads.iter().any(|payload| payload.is_empty()) {
            bail!("device-local staging requires at least one non-empty payload");
        }
        let staging_bytes = payloads.iter().try_fold(0usize, |total, payload| {
            total.checked_add(payload.len().next_multiple_of(4))
        });
        let staging_bytes = staging_bytes.context("device-local staging size overflow")?;
        let staging =
            Buffer::new_staging(&self.instance, self.physical, &self.device, staging_bytes)?;
        let mut destinations = Vec::with_capacity(payloads.len());
        let mut offsets = Vec::with_capacity(payloads.len());
        let mut copy_bytes = Vec::with_capacity(payloads.len());
        let mut offset = 0usize;
        let zero_padding = [0u8; 3];
        for payload in payloads {
            let aligned_bytes = payload.len().next_multiple_of(4);
            staging.write_bytes_at(offset, payload)?;
            let padding_bytes = aligned_bytes - payload.len();
            if padding_bytes > 0 {
                staging.write_bytes_at(offset + payload.len(), &zero_padding[..padding_bytes])?;
            }
            destinations.push(Buffer::new_device_local_storage(
                &self.instance,
                self.physical,
                &self.device,
                aligned_bytes,
            )?);
            offsets.push(offset);
            copy_bytes.push(aligned_bytes);
            offset += aligned_bytes;
        }

        unsafe {
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|error| {
                    anyhow!("resident staging command-pool reset failed: {error:?}")
                })?;
        }
        let command_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&command_info) }
            .map_err(|error| anyhow!("resident staging command allocation failed: {error:?}"))?[0];
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(command, &begin_info)
                .map_err(|error| anyhow!("resident staging command begin failed: {error:?}"))?;
            for ((destination, &source_offset), &bytes) in
                destinations.iter().zip(&offsets).zip(&copy_bytes)
            {
                self.device.cmd_copy_buffer(
                    command,
                    staging.buffer,
                    destination.buffer,
                    &[vk::BufferCopy {
                        src_offset: source_offset as u64,
                        dst_offset: 0,
                        size: bytes as u64,
                    }],
                );
            }
            let barriers = destinations
                .iter()
                .zip(&copy_bytes)
                .map(|(destination, &bytes)| {
                    vk::BufferMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(destination.buffer)
                        .offset(0)
                        .size(bytes as u64)
                        .build()
                })
                .collect::<Vec<_>>();
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
            self.device
                .end_command_buffer(command)
                .map_err(|error| anyhow!("resident staging command end failed: {error:?}"))?;
        }
        let commands = [command];
        let submit = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
        unsafe {
            self.device
                .queue_submit(self.queue, &submit, vk::Fence::null())
                .map_err(|error| anyhow!("resident staging queue submit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| anyhow!("resident staging queue wait failed: {error:?}"))?;
        }
        Ok(destinations)
    }

    fn prepare_resident_linear(
        &mut self,
        weight: &[u8],
        weight_elements: usize,
        bias: &[u8],
        bias_elements: usize,
    ) -> Result<(u64, u64)> {
        if weight_elements == 0
            || weight.len() != weight_elements.saturating_mul(size_of::<u16>())
            || bias_elements == 0
            || bias.len() != bias_elements.saturating_mul(size_of::<f32>())
        {
            bail!("resident linear payload does not match its element counts");
        }
        let mut buffers = self.stage_device_local_buffers(&[weight, bias])?;
        let bias_buffer = buffers.pop().expect("two staged buffers were requested");
        let weight_buffer = buffers.pop().expect("two staged buffers were requested");

        let weight_id = self.insert_resident(
            weight_buffer,
            weight.len(),
            weight_elements,
            ResidentElementType::F16,
            ResidentClass::Weight,
        );
        let bias_id = self.insert_resident(
            bias_buffer,
            bias.len(),
            bias_elements,
            ResidentElementType::F32,
            ResidentClass::Weight,
        );
        let uploaded = weight.len().saturating_add(bias.len()) as u64;
        self.stats.uploaded_bytes = self.stats.uploaded_bytes.saturating_add(uploaded);
        self.stats.resident_weight_uploads += 1;
        self.stats.resident_uploaded_bytes =
            self.stats.resident_uploaded_bytes.saturating_add(uploaded);
        self.stats.cached_weight_bytes = self.stats.cached_weight_bytes.saturating_add(uploaded);
        Ok((weight_id, bias_id))
    }

    fn prepare_resident_vector(&mut self, vector: &[u8], elements: usize) -> Result<u64> {
        if elements == 0 || vector.len() != elements.saturating_mul(size_of::<f32>()) {
            bail!("resident vector payload does not match its element count");
        }
        let mut buffers = self.stage_device_local_buffers(&[vector])?;
        let buffer = buffers.pop().expect("one staged buffer was requested");
        let id = self.insert_resident(
            buffer,
            vector.len(),
            elements,
            ResidentElementType::F32,
            ResidentClass::Weight,
        );
        let uploaded = vector.len() as u64;
        self.stats.uploaded_bytes = self.stats.uploaded_bytes.saturating_add(uploaded);
        self.stats.resident_weight_uploads += 1;
        self.stats.resident_uploaded_bytes =
            self.stats.resident_uploaded_bytes.saturating_add(uploaded);
        self.stats.cached_weight_bytes = self.stats.cached_weight_bytes.saturating_add(uploaded);
        Ok(id)
    }

    fn resident_linear(
        &mut self,
        input_id: u64,
        weight_id: u64,
        bias_id: u64,
        rows: u32,
        outputs: u32,
        width: u32,
    ) -> Result<u64> {
        if rows == 0 || outputs == 0 || width == 0 || width % 4 != 0 {
            bail!("resident Vulkan GEMM dimensions must be non-zero and width divisible by four");
        }
        self.require_resident(
            input_id,
            rows as usize * width as usize,
            ResidentElementType::F32,
        )?;
        self.require_resident(
            weight_id,
            outputs as usize * width as usize,
            ResidentElementType::F16,
        )?;
        self.require_resident(bias_id, outputs as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, weight_id, bias_id],
            rows as usize * outputs as usize,
            self.resident_linear_pipeline,
            bytes_of(&[rows, outputs, width, 0]),
            [outputs.div_ceil(32), rows.div_ceil(8), 1],
            KernelKind::Gemm,
        );
        self.stats.gemm.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_conv3d(
        &mut self,
        input_id: u64,
        weight_id: u64,
        bias_id: u64,
        dimensions: &[u32; 15],
        output_elements: usize,
    ) -> Result<u64> {
        let [
            batch,
            input_channels,
            input_time,
            input_height,
            input_width,
            output_channels,
            kernel_time,
            kernel_height,
            kernel_width,
            _,
            _,
            _,
            output_time,
            output_height,
            output_width,
        ] = *dimensions;
        if [
            batch,
            input_channels,
            input_time,
            input_height,
            input_width,
            output_channels,
            kernel_time,
            kernel_height,
            kernel_width,
            output_time,
            output_height,
            output_width,
        ]
        .contains(&0)
        {
            bail!("resident Vulkan Conv3D dimensions must be non-zero");
        }
        let input_elements = batch as usize
            * input_channels as usize
            * input_time as usize
            * input_height as usize
            * input_width as usize;
        let weight_elements = output_channels as usize
            * input_channels as usize
            * kernel_time as usize
            * kernel_height as usize
            * kernel_width as usize;
        let expected_output = batch as usize
            * output_channels as usize
            * output_time as usize
            * output_height as usize
            * output_width as usize;
        if output_elements != expected_output {
            bail!("resident Vulkan Conv3D output element count is inconsistent");
        }
        self.require_resident(input_id, input_elements, ResidentElementType::F32)?;
        self.require_resident(weight_id, weight_elements, ResidentElementType::F16)?;
        self.require_resident(bias_id, output_channels as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, weight_id, bias_id],
            output_elements,
            self.resident_conv3d_pipeline,
            bytes_of(dimensions),
            [
                u32::try_from(output_elements)
                    .context("resident Vulkan Conv3D output exceeds u32")?
                    .div_ceil(256),
                1,
                1,
            ],
            KernelKind::Conv2d,
        );
        self.stats.conv2d.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_conv2d(
        &mut self,
        input_id: u64,
        weight_id: u64,
        bias_id: u64,
        dimensions: &[u32; 13],
        output_elements: usize,
    ) -> Result<u64> {
        let [
            batch,
            input_channels,
            input_height,
            input_width,
            output_channels,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            _,
            _,
            output_height,
            output_width,
        ] = *dimensions;
        if [
            batch,
            input_channels,
            input_height,
            input_width,
            output_channels,
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            output_height,
            output_width,
        ]
        .contains(&0)
        {
            bail!("resident Vulkan Conv2D dimensions must be non-zero");
        }
        let input_elements =
            batch as usize * input_channels as usize * input_height as usize * input_width as usize;
        let weight_elements = output_channels as usize
            * input_channels as usize
            * kernel_height as usize
            * kernel_width as usize;
        let expected_output = batch as usize
            * output_channels as usize
            * output_height as usize
            * output_width as usize;
        if output_elements != expected_output {
            bail!("resident Vulkan Conv2D output element count is inconsistent");
        }
        self.require_resident(input_id, input_elements, ResidentElementType::F32)?;
        self.require_resident(weight_id, weight_elements, ResidentElementType::F16)?;
        self.require_resident(bias_id, output_channels as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, weight_id, bias_id],
            output_elements,
            self.resident_conv2d_pipeline,
            bytes_of(dimensions),
            [
                u32::try_from(output_elements)
                    .context("resident Vulkan Conv2D output exceeds u32")?
                    .div_ceil(256),
                1,
                1,
            ],
            KernelKind::Conv2d,
        );
        self.stats.conv2d.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_vae_spatial_layout(
        &mut self,
        input_id: u64,
        input_elements: usize,
        parameters: &[u32; 8],
    ) -> Result<u64> {
        let [
            operation,
            batch,
            channels,
            time,
            height,
            width,
            chunk,
            output_elements,
        ] = *parameters;
        if operation > 4
            || batch == 0
            || channels == 0
            || time == 0
            || height == 0
            || width == 0
            || (operation == 4 && chunk == 0)
            || (operation != 4 && chunk > 2)
            || output_elements == 0
        {
            bail!("resident VAE spatial layout dimensions are invalid");
        }
        let base_elements = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(time))
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .context("resident VAE spatial layout base size overflow")?;
        let expected_input = if operation == 2 {
            base_elements
                .checked_mul(3)
                .context("resident VAE QKV layout input size overflow")?
        } else {
            base_elements
        };
        let expected_output = if operation == 4 {
            base_elements
                .checked_mul(chunk)
                .and_then(|value| value.checked_mul(chunk))
                .context("resident VAE nearest upsample output size overflow")?
        } else {
            base_elements
        };
        if input_elements != expected_input as usize
            || output_elements != expected_output
            || (operation != 2 && operation != 4 && chunk != 0)
            || (operation == 2 && chunk > 2)
        {
            bail!("resident VAE spatial layout element count is inconsistent");
        }
        self.require_resident(input_id, input_elements, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id],
            output_elements as usize,
            self.vae_spatial_layout_pipeline,
            bytes_of(parameters),
            [output_elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_ncthw_temporal(
        &mut self,
        input0_id: u64,
        input1_id: u64,
        input0_elements: usize,
        input1_elements: usize,
        parameters: &[u32; 11],
    ) -> Result<u64> {
        let [
            operation,
            output_elements,
            batch,
            output_channels,
            output_time,
            height,
            width,
            input0_channels,
            input0_time,
            input1_time,
            parameter0,
        ] = *parameters;
        if operation > 3
            || [
                output_elements,
                batch,
                output_channels,
                output_time,
                height,
                width,
                input0_channels,
                input0_time,
            ]
            .contains(&0)
        {
            bail!("resident NCTHW temporal parameters are invalid");
        }
        let expected_input0 = batch as usize
            * input0_channels as usize
            * input0_time as usize
            * height as usize
            * width as usize;
        let expected_output = batch as usize
            * output_channels as usize
            * output_time as usize
            * height as usize
            * width as usize;
        if input0_elements != expected_input0 || output_elements as usize != expected_output {
            bail!("resident NCTHW temporal element counts are inconsistent");
        }
        match operation {
            0 => {
                if parameter0 as usize + output_time as usize > input0_time as usize
                    || output_channels != input0_channels
                {
                    bail!("resident NCTHW temporal slice is out of bounds");
                }
            }
            1 => {
                let expected_input1 = batch as usize
                    * input0_channels as usize
                    * input1_time as usize
                    * height as usize
                    * width as usize;
                if input1_time == 0
                    || input1_elements != expected_input1
                    || output_channels != input0_channels
                    || output_time != input0_time + input1_time
                {
                    bail!("resident NCTHW temporal concat dimensions are inconsistent");
                }
            }
            2 => {
                if parameter0 == 0
                    || output_channels != input0_channels
                    || output_time != input0_time + parameter0
                {
                    bail!("resident NCTHW zero-time prepend dimensions are inconsistent");
                }
            }
            3 => {
                if input0_channels % 2 != 0
                    || output_channels * 2 != input0_channels
                    || output_time != input0_time * 2
                {
                    bail!("resident Wan channel-to-time dimensions are inconsistent");
                }
            }
            _ => unreachable!(),
        }
        self.require_resident(input0_id, input0_elements, ResidentElementType::F32)?;
        self.require_resident(input1_id, input1_elements, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input0_id, input1_id],
            output_elements as usize,
            self.ncthw_temporal_pipeline,
            bytes_of(parameters),
            [output_elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_silu(&mut self, input_id: u64, elements: u32) -> Result<u64> {
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, input_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 3, 0, 0]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_gelu_tanh(&mut self, input_id: u64, elements: u32) -> Result<u64> {
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, input_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 4, 0, 0]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_scale(&mut self, input_id: u64, elements: u32, value: f32) -> Result<u64> {
        if !value.is_finite() {
            bail!("resident Vulkan scale must be finite");
        }
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, input_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 2, value.to_bits(), 0]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_affine(
        &mut self,
        input_id: u64,
        elements: u32,
        scale: f32,
        bias: f32,
    ) -> Result<u64> {
        if elements == 0 || !scale.is_finite() || !bias.is_finite() {
            bail!("resident Vulkan affine parameters must be valid");
        }
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, input_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 9, scale.to_bits(), bias.to_bits()]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_ncthw_channel_affine(
        &mut self,
        input_id: u64,
        parameter_id: u64,
        elements: u32,
        channels: u32,
        channel_plane: u32,
    ) -> Result<u64> {
        if elements == 0
            || channels == 0
            || channel_plane == 0
            || channels
                .checked_mul(channel_plane)
                .is_none_or(|sample_elements| elements % sample_elements != 0)
        {
            bail!("resident NCTHW channel affine dimensions are invalid");
        }
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        self.require_resident(
            parameter_id,
            2 * channels as usize,
            ResidentElementType::F32,
        )?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, parameter_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 10, channel_plane, channels]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_clamp(
        &mut self,
        input_id: u64,
        elements: u32,
        minimum: f32,
        maximum: f32,
    ) -> Result<u64> {
        if elements == 0 || !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            bail!("resident Vulkan clamp parameters must be valid");
        }
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, input_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 5, minimum.to_bits(), maximum.to_bits()]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn resident_patch_layout(
        &mut self,
        input_id: u64,
        operation: u32,
        elements: u32,
        channels: u32,
        time: u32,
        height: u32,
        width: u32,
        patch_time: u32,
        patch_height: u32,
        patch_width: u32,
        output_channels: u32,
    ) -> Result<u64> {
        if operation > 1
            || elements == 0
            || time == 0
            || height == 0
            || width == 0
            || patch_time == 0
            || patch_height == 0
            || patch_width == 0
            || time % patch_time != 0
            || height % patch_height != 0
            || width % patch_width != 0
            || (operation == 0 && channels == 0)
            || (operation == 1 && output_channels == 0)
        {
            bail!("resident patch-layout dimensions or operation are invalid");
        }
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id],
            elements as usize,
            self.patch_layout_pipeline,
            bytes_of(&[
                operation,
                elements,
                channels,
                time,
                height,
                width,
                patch_time,
                patch_height,
                patch_width,
                output_channels,
            ]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_wan_head_modulate(
        &mut self,
        input_id: u64,
        timestep_id: u64,
        modulation_id: u64,
        elements: u32,
        width: u32,
    ) -> Result<u64> {
        if elements == 0 || width == 0 || elements % width != 0 {
            bail!("resident Wan head modulation dimensions are invalid");
        }
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        self.require_resident(timestep_id, width as usize, ResidentElementType::F32)?;
        self.require_resident(modulation_id, 2 * width as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, timestep_id, modulation_id],
            elements as usize,
            self.wan_head_modulate_pipeline,
            bytes_of(&[elements, width]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_add_vector(&mut self, input_id: u64, vector_id: u64, elements: u32) -> Result<u64> {
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        self.require_resident(vector_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, vector_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 0, 0, 0]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_multiply(&mut self, left_id: u64, right_id: u64, elements: u32) -> Result<u64> {
        self.require_resident(left_id, elements as usize, ResidentElementType::F32)?;
        self.require_resident(right_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[left_id, right_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 1, 0, 0]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_layer_norm(
        &mut self,
        input_id: u64,
        affine_ids: Option<(u64, u64)>,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<u64> {
        let elements = rows
            .checked_mul(width)
            .context("resident LayerNorm element count overflow")?;
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let (weight_id, bias_id, affine) = if let Some((weight_id, bias_id)) = affine_ids {
            self.require_resident(weight_id, width as usize, ResidentElementType::F32)?;
            self.require_resident(bias_id, width as usize, ResidentElementType::F32)?;
            (weight_id, bias_id, 1)
        } else {
            (input_id, input_id, 0)
        };
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, weight_id, bias_id],
            elements as usize,
            self.layernorm_pipeline,
            bytes_of(&[rows, width, epsilon.to_bits(), affine]),
            [rows, 1, 1],
            KernelKind::Norm,
        );
        self.stats.norm.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_rms_norm(
        &mut self,
        input_id: u64,
        weight_id: u64,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<u64> {
        let elements = rows
            .checked_mul(width)
            .context("resident RMSNorm element count overflow")?;
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        self.require_resident(weight_id, width as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, weight_id],
            elements as usize,
            self.rmsnorm_pipeline,
            bytes_of(&[rows, width, epsilon.to_bits(), 0]),
            [rows, 1, 1],
            KernelKind::Norm,
        );
        self.stats.norm.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_channel_rms_norm(
        &mut self,
        input_id: u64,
        weight_id: u64,
        dimensions: [u32; 4],
        elements: usize,
    ) -> Result<u64> {
        let [batch, channels, volume, _] = dimensions;
        let expected = batch as usize * channels as usize * volume as usize;
        if batch == 0 || channels == 0 || volume == 0 || elements != expected {
            bail!("resident channel RMSNorm dimensions are invalid");
        }
        self.require_resident(input_id, elements, ResidentElementType::F32)?;
        self.require_resident(weight_id, channels as usize, ResidentElementType::F32)?;
        let locations = batch
            .checked_mul(volume)
            .context("resident channel RMSNorm location count overflow")?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, weight_id],
            elements,
            self.channel_rmsnorm_pipeline,
            bytes_of(&dimensions),
            [locations, 1, 1],
            KernelKind::Norm,
        );
        self.stats.norm.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_rope(
        &mut self,
        input_id: u64,
        position_id: u64,
        rows: u32,
        heads: u32,
        head_dim: u32,
    ) -> Result<u64> {
        if rows == 0 || heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
            bail!("resident RoPE dimensions must be non-zero with an even head dimension");
        }
        let width = heads
            .checked_mul(head_dim)
            .context("resident RoPE width overflow")?;
        let elements = rows
            .checked_mul(width)
            .context("resident RoPE element count overflow")?;
        let position_elements = rows
            .checked_mul(head_dim / 2)
            .and_then(|values| values.checked_mul(4))
            .context("resident RoPE position count overflow")?;
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        self.require_resident(
            position_id,
            position_elements as usize,
            ResidentElementType::F32,
        )?;
        let pair_count = elements / 2;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, position_id],
            elements as usize,
            self.rope_pipeline,
            bytes_of(&[rows, heads, head_dim, 0]),
            [pair_count.div_ceil(256), 1, 1],
            KernelKind::Attention,
        );
        self.stats.attention.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_attention_scores(
        &mut self,
        query_id: u64,
        key_id: u64,
        queries: u32,
        keys: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
        batches: u32,
    ) -> Result<u64> {
        if queries == 0
            || keys == 0
            || heads == 0
            || head_dim == 0
            || batches == 0
            || !scale.is_finite()
        {
            bail!("resident attention score dimensions and scale must be valid");
        }
        let width = heads
            .checked_mul(head_dim)
            .context("resident attention score width overflow")?;
        self.require_resident(
            query_id,
            batches as usize * queries as usize * width as usize,
            ResidentElementType::F32,
        )?;
        self.require_resident(
            key_id,
            batches as usize * keys as usize * width as usize,
            ResidentElementType::F32,
        )?;
        let output_elements = batches
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(queries))
            .and_then(|values| values.checked_mul(keys))
            .context("resident attention score count overflow")?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[query_id, key_id],
            output_elements as usize,
            self.f32_attention_pipeline,
            bytes_of(&[
                0,
                queries,
                keys,
                heads,
                head_dim,
                scale.to_bits(),
                batches,
                0,
            ]),
            [output_elements.div_ceil(256), 1, 1],
            KernelKind::Attention,
        );
        self.stats.attention.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_softmax(&mut self, input_id: u64, rows: u32, width: u32) -> Result<u64> {
        if rows == 0 || width == 0 {
            bail!("resident softmax dimensions must be non-zero");
        }
        let elements = rows
            .checked_mul(width)
            .context("resident softmax element count overflow")?;
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, input_id],
            elements as usize,
            self.f32_attention_pipeline,
            bytes_of(&[1, rows, width, 1, 1, 0, 1, 0]),
            [rows, 1, 1],
            KernelKind::Attention,
        );
        self.stats.attention.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_attention_values(
        &mut self,
        probability_id: u64,
        value_id: u64,
        queries: u32,
        keys: u32,
        heads: u32,
        head_dim: u32,
        batches: u32,
    ) -> Result<u64> {
        if queries == 0 || keys == 0 || heads == 0 || head_dim == 0 || batches == 0 {
            bail!("resident attention value dimensions must be non-zero");
        }
        let width = heads
            .checked_mul(head_dim)
            .context("resident attention value width overflow")?;
        let probability_elements = batches
            .checked_mul(heads)
            .and_then(|values| values.checked_mul(queries))
            .and_then(|values| values.checked_mul(keys))
            .context("resident attention probability count overflow")?;
        let output_elements = batches
            .checked_mul(queries)
            .and_then(|values| values.checked_mul(width))
            .context("resident attention output count overflow")?;
        self.require_resident(
            probability_id,
            probability_elements as usize,
            ResidentElementType::F32,
        )?;
        self.require_resident(
            value_id,
            batches as usize * keys as usize * width as usize,
            ResidentElementType::F32,
        )?;
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[probability_id, value_id],
            output_elements as usize,
            self.f32_attention_pipeline,
            bytes_of(&[2, queries, keys, heads, head_dim, 0, batches, 0]),
            [output_elements.div_ceil(256), 1, 1],
            KernelKind::Attention,
        );
        self.stats.attention.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_wan_modulate(
        &mut self,
        input_id: u64,
        modulation_id: u64,
        elements: u32,
        width: u32,
        shift_chunk: u32,
        scale_chunk: u32,
    ) -> Result<u64> {
        if width == 0 || elements == 0 || elements % width != 0 {
            bail!("resident Wan modulation dimensions are invalid");
        }
        if shift_chunk > u16::MAX.into() || scale_chunk > u16::MAX.into() {
            bail!("resident Wan modulation chunk index exceeds u16");
        }
        let required_chunks = shift_chunk.max(scale_chunk) + 1;
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let modulation = self
            .resident_buffers
            .get(&modulation_id)
            .with_context(|| format!("unknown resident Vulkan buffer {modulation_id}"))?;
        if modulation.element_type != ResidentElementType::F32
            || modulation.elements < required_chunks as usize * width as usize
        {
            bail!("resident Wan modulation buffer is too small or has the wrong type");
        }
        let packed_chunks = shift_chunk | (scale_chunk << 16);
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, modulation_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 7, width, packed_chunks]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn resident_multiply_vector_chunk(
        &mut self,
        input_id: u64,
        vector_id: u64,
        elements: u32,
        width: u32,
        chunk: u32,
    ) -> Result<u64> {
        if elements == 0 || width == 0 || elements % width != 0 {
            bail!("resident vector multiply dimensions are invalid");
        }
        self.require_resident(input_id, elements as usize, ResidentElementType::F32)?;
        let vector = self
            .resident_buffers
            .get(&vector_id)
            .with_context(|| format!("unknown resident Vulkan buffer {vector_id}"))?;
        let required_values = (chunk as usize + 1).saturating_mul(width as usize);
        if vector.element_type != ResidentElementType::F32 || vector.elements < required_values {
            bail!("resident vector multiply buffer is too small or has the wrong type");
        }
        let wall_started = Instant::now();
        let result = self.dispatch_resident(
            &[input_id, vector_id],
            elements as usize,
            self.elementwise_pipeline,
            bytes_of(&[elements, 8, width, chunk]),
            [elements.div_ceil(256), 1, 1],
            KernelKind::Elementwise,
        );
        self.stats.elementwise.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn require_resident(
        &self,
        id: u64,
        elements: usize,
        element_type: ResidentElementType,
    ) -> Result<()> {
        let resident = self
            .resident_buffers
            .get(&id)
            .with_context(|| format!("unknown resident Vulkan buffer {id}"))?;
        if resident.elements != elements || resident.element_type != element_type {
            bail!(
                "resident Vulkan buffer {id} has {:?}/{} elements, expected {:?}/{elements}",
                resident.element_type,
                resident.elements,
                element_type
            );
        }
        Ok(())
    }

    fn resident_is_device_local(&self, id: u64) -> Result<bool> {
        let resident = self
            .resident_buffers
            .get(&id)
            .with_context(|| format!("unknown resident Vulkan buffer {id}"))?;
        Ok(resident.buffer.memory_class == BufferMemoryClass::DeviceLocal)
    }

    fn download_resident(&mut self, id: u64, elements: usize) -> Result<Vec<f32>> {
        self.require_resident(id, elements, ResidentElementType::F32)?;
        let values = self
            .resident_buffers
            .get(&id)
            .expect("resident buffer was just validated")
            .buffer
            .read_f32(elements)?;
        let bytes = elements * size_of::<f32>();
        self.stats.resident_downloads += 1;
        self.stats.resident_downloaded_bytes = self
            .stats
            .resident_downloaded_bytes
            .saturating_add(bytes as u64);
        Ok(values)
    }

    fn insert_resident(
        &mut self,
        buffer: Buffer,
        logical_bytes: usize,
        elements: usize,
        element_type: ResidentElementType,
        class: ResidentClass,
    ) -> u64 {
        let id = self.next_resident_id;
        self.next_resident_id = self
            .next_resident_id
            .checked_add(1)
            .expect("resident Vulkan buffer identifier space exhausted");
        self.stats.resident_allocated_bytes = self
            .stats
            .resident_allocated_bytes
            .saturating_add(logical_bytes as u64);
        self.stats.peak_resident_allocated_bytes = self
            .stats
            .peak_resident_allocated_bytes
            .max(self.stats.resident_allocated_bytes);
        if buffer.memory_class == BufferMemoryClass::DeviceLocal {
            self.stats.resident_device_local_bytes = self
                .stats
                .resident_device_local_bytes
                .saturating_add(logical_bytes as u64);
            self.stats.peak_resident_device_local_bytes = self
                .stats
                .peak_resident_device_local_bytes
                .max(self.stats.resident_device_local_bytes);
            self.stats.resident_device_local_allocation_bytes = self
                .stats
                .resident_device_local_allocation_bytes
                .saturating_add(buffer.allocation_bytes);
            self.stats.peak_resident_device_local_allocation_bytes = self
                .stats
                .peak_resident_device_local_allocation_bytes
                .max(self.stats.resident_device_local_allocation_bytes);
        }
        let previous = self.resident_buffers.insert(
            id,
            ResidentBuffer {
                buffer,
                logical_bytes,
                elements,
                element_type,
                class,
            },
        );
        debug_assert!(previous.is_none());
        id
    }

    fn release_resident(&mut self, id: u64) {
        let Some(resident) = self.resident_buffers.remove(&id) else {
            return;
        };
        self.stats.resident_allocated_bytes = self
            .stats
            .resident_allocated_bytes
            .saturating_sub(resident.logical_bytes as u64);
        if resident.buffer.memory_class == BufferMemoryClass::DeviceLocal {
            self.stats.resident_device_local_bytes = self
                .stats
                .resident_device_local_bytes
                .saturating_sub(resident.logical_bytes as u64);
            self.stats.resident_device_local_allocation_bytes = self
                .stats
                .resident_device_local_allocation_bytes
                .saturating_sub(resident.buffer.allocation_bytes);
        }
        if resident.class == ResidentClass::Weight {
            self.stats.cached_weight_bytes = self
                .stats
                .cached_weight_bytes
                .saturating_sub(resident.logical_bytes as u64);
        }
    }

    fn dispatch_resident(
        &mut self,
        input_ids: &[u64],
        output_len: usize,
        pipeline: vk::Pipeline,
        push_constants: &[u8],
        groups: [u32; 3],
        kind: KernelKind,
    ) -> Result<u64> {
        if input_ids.is_empty() || input_ids.len() > 3 {
            bail!("resident Vulkan dispatch requires between one and three inputs");
        }
        if output_len == 0 || groups.contains(&0) {
            bail!("resident Vulkan dispatch dimensions must be non-zero");
        }
        if push_constants.len() > 64 || push_constants.len() % 4 != 0 {
            bail!("invalid resident Vulkan push-constant payload");
        }

        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|error| anyhow!("descriptor pool reset failed: {error:?}"))?;
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|error| anyhow!("command pool reset failed: {error:?}"))?;
        }

        let mut input_descriptors = Vec::with_capacity(input_ids.len());
        let mut unique_input_bytes = 0usize;
        let mut seen_ids = Vec::with_capacity(input_ids.len());
        for &id in input_ids {
            let resident = self
                .resident_buffers
                .get(&id)
                .with_context(|| format!("unknown resident Vulkan buffer {id}"))?;
            input_descriptors.push(resident.buffer.descriptor(resident.logical_bytes));
            if !seen_ids.contains(&id) {
                seen_ids.push(id);
                unique_input_bytes = unique_input_bytes.saturating_add(resident.logical_bytes);
            }
        }

        let output_bytes = output_len
            .checked_mul(size_of::<f32>())
            .context("resident Vulkan output byte size overflow")?;
        let output = Buffer::new(&self.instance, self.physical, &self.device, output_bytes)?;
        let layouts = [self.set_layout];
        let set_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&set_info) }
            .map_err(|error| anyhow!("resident descriptor allocation failed: {error:?}"))?[0];
        let mut buffer_infos = input_descriptors
            .iter()
            .copied()
            .map(|descriptor| [descriptor])
            .collect::<Vec<_>>();
        buffer_infos.push([output.descriptor(output_bytes)]);
        let writes = (0..buffer_infos.len())
            .map(|binding| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_infos[binding])
                    .build()
            })
            .collect::<Vec<_>>();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let command_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&command_info) }
            .map_err(|error| anyhow!("resident command allocation failed: {error:?}"))?[0];
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .map_err(|error| anyhow!("resident command begin failed: {error:?}"))?;
            self.device
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
            self.device
                .cmd_dispatch(command, groups[0], groups[1], groups[2]);
            self.device
                .end_command_buffer(command)
                .map_err(|error| anyhow!("resident command end failed: {error:?}"))?;
        }

        let commands = [command];
        let submit = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
        let started = Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &submit, vk::Fence::null())
                .map_err(|error| anyhow!("resident queue submit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| anyhow!("resident queue wait failed: {error:?}"))?;
        }
        let milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
        self.stats.peak_dispatch_bytes = self
            .stats
            .peak_dispatch_bytes
            .max(unique_input_bytes.saturating_add(output_bytes) as u64);
        let kernel = match kind {
            KernelKind::Elementwise => &mut self.stats.elementwise,
            KernelKind::Norm => &mut self.stats.norm,
            KernelKind::Gemm => &mut self.stats.gemm,
            KernelKind::Conv2d => &mut self.stats.conv2d,
            KernelKind::Attention => &mut self.stats.attention,
        };
        kernel.calls += 1;
        kernel.dispatch_milliseconds += milliseconds;
        Ok(self.insert_resident(
            output,
            output_bytes,
            output_len,
            ResidentElementType::F32,
            ResidentClass::Activation,
        ))
    }

    fn conv2d(
        &mut self,
        input: &[u8],
        weights: &MappedTensor,
        dimensions: &[u32; 13],
        output_len: usize,
    ) -> Result<(Vec<f32>, f64)> {
        let wall_started = Instant::now();
        let output_plane = dimensions[11]
            .checked_mul(dimensions[12])
            .context("Vulkan convolution output plane overflow")?;
        let reduction_width = dimensions[1]
            .checked_mul(dimensions[5])
            .and_then(|value| value.checked_mul(dimensions[6]))
            .context("Vulkan convolution reduction width overflow")?;
        let rows = dimensions[0]
            .checked_mul(output_plane)
            .context("Vulkan convolution row count overflow")?;
        let operations = u64::from(rows)
            .saturating_mul(u64::from(reduction_width))
            .saturating_mul(u64::from(dimensions[4]));
        let result = if reduction_width % 4 == 0 && operations >= 50_000_000 {
            self.conv2d_im2col(input, weights, dimensions, output_len)
        } else {
            self.dispatch(
                &[DispatchInput::Upload(input), DispatchInput::Mapped(weights)],
                output_len,
                self.conv2d_pipeline,
                bytes_of(dimensions),
                [
                    output_plane.div_ceil(8),
                    dimensions[4].div_ceil(32),
                    dimensions[0],
                ],
                KernelKind::Conv2d,
                None,
            )
        };
        self.stats.conv2d.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn conv2d_im2col(
        &mut self,
        input: &[u8],
        weights: &MappedTensor,
        dimensions: &[u32; 13],
        output_len: usize,
    ) -> Result<(Vec<f32>, f64)> {
        const MAX_COLUMN_BYTES: usize = 32 * 1024 * 1024;
        const MAX_SHADER_INVOCATIONS: usize = 65_535 * 256;
        let output_plane = dimensions[11] as usize * dimensions[12] as usize;
        let rows = dimensions[0] as usize * output_plane;
        let width = dimensions[1] as usize * dimensions[5] as usize * dimensions[6] as usize;
        let outputs = dimensions[4] as usize;
        let max_rows_by_bytes = MAX_COLUMN_BYTES / (width * size_of::<u16>());
        let max_rows_by_dispatch = MAX_SHADER_INVOCATIONS / width;
        let tile_rows = rows.min(max_rows_by_bytes).min(max_rows_by_dispatch).max(1);
        let column_bytes = tile_rows * width * size_of::<u16>();
        let output_bytes = output_len * size_of::<f32>();

        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|error| anyhow!("descriptor pool reset failed: {error:?}"))?;
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|error| anyhow!("command pool reset failed: {error:?}"))?;
        }
        for (binding, bytes) in [(0, input.len()), (2, output_bytes), (3, column_bytes)] {
            self.ensure_buffer(binding, bytes)?;
        }
        self.buffers[0]
            .as_ref()
            .expect("input scratch buffer exists")
            .write_bytes(input)?;
        let weight_descriptor = self.mapped_descriptor(weights, 1)?;

        let layouts = [self.set_layout, self.set_layout];
        let set_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let sets = unsafe { self.device.allocate_descriptor_sets(&set_info) }
            .map_err(|error| anyhow!("descriptor allocation failed: {error:?}"))?;
        let im2col_infos = [
            [self.buffers[0]
                .as_ref()
                .expect("input scratch buffer exists")
                .descriptor(input.len())],
            [self.buffers[3]
                .as_ref()
                .expect("column scratch buffer exists")
                .descriptor(column_bytes)],
        ];
        let gemm_infos = [
            [self.buffers[3]
                .as_ref()
                .expect("column scratch buffer exists")
                .descriptor(column_bytes)],
            [weight_descriptor],
            [self.buffers[2]
                .as_ref()
                .expect("output scratch buffer exists")
                .descriptor(output_bytes)],
        ];
        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&im2col_infos[0])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(sets[0])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&im2col_infos[1])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(sets[1])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&gemm_infos[0])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(sets[1])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&gemm_infos[1])
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(sets[1])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&gemm_infos[2])
                .build(),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let command_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&command_info) }
            .map_err(|error| anyhow!("command allocation failed: {error:?}"))?[0];
        let barrier = [vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()];
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .map_err(|error| anyhow!("command begin failed: {error:?}"))?;
            for row_start in (0..rows).step_by(tile_rows) {
                let current_rows = tile_rows.min(rows - row_start);
                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.im2col_pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[sets[0]],
                    &[],
                );
                let im2col_dimensions = [
                    dimensions[1],
                    dimensions[2],
                    dimensions[3],
                    dimensions[5],
                    dimensions[6],
                    dimensions[7],
                    dimensions[8],
                    dimensions[9],
                    dimensions[10],
                    dimensions[11],
                    dimensions[12],
                    row_start as u32,
                    current_rows as u32,
                ];
                self.device.cmd_push_constants(
                    command,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes_of(&im2col_dimensions),
                );
                self.device.cmd_dispatch(
                    command,
                    (current_rows * width).div_ceil(256) as u32,
                    1,
                    1,
                );
                self.device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );

                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.gemm_pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[sets[1]],
                    &[],
                );
                let gemm_dimensions = [
                    current_rows as u32,
                    outputs as u32,
                    width as u32,
                    row_start as u32,
                    output_plane as u32,
                ];
                self.device.cmd_push_constants(
                    command,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes_of(&gemm_dimensions),
                );
                self.device.cmd_dispatch(
                    command,
                    (outputs as u32).div_ceil(32),
                    (current_rows as u32).div_ceil(8),
                    1,
                );
                self.device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                );
            }
            self.device
                .end_command_buffer(command)
                .map_err(|error| anyhow!("command end failed: {error:?}"))?;
        }

        let commands = [command];
        let submit = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
        let started = Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &submit, vk::Fence::null())
                .map_err(|error| anyhow!("queue submit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| anyhow!("queue wait failed: {error:?}"))?;
        }
        let milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
        let output = self.buffers[2]
            .as_ref()
            .expect("output scratch buffer exists")
            .read_f32(output_len)?;
        self.stats.conv2d.calls += 1;
        self.stats.conv2d.dispatch_milliseconds += milliseconds;
        self.stats.uploaded_bytes = self.stats.uploaded_bytes.saturating_add(input.len() as u64);
        self.stats.peak_dispatch_bytes = self
            .stats
            .peak_dispatch_bytes
            .max((input.len() + output_bytes + column_bytes) as u64);
        Ok((output, milliseconds))
    }

    #[allow(clippy::too_many_arguments)]
    fn resnet(
        &mut self,
        input_f32: &[u8],
        input_f16: Option<&[u8]>,
        norm1_parameters: &[u8],
        norm2_parameters: &[u8],
        residual_biases: &[u8],
        conv1: &MappedTensor,
        conv2: &MappedTensor,
        shortcut: Option<&MappedTensor>,
        conv1_dimensions: &[u32; 13],
        conv2_dimensions: &[u32; 13],
        shortcut_dimensions: Option<&[u32; 13]>,
        groups: u32,
        epsilon: f32,
        has_channel_bias: bool,
        output_len: usize,
    ) -> Result<Vec<f32>> {
        let wall_started = Instant::now();
        if shortcut.is_some() != shortcut_dimensions.is_some()
            || shortcut.is_some() != input_f16.is_some()
        {
            bail!("Vulkan ResNet shortcut inputs are inconsistent");
        }
        let batch = conv1_dimensions[0] as usize;
        let input_channels = conv1_dimensions[1] as usize;
        let height = conv1_dimensions[2] as usize;
        let width = conv1_dimensions[3] as usize;
        let output_channels = conv1_dimensions[4] as usize;
        let input_len = batch
            .checked_mul(input_channels)
            .and_then(|value| value.checked_mul(height))
            .and_then(|value| value.checked_mul(width))
            .context("Vulkan ResNet input size overflow")?;
        if input_f32.len() != input_len * size_of::<f32>()
            || input_f16.is_some_and(|values| values.len() != input_len * size_of::<u16>())
            || norm1_parameters.len() != input_channels * 2 * size_of::<f32>()
            || residual_biases.len()
                != output_channels * (1 + usize::from(shortcut.is_some())) * size_of::<f32>()
        {
            bail!("Vulkan ResNet payload lengths do not match its dimensions");
        }

        let alignment = usize::try_from(self.storage_buffer_alignment)
            .context("storage-buffer alignment does not fit this platform")?;
        let norm1_offset = 0usize;
        let norm2_offset = align_up(norm1_parameters.len(), alignment)
            .context("Vulkan ResNet norm2 parameter offset overflow")?;
        let residual_offset = align_up(
            norm2_offset
                .checked_add(norm2_parameters.len())
                .context("Vulkan ResNet residual parameter offset overflow")?,
            alignment,
        )
        .context("Vulkan ResNet residual parameter alignment overflow")?;
        let parameter_bytes = residual_offset
            .checked_add(residual_biases.len())
            .context("Vulkan ResNet parameter buffer overflow")?;
        let norm1_output_bytes = input_len * size_of::<u16>();
        let norm2_output_bytes = output_len * size_of::<u16>();
        let output_bytes = output_len * size_of::<f32>();
        let column_bytes = [
            convolution_column_bytes(conv1_dimensions),
            convolution_column_bytes(conv2_dimensions),
            shortcut_dimensions.map_or(0, convolution_column_bytes),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        for (binding, bytes) in [
            (0, input_f32.len()),
            (1, norm1_output_bytes.max(input_f16.map_or(0, <[u8]>::len))),
            (2, parameter_bytes),
            (3, output_bytes),
            (4, norm2_output_bytes),
            (5, output_bytes),
            (6, if shortcut.is_some() { output_bytes } else { 1 }),
            (7, column_bytes.max(1)),
        ] {
            self.ensure_buffer(binding, bytes)?;
        }
        self.buffers[0]
            .as_ref()
            .expect("ResNet input buffer exists")
            .write_bytes(input_f32)?;
        if let Some(input_f16) = input_f16 {
            self.buffers[1]
                .as_ref()
                .expect("ResNet FP16 input buffer exists")
                .write_bytes(input_f16)?;
        }
        {
            let parameters = self.buffers[2]
                .as_ref()
                .expect("ResNet parameter buffer exists");
            parameters.write_bytes_at(norm1_offset, norm1_parameters)?;
            parameters.write_bytes_at(norm2_offset, norm2_parameters)?;
            parameters.write_bytes_at(residual_offset, residual_biases)?;
        }

        let conv1_weight = self.mapped_descriptor(conv1, 1)?;
        let conv2_weight = self.mapped_descriptor(conv2, 1)?;
        let shortcut_weight = shortcut
            .map(|weight| self.mapped_descriptor(weight, 1))
            .transpose()?;

        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|error| anyhow!("descriptor pool reset failed: {error:?}"))?;
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|error| anyhow!("command pool reset failed: {error:?}"))?;
        }
        let command_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&command_info) }
            .map_err(|error| anyhow!("command allocation failed: {error:?}"))?[0];
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .map_err(|error| anyhow!("command begin failed: {error:?}"))?;
        }

        if let (Some(dimensions), Some(weight)) = (shortcut_dimensions, shortcut_weight) {
            self.record_convolution(
                command,
                self.buffers[1]
                    .as_ref()
                    .expect("shortcut input exists")
                    .descriptor(input_f16.expect("shortcut input was checked").len()),
                weight,
                self.buffers[6]
                    .as_ref()
                    .expect("shortcut output exists")
                    .descriptor(output_bytes),
                dimensions,
            )?;
            self.record_compute_barrier(command);
        }

        self.record_groupnorm_silu(
            command,
            self.buffers[0]
                .as_ref()
                .expect("norm1 input exists")
                .descriptor(input_f32.len()),
            self.buffers[2]
                .as_ref()
                .expect("ResNet parameter buffer exists")
                .descriptor_at(norm1_offset, norm1_parameters.len())?,
            self.buffers[1]
                .as_ref()
                .expect("norm1 output exists")
                .descriptor(norm1_output_bytes),
            [
                batch as u32,
                input_channels as u32,
                height as u32,
                width as u32,
                groups,
                0,
                0,
                epsilon.to_bits(),
            ],
        )?;
        self.record_compute_barrier(command);
        self.record_convolution(
            command,
            self.buffers[1]
                .as_ref()
                .expect("conv1 input exists")
                .descriptor(norm1_output_bytes),
            conv1_weight,
            self.buffers[3]
                .as_ref()
                .expect("conv1 output exists")
                .descriptor(output_bytes),
            conv1_dimensions,
        )?;
        self.record_compute_barrier(command);
        self.record_groupnorm_silu(
            command,
            self.buffers[3]
                .as_ref()
                .expect("norm2 input exists")
                .descriptor(output_bytes),
            self.buffers[2]
                .as_ref()
                .expect("ResNet parameter buffer exists")
                .descriptor_at(norm2_offset, norm2_parameters.len())?,
            self.buffers[4]
                .as_ref()
                .expect("norm2 output exists")
                .descriptor(norm2_output_bytes),
            [
                batch as u32,
                output_channels as u32,
                height as u32,
                width as u32,
                groups,
                1,
                u32::from(has_channel_bias),
                epsilon.to_bits(),
            ],
        )?;
        self.record_compute_barrier(command);
        self.record_convolution(
            command,
            self.buffers[4]
                .as_ref()
                .expect("conv2 input exists")
                .descriptor(norm2_output_bytes),
            conv2_weight,
            self.buffers[5]
                .as_ref()
                .expect("conv2 output exists")
                .descriptor(output_bytes),
            conv2_dimensions,
        )?;
        self.record_compute_barrier(command);

        let residual = if shortcut.is_some() {
            self.buffers[6]
                .as_ref()
                .expect("shortcut output exists")
                .descriptor(output_bytes)
        } else {
            self.buffers[0]
                .as_ref()
                .expect("identity residual exists")
                .descriptor(input_f32.len())
        };
        let add_set = self.allocate_descriptor_set(&[
            (0, residual),
            (
                1,
                self.buffers[5]
                    .as_ref()
                    .expect("residual hidden input exists")
                    .descriptor(output_bytes),
            ),
            (
                2,
                self.buffers[3]
                    .as_ref()
                    .expect("residual output exists")
                    .descriptor(output_bytes),
            ),
            (
                3,
                self.buffers[2]
                    .as_ref()
                    .expect("ResNet parameter buffer exists")
                    .descriptor_at(residual_offset, residual_biases.len())?,
            ),
        ])?;
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.residual_add_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[add_set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&[
                    output_len as u32,
                    output_channels as u32,
                    (height * width) as u32,
                    u32::from(shortcut.is_some()),
                ]),
            );
            self.device
                .cmd_dispatch(command, (output_len as u32).div_ceil(256), 1, 1);
            self.device
                .end_command_buffer(command)
                .map_err(|error| anyhow!("command end failed: {error:?}"))?;
        }

        let commands = [command];
        let submit = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
        let dispatch_started = Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &submit, vk::Fence::null())
                .map_err(|error| anyhow!("queue submit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| anyhow!("queue wait failed: {error:?}"))?;
        }
        let dispatch_milliseconds = dispatch_started.elapsed().as_secs_f64() * 1_000.0;
        let output = self.buffers[3]
            .as_ref()
            .expect("ResNet output exists")
            .read_f32(output_len)?;
        let convolution_calls = 2 + u64::from(shortcut.is_some());
        self.stats.conv2d.calls += convolution_calls;
        self.stats.conv2d.dispatch_milliseconds += dispatch_milliseconds;
        self.stats.conv2d.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        let uploaded = input_f32.len()
            + input_f16.map_or(0, <[u8]>::len)
            + norm1_parameters.len()
            + norm2_parameters.len()
            + residual_biases.len();
        self.stats.uploaded_bytes = self.stats.uploaded_bytes.saturating_add(uploaded as u64);
        self.stats.peak_dispatch_bytes = self.stats.peak_dispatch_bytes.max(
            (input_f32.len()
                + norm1_output_bytes
                + parameter_bytes
                + output_bytes * 4
                + norm2_output_bytes
                + column_bytes) as u64,
        );
        Ok(output)
    }

    fn record_groupnorm_silu(
        &self,
        command: vk::CommandBuffer,
        input: vk::DescriptorBufferInfo,
        parameters: vk::DescriptorBufferInfo,
        output: vk::DescriptorBufferInfo,
        dimensions: [u32; 8],
    ) -> Result<()> {
        let set = self.allocate_descriptor_set(&[(0, input), (1, parameters), (3, output)])?;
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.groupnorm_silu_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&dimensions),
            );
            self.device
                .cmd_dispatch(command, dimensions[0] * dimensions[4], 1, 1);
        }
        Ok(())
    }

    fn record_convolution(
        &self,
        command: vk::CommandBuffer,
        input: vk::DescriptorBufferInfo,
        weight: vk::DescriptorBufferInfo,
        output: vk::DescriptorBufferInfo,
        dimensions: &[u32; 13],
    ) -> Result<()> {
        let output_plane = dimensions[11]
            .checked_mul(dimensions[12])
            .context("Vulkan convolution output plane overflow")?;
        let reduction_width = dimensions[1]
            .checked_mul(dimensions[5])
            .and_then(|value| value.checked_mul(dimensions[6]))
            .context("Vulkan convolution reduction width overflow")?;
        let rows = dimensions[0]
            .checked_mul(output_plane)
            .context("Vulkan convolution row count overflow")?;
        let operations = u64::from(rows)
            .saturating_mul(u64::from(reduction_width))
            .saturating_mul(u64::from(dimensions[4]));
        if reduction_width % 4 != 0 || operations < 50_000_000 {
            let set = self.allocate_descriptor_set(&[(0, input), (1, weight), (2, output)])?;
            unsafe {
                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.conv2d_pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
                self.device.cmd_push_constants(
                    command,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes_of(dimensions),
                );
                self.device.cmd_dispatch(
                    command,
                    output_plane.div_ceil(8),
                    dimensions[4].div_ceil(32),
                    dimensions[0],
                );
            }
            return Ok(());
        }

        let rows = rows as usize;
        let width = reduction_width as usize;
        let outputs = dimensions[4] as usize;
        let output_plane = output_plane as usize;
        let tile_rows = convolution_tile_rows(dimensions);
        let column_bytes = tile_rows * width * size_of::<u16>();
        let column = self.buffers[7]
            .as_ref()
            .context("Vulkan ResNet column buffer is missing")?
            .descriptor(column_bytes);
        let im2col_set = self.allocate_descriptor_set(&[(0, input), (3, column)])?;
        let gemm_set = self.allocate_descriptor_set(&[(0, column), (1, weight), (2, output)])?;
        for row_start in (0..rows).step_by(tile_rows) {
            let current_rows = tile_rows.min(rows - row_start);
            let im2col_dimensions = [
                dimensions[1],
                dimensions[2],
                dimensions[3],
                dimensions[5],
                dimensions[6],
                dimensions[7],
                dimensions[8],
                dimensions[9],
                dimensions[10],
                dimensions[11],
                dimensions[12],
                row_start as u32,
                current_rows as u32,
            ];
            unsafe {
                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.im2col_pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[im2col_set],
                    &[],
                );
                self.device.cmd_push_constants(
                    command,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes_of(&im2col_dimensions),
                );
                self.device.cmd_dispatch(
                    command,
                    (current_rows * width).div_ceil(256) as u32,
                    1,
                    1,
                );
            }
            self.record_compute_barrier(command);
            let gemm_dimensions = [
                current_rows as u32,
                outputs as u32,
                width as u32,
                row_start as u32,
                output_plane as u32,
            ];
            unsafe {
                self.device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.gemm_pipeline,
                );
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[gemm_set],
                    &[],
                );
                self.device.cmd_push_constants(
                    command,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes_of(&gemm_dimensions),
                );
                self.device.cmd_dispatch(
                    command,
                    (outputs as u32).div_ceil(32),
                    (current_rows as u32).div_ceil(8),
                    1,
                );
            }
            self.record_compute_barrier(command);
        }
        Ok(())
    }

    fn allocate_descriptor_set(
        &self,
        bindings: &[(u32, vk::DescriptorBufferInfo)],
    ) -> Result<vk::DescriptorSet> {
        let layouts = [self.set_layout];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let set = unsafe { self.device.allocate_descriptor_sets(&info) }
            .map_err(|error| anyhow!("descriptor allocation failed: {error:?}"))?[0];
        let buffer_infos = bindings
            .iter()
            .map(|(_, descriptor)| [*descriptor])
            .collect::<Vec<_>>();
        let writes = bindings
            .iter()
            .enumerate()
            .map(|(index, (binding, _))| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(set)
                    .dst_binding(*binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_infos[index])
                    .build()
            })
            .collect::<Vec<_>>();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        Ok(set)
    }

    fn record_compute_barrier(&self, command: vk::CommandBuffer) {
        let barriers = [vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build()];
        unsafe {
            self.device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &barriers,
                &[],
                &[],
            )
        };
    }

    fn feed_forward(
        &mut self,
        input: &[u8],
        first_bias: &[u8],
        first_weight: &MappedTensor,
        second_weight: &MappedTensor,
        rows: u32,
        channels: u32,
        hidden: u32,
    ) -> Result<Vec<f32>> {
        let wall_started = Instant::now();
        if rows == 0 || channels == 0 || hidden == 0 || channels % 4 != 0 || hidden % 4 != 0 {
            bail!("invalid Vulkan feed-forward dimensions");
        }
        let input_bytes = rows as usize * channels as usize * size_of::<u16>();
        let bias_bytes = hidden as usize * 2 * size_of::<f32>();
        let projected_bytes = rows as usize * hidden as usize * 2 * size_of::<f32>();
        let gated_bytes = rows as usize * hidden as usize * size_of::<u16>();
        let output_elements = rows as usize * channels as usize;
        let output_bytes = output_elements * size_of::<f32>();
        if input.len() != input_bytes || first_bias.len() != bias_bytes {
            bail!("Vulkan feed-forward payload lengths are invalid");
        }
        for (binding, bytes) in [
            (0, input_bytes),
            (1, bias_bytes),
            (2, projected_bytes),
            (3, gated_bytes),
            (4, output_bytes),
        ] {
            self.ensure_buffer(binding, bytes)?;
        }
        self.buffers[0]
            .as_ref()
            .expect("feed-forward input exists")
            .write_bytes(input)?;
        self.buffers[1]
            .as_ref()
            .expect("feed-forward bias exists")
            .write_bytes(first_bias)?;
        let first_weight = self.mapped_descriptor(first_weight, 1)?;
        let second_weight = self.mapped_descriptor(second_weight, 1)?;

        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|error| anyhow!("descriptor pool reset failed: {error:?}"))?;
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|error| anyhow!("command pool reset failed: {error:?}"))?;
        }
        let command_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&command_info) }
            .map_err(|error| anyhow!("command allocation failed: {error:?}"))?[0];
        let first_set = self.allocate_descriptor_set(&[
            (
                0,
                self.buffers[0]
                    .as_ref()
                    .expect("feed-forward input exists")
                    .descriptor(input_bytes),
            ),
            (1, first_weight),
            (
                2,
                self.buffers[2]
                    .as_ref()
                    .expect("feed-forward projection exists")
                    .descriptor(projected_bytes),
            ),
        ])?;
        let geglu_set = self.allocate_descriptor_set(&[
            (
                0,
                self.buffers[2]
                    .as_ref()
                    .expect("GEGLU input exists")
                    .descriptor(projected_bytes),
            ),
            (
                1,
                self.buffers[1]
                    .as_ref()
                    .expect("GEGLU bias exists")
                    .descriptor(bias_bytes),
            ),
            (
                3,
                self.buffers[3]
                    .as_ref()
                    .expect("GEGLU output exists")
                    .descriptor(gated_bytes),
            ),
        ])?;
        let second_set = self.allocate_descriptor_set(&[
            (
                0,
                self.buffers[3]
                    .as_ref()
                    .expect("second feed-forward input exists")
                    .descriptor(gated_bytes),
            ),
            (1, second_weight),
            (
                2,
                self.buffers[4]
                    .as_ref()
                    .expect("feed-forward output exists")
                    .descriptor(output_bytes),
            ),
        ])?;
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .map_err(|error| anyhow!("command begin failed: {error:?}"))?;
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.gemm_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[first_set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&[rows, hidden * 2, channels, 0, 0]),
            );
            self.device
                .cmd_dispatch(command, (hidden * 2).div_ceil(32), rows.div_ceil(8), 1);
        }
        self.record_compute_barrier(command);
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.geglu_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[geglu_set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&[rows, hidden]),
            );
            self.device
                .cmd_dispatch(command, (rows * hidden).div_ceil(256), 1, 1);
        }
        self.record_compute_barrier(command);
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.gemm_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[second_set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&[rows, channels, hidden, 0, 0]),
            );
            self.device
                .cmd_dispatch(command, channels.div_ceil(32), rows.div_ceil(8), 1);
            self.device
                .end_command_buffer(command)
                .map_err(|error| anyhow!("command end failed: {error:?}"))?;
        }

        let commands = [command];
        let submit = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
        let dispatch_started = Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &submit, vk::Fence::null())
                .map_err(|error| anyhow!("queue submit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| anyhow!("queue wait failed: {error:?}"))?;
        }
        let dispatch_milliseconds = dispatch_started.elapsed().as_secs_f64() * 1_000.0;
        let output = self.buffers[4]
            .as_ref()
            .expect("feed-forward output exists")
            .read_f32(output_elements)?;
        self.stats.gemm.calls += 2;
        self.stats.gemm.dispatch_milliseconds += dispatch_milliseconds;
        self.stats.gemm.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .saturating_add((input.len() + first_bias.len()) as u64);
        self.stats.peak_dispatch_bytes = self
            .stats
            .peak_dispatch_bytes
            .max((input_bytes + bias_bytes + projected_bytes + gated_bytes + output_bytes) as u64);
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn projected_attention(
        &mut self,
        query: &[u8],
        key_value: &[u8],
        query_weight: &MappedTensor,
        key_weight: &MappedTensor,
        value_weight: &MappedTensor,
        output_weight: &MappedTensor,
        batch: u32,
        queries: u32,
        keys: u32,
        channels: u32,
        key_value_width: u32,
        heads: u32,
    ) -> Result<Vec<f32>> {
        let wall_started = Instant::now();
        if batch == 0
            || queries == 0
            || keys == 0
            || channels == 0
            || channels % heads != 0
            || channels % 4 != 0
            || key_value_width % 4 != 0
            || keys > 4096
        {
            bail!("invalid Vulkan projected-attention dimensions");
        }
        let query_elements = batch as usize * queries as usize * channels as usize;
        let key_value_elements = batch as usize * keys as usize * key_value_width as usize;
        let key_elements = batch as usize * keys as usize * channels as usize;
        if query.len() != query_elements * size_of::<u16>()
            || key_value.len() != key_value_elements * size_of::<u16>()
        {
            bail!("Vulkan projected-attention input lengths are invalid");
        }
        let query_half_bytes = query_elements * size_of::<u16>();
        let key_half_bytes = key_elements * size_of::<u16>();
        let attention_bytes = query_elements * size_of::<f32>();
        for (binding, bytes) in [
            (0, query.len()),
            (1, key_value.len()),
            (2, query_half_bytes),
            (3, key_half_bytes),
            (4, key_half_bytes),
            (5, attention_bytes),
            (6, query_half_bytes),
            (7, attention_bytes),
        ] {
            self.ensure_buffer(binding, bytes)?;
        }
        self.buffers[0]
            .as_ref()
            .expect("projected-attention query buffer exists")
            .write_bytes(query)?;
        self.buffers[1]
            .as_ref()
            .expect("projected-attention key/value buffer exists")
            .write_bytes(key_value)?;
        let query_weight = self.mapped_descriptor(query_weight, 1)?;
        let key_weight = self.mapped_descriptor(key_weight, 1)?;
        let value_weight = self.mapped_descriptor(value_weight, 1)?;
        let output_weight = self.mapped_descriptor(output_weight, 1)?;

        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|error| anyhow!("descriptor pool reset failed: {error:?}"))?;
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|error| anyhow!("command pool reset failed: {error:?}"))?;
        }
        let command_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command = unsafe { self.device.allocate_command_buffers(&command_info) }
            .map_err(|error| anyhow!("command allocation failed: {error:?}"))?[0];
        unsafe {
            self.device
                .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                .map_err(|error| anyhow!("command begin failed: {error:?}"))?;
        }

        self.record_gemm_heads(
            command,
            self.buffers[0]
                .as_ref()
                .expect("query projection input exists")
                .descriptor(query.len()),
            query_weight,
            self.buffers[2]
                .as_ref()
                .expect("query projection output exists")
                .descriptor(query_half_bytes),
            [batch * queries, channels, channels, batch, heads, queries],
        )?;
        self.record_gemm_heads(
            command,
            self.buffers[1]
                .as_ref()
                .expect("key projection input exists")
                .descriptor(key_value.len()),
            key_weight,
            self.buffers[3]
                .as_ref()
                .expect("key projection output exists")
                .descriptor(key_half_bytes),
            [batch * keys, channels, key_value_width, batch, heads, keys],
        )?;
        self.record_gemm_heads(
            command,
            self.buffers[1]
                .as_ref()
                .expect("value projection input exists")
                .descriptor(key_value.len()),
            value_weight,
            self.buffers[4]
                .as_ref()
                .expect("value projection output exists")
                .descriptor(key_half_bytes),
            [batch * keys, channels, key_value_width, batch, heads, keys],
        )?;
        self.record_compute_barrier(command);

        let attention_set = self.allocate_descriptor_set(&[
            (
                0,
                self.buffers[2]
                    .as_ref()
                    .expect("attention query exists")
                    .descriptor(query_half_bytes),
            ),
            (
                1,
                self.buffers[3]
                    .as_ref()
                    .expect("attention key exists")
                    .descriptor(key_half_bytes),
            ),
            (
                2,
                self.buffers[4]
                    .as_ref()
                    .expect("attention value exists")
                    .descriptor(key_half_bytes),
            ),
            (
                3,
                self.buffers[5]
                    .as_ref()
                    .expect("attention output exists")
                    .descriptor(attention_bytes),
            ),
        ])?;
        let head_width = channels / heads;
        let attention_dimensions = [
            batch * heads,
            queries,
            keys,
            head_width,
            (1.0 / (head_width as f32).sqrt()).to_bits(),
            0,
            0,
        ];
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.attention_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[attention_set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&attention_dimensions),
            );
            self.device.cmd_dispatch(command, queries, batch * heads, 1);
        }
        self.record_compute_barrier(command);

        let merge_set = self.allocate_descriptor_set(&[
            (
                0,
                self.buffers[5]
                    .as_ref()
                    .expect("head-merge input exists")
                    .descriptor(attention_bytes),
            ),
            (
                3,
                self.buffers[6]
                    .as_ref()
                    .expect("head-merge output exists")
                    .descriptor(query_half_bytes),
            ),
        ])?;
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.merge_heads_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[merge_set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&[batch, heads, queries, head_width]),
            );
            self.device
                .cmd_dispatch(command, (query_elements as u32).div_ceil(256), 1, 1);
        }
        self.record_compute_barrier(command);

        let output_set = self.allocate_descriptor_set(&[
            (
                0,
                self.buffers[6]
                    .as_ref()
                    .expect("output projection input exists")
                    .descriptor(query_half_bytes),
            ),
            (1, output_weight),
            (
                2,
                self.buffers[7]
                    .as_ref()
                    .expect("output projection output exists")
                    .descriptor(attention_bytes),
            ),
        ])?;
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.gemm_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[output_set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&[batch * queries, channels, channels, 0, 0]),
            );
            self.device.cmd_dispatch(
                command,
                channels.div_ceil(32),
                (batch * queries).div_ceil(8),
                1,
            );
            self.device
                .end_command_buffer(command)
                .map_err(|error| anyhow!("command end failed: {error:?}"))?;
        }

        let commands = [command];
        let submit = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
        let dispatch_started = Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &submit, vk::Fence::null())
                .map_err(|error| anyhow!("queue submit failed: {error:?}"))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|error| anyhow!("queue wait failed: {error:?}"))?;
        }
        let dispatch_milliseconds = dispatch_started.elapsed().as_secs_f64() * 1_000.0;
        let output = self.buffers[7]
            .as_ref()
            .expect("projected-attention output exists")
            .read_f32(query_elements)?;
        self.stats.attention.calls += 1;
        self.stats.attention.dispatch_milliseconds += dispatch_milliseconds;
        self.stats.attention.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .saturating_add((query.len() + key_value.len()) as u64);
        self.stats.peak_dispatch_bytes = self.stats.peak_dispatch_bytes.max(
            (query.len()
                + key_value.len()
                + query_half_bytes * 2
                + key_half_bytes * 2
                + attention_bytes * 2) as u64,
        );
        Ok(output)
    }

    fn record_gemm_heads(
        &self,
        command: vk::CommandBuffer,
        input: vk::DescriptorBufferInfo,
        weight: vk::DescriptorBufferInfo,
        output: vk::DescriptorBufferInfo,
        dimensions: [u32; 6],
    ) -> Result<()> {
        let set = self.allocate_descriptor_set(&[(0, input), (1, weight), (2, output)])?;
        unsafe {
            self.device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.gemm_heads_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes_of(&dimensions),
            );
            self.device.cmd_dispatch(
                command,
                dimensions[1].div_ceil(32),
                dimensions[0].div_ceil(8),
                1,
            );
        }
        Ok(())
    }

    fn attention(
        &mut self,
        query: &[u8],
        key: &[u8],
        value: &[u8],
        dimensions: &[u32; 7],
        output_len: usize,
    ) -> Result<(Vec<f32>, f64)> {
        let wall_started = Instant::now();
        let result = self.dispatch(
            &[
                DispatchInput::Upload(query),
                DispatchInput::Upload(key),
                DispatchInput::Upload(value),
            ],
            output_len,
            self.attention_pipeline,
            bytes_of(dimensions),
            [dimensions[1], dimensions[0], 1],
            KernelKind::Attention,
            Some(64),
        );
        self.stats.attention.wall_milliseconds += wall_started.elapsed().as_secs_f64() * 1_000.0;
        result
    }

    fn dispatch(
        &mut self,
        inputs: &[DispatchInput<'_>],
        output_len: usize,
        pipeline: vk::Pipeline,
        push_constants: &[u8],
        groups: [u32; 3],
        kind: KernelKind,
        max_groups_x_per_submit: Option<u32>,
    ) -> Result<(Vec<f32>, f64)> {
        if inputs.is_empty() || inputs.len() > 3 {
            bail!("Vulkan dispatch requires between one and three input buffers");
        }
        if push_constants.len() > 64 || push_constants.len() % 4 != 0 {
            bail!("invalid Vulkan push-constant payload");
        }

        unsafe {
            self.device
                .reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|error| anyhow!("descriptor pool reset failed: {error:?}"))?;
            self.device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(|error| anyhow!("command pool reset failed: {error:?}"))?;
        }

        let mut input_descriptors = Vec::with_capacity(inputs.len());
        for (binding, input) in inputs.iter().copied().enumerate() {
            let descriptor = match input {
                DispatchInput::Upload(bytes) => {
                    self.ensure_buffer(binding, bytes.len())?;
                    let buffer = self.buffers[binding]
                        .as_ref()
                        .expect("scratch buffer was just allocated");
                    buffer.write_bytes(bytes)?;
                    buffer.descriptor(bytes.len())
                }
                DispatchInput::Mapped(tensor) => self.mapped_descriptor(tensor, binding)?,
            };
            input_descriptors.push(descriptor);
        }
        let output_binding = inputs.len();
        let output_bytes = output_len * size_of::<f32>();
        self.ensure_buffer(output_binding, output_bytes)?;

        let layouts = [self.set_layout];
        let set_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&set_info) }
            .map_err(|error| anyhow!("descriptor allocation failed: {error:?}"))?[0];
        let mut buffer_infos = input_descriptors
            .iter()
            .copied()
            .map(|descriptor| [descriptor])
            .collect::<Vec<_>>();
        buffer_infos.push([self.buffers[output_binding]
            .as_ref()
            .expect("output scratch buffer exists")
            .descriptor(output_bytes)]);
        let writes = (0..buffer_infos.len())
            .map(|binding| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buffer_infos[binding])
                    .build()
            })
            .collect::<Vec<_>>();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        if max_groups_x_per_submit.is_some() && push_constants.len() < 7 * size_of::<u32>() {
            bail!("chunked Vulkan dispatch requires a query-offset push constant");
        }
        let groups_per_submit = max_groups_x_per_submit.unwrap_or(groups[0]).max(1);
        let mut group_offset = 0;
        let mut milliseconds = 0.0;
        let mut submission_count = 0;
        while group_offset < groups[0] {
            if submission_count > 0 {
                unsafe {
                    self.device
                        .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                        .map_err(|error| anyhow!("command pool reset failed: {error:?}"))?;
                }
            }
            let groups_this_submit = groups_per_submit.min(groups[0] - group_offset);
            let mut submitted_constants = push_constants.to_vec();
            if max_groups_x_per_submit.is_some() {
                submitted_constants[24..28].copy_from_slice(&group_offset.to_ne_bytes());
            }
            let command_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command = unsafe { self.device.allocate_command_buffers(&command_info) }
                .map_err(|error| anyhow!("command allocation failed: {error:?}"))?[0];
            unsafe {
                self.device
                    .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
                    .map_err(|error| anyhow!("command begin failed: {error:?}"))?;
                self.device
                    .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
                self.device.cmd_push_constants(
                    command,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &submitted_constants,
                );
                self.device
                    .cmd_dispatch(command, groups_this_submit, groups[1], groups[2]);
                self.device
                    .end_command_buffer(command)
                    .map_err(|error| anyhow!("command end failed: {error:?}"))?;
            }

            let commands = [command];
            let submit = [vk::SubmitInfo::builder().command_buffers(&commands).build()];
            let started = Instant::now();
            unsafe {
                self.device
                    .queue_submit(self.queue, &submit, vk::Fence::null())
                    .map_err(|error| anyhow!("queue submit failed: {error:?}"))?;
                self.device.queue_wait_idle(self.queue).map_err(|error| {
                    anyhow!(
                        "queue wait failed for group-x rows {group_offset}..{}: {error:?}",
                        group_offset + groups_this_submit
                    )
                })?;
            }
            milliseconds += started.elapsed().as_secs_f64() * 1_000.0;
            submission_count += 1;
            group_offset += groups_this_submit;
        }
        let uploaded_bytes = inputs
            .iter()
            .copied()
            .map(DispatchInput::uploaded_len)
            .sum::<usize>();
        let dispatch_bytes = uploaded_bytes as u64 + (output_len * size_of::<f32>()) as u64;
        self.stats.uploaded_bytes = self
            .stats
            .uploaded_bytes
            .saturating_add(uploaded_bytes as u64);
        self.stats.peak_dispatch_bytes = self.stats.peak_dispatch_bytes.max(dispatch_bytes);
        let kernel = match kind {
            KernelKind::Elementwise => &mut self.stats.elementwise,
            KernelKind::Norm => &mut self.stats.norm,
            KernelKind::Gemm => &mut self.stats.gemm,
            KernelKind::Conv2d => &mut self.stats.conv2d,
            KernelKind::Attention => &mut self.stats.attention,
        };
        kernel.calls += submission_count;
        kernel.dispatch_milliseconds += milliseconds;
        let output = self.buffers[output_binding]
            .as_ref()
            .expect("output scratch buffer exists")
            .read_f32(output_len)?;
        Ok((output, milliseconds))
    }

    fn ensure_buffer(&mut self, binding: usize, required: usize) -> Result<()> {
        if self.buffers[binding]
            .as_ref()
            .is_some_and(|buffer| buffer.bytes >= required)
        {
            return Ok(());
        }
        let capacity = required.checked_next_power_of_two().unwrap_or(required);
        self.buffers[binding] = Some(Buffer::new(
            &self.instance,
            self.physical,
            &self.device,
            capacity,
        )?);
        Ok(())
    }

    fn mapped_descriptor(
        &mut self,
        tensor: &MappedTensor,
        _fallback_binding: usize,
    ) -> Result<vk::DescriptorBufferInfo> {
        let key = tensor.mapping().as_ptr() as usize;
        if !self.model_mappings.contains_key(&key) {
            let (capacity_bytes, tensor_count_hint) =
                staged_arena_budget(key).unwrap_or((tensor.mapping().len(), tensor.tensor_count()));
            let mapping = if let Some(external_host) = self.external_host.as_ref() {
                match ImportedMapping::new(
                    &self.instance,
                    self.physical,
                    &self.device,
                    external_host,
                    self.external_host_alignment,
                    Arc::clone(tensor.mapping()),
                ) {
                    Ok(imported) => ModelMapping::Imported(imported),
                    Err(error) => {
                        eprintln!(
                            "Quartz Vulkan: direct model mapping unavailable ({error:#}); using a persistent Saient weight arena"
                        );
                        ModelMapping::Uploaded(UploadedMapping::new(
                            &self.instance,
                            self.physical,
                            &self.device,
                            self.storage_buffer_alignment,
                            Arc::clone(tensor.mapping()),
                            capacity_bytes,
                            tensor_count_hint,
                        )?)
                    }
                }
            } else {
                ModelMapping::Uploaded(UploadedMapping::new(
                    &self.instance,
                    self.physical,
                    &self.device,
                    self.storage_buffer_alignment,
                    Arc::clone(tensor.mapping()),
                    capacity_bytes,
                    tensor_count_hint,
                )?)
            };
            self.model_mappings.insert(key, mapping);
        }
        let (descriptor, newly_cached) = self
            .model_mappings
            .get_mut(&key)
            .expect("model mapping was just inserted")
            .descriptor(tensor)?;
        self.stats.cached_weight_bytes = self
            .stats
            .cached_weight_bytes
            .saturating_add(newly_cached as u64);
        Ok(descriptor)
    }
}

impl Drop for VulkanRuntime {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.model_mappings.clear();
            self.resident_buffers.clear();
            for buffer in &mut self.buffers {
                drop(buffer.take());
            }
            if self.command_pool != vk::CommandPool::null() {
                self.device.destroy_command_pool(self.command_pool, None);
            }
            if self.descriptor_pool != vk::DescriptorPool::null() {
                self.device
                    .destroy_descriptor_pool(self.descriptor_pool, None);
            }
            if self.conv2d_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.conv2d_pipeline, None);
            }
            if self.attention_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.attention_pipeline, None);
            }
            if self.im2col_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.im2col_pipeline, None);
            }
            if self.groupnorm_silu_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.groupnorm_silu_pipeline, None);
            }
            if self.residual_add_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.residual_add_pipeline, None);
            }
            if self.gemm_heads_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.gemm_heads_pipeline, None);
            }
            if self.merge_heads_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.merge_heads_pipeline, None);
            }
            if self.geglu_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.geglu_pipeline, None);
            }
            if self.elementwise_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.elementwise_pipeline, None);
            }
            if self.channel_rmsnorm_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.channel_rmsnorm_pipeline, None);
            }
            if self.resident_linear_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.resident_linear_pipeline, None);
            }
            if self.layernorm_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.layernorm_pipeline, None);
            }
            if self.rmsnorm_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.rmsnorm_pipeline, None);
            }
            if self.rope_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.rope_pipeline, None);
            }
            if self.f32_attention_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.f32_attention_pipeline, None);
            }
            if self.patch_layout_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.patch_layout_pipeline, None);
            }
            if self.wan_head_modulate_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.wan_head_modulate_pipeline, None);
            }
            if self.resident_conv3d_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.resident_conv3d_pipeline, None);
            }
            if self.resident_conv2d_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.resident_conv2d_pipeline, None);
            }
            if self.ncthw_temporal_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.ncthw_temporal_pipeline, None);
            }
            if self.vae_spatial_layout_pipeline != vk::Pipeline::null() {
                self.device
                    .destroy_pipeline(self.vae_spatial_layout_pipeline, None);
            }
            if self.gemm_pipeline != vk::Pipeline::null() {
                self.device.destroy_pipeline(self.gemm_pipeline, None);
            }
            if self.pipeline_layout != vk::PipelineLayout::null() {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None);
            }
            if self.set_layout != vk::DescriptorSetLayout::null() {
                self.device
                    .destroy_descriptor_set_layout(self.set_layout, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn descriptor_binding(binding: u32) -> vk::DescriptorSetLayoutBinding {
    vk::DescriptorSetLayoutBinding::builder()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .build()
}

enum ModelMapping {
    Imported(ImportedMapping),
    Uploaded(UploadedMapping),
}

impl ModelMapping {
    fn descriptor(&mut self, tensor: &MappedTensor) -> Result<(vk::DescriptorBufferInfo, usize)> {
        match self {
            Self::Imported(mapping) => Ok((
                mapping.descriptor(tensor.offset(), tensor.bytes().len())?,
                0,
            )),
            Self::Uploaded(mapping) => mapping.descriptor(tensor),
        }
    }

    /// Evict cached tensors from an uploaded arena; zero-copy imports hold no
    /// separate device copy, so there is nothing to evict.
    fn reset(&mut self) {
        if let Self::Uploaded(mapping) = self {
            mapping.reset();
        }
    }
}

struct UploadedMapping {
    buffer: Buffer,
    alignment: usize,
    next_offset: usize,
    tensors: HashMap<usize, (usize, usize)>,
    _mapping: Arc<memmap2::Mmap>,
}

impl UploadedMapping {
    fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        alignment: u64,
        mapping: Arc<memmap2::Mmap>,
        capacity_bytes: usize,
        tensor_count_hint: usize,
    ) -> Result<Self> {
        let alignment = usize::try_from(alignment)
            .context("storage-buffer alignment does not fit this platform")?;
        if alignment == 0 || !alignment.is_power_of_two() {
            bail!("Vulkan reported invalid storage-buffer alignment {alignment}");
        }
        let padding = alignment
            .checked_mul(tensor_count_hint)
            .context("persistent weight-arena padding overflow")?;
        let bytes = capacity_bytes
            .checked_add(padding)
            .context("persistent weight-arena size overflow")?;
        if std::env::var("QUARTZ_DEBUG_ARENA_SIZE").is_ok() {
            eprintln!("Quartz Vulkan: allocating weight arena of {bytes} bytes");
        }
        Ok(Self {
            buffer: Buffer::new(instance, physical, device, bytes)?,
            alignment,
            next_offset: 0,
            tensors: HashMap::with_capacity(tensor_count_hint),
            _mapping: mapping,
        })
    }

    /// Drop every cached tensor and rewind the write cursor so the same
    /// bounded buffer can be reused for the next stage's weights. Safe to call
    /// between dispatches because every Vulkan submission in this runtime
    /// already blocks on `queue_wait_idle` before returning, so no dispatch
    /// can still be reading a tensor we're about to overwrite.
    fn reset(&mut self) {
        self.tensors.clear();
        self.next_offset = 0;
    }

    fn descriptor(&mut self, tensor: &MappedTensor) -> Result<(vk::DescriptorBufferInfo, usize)> {
        let source_offset = tensor.offset();
        let bytes = tensor.bytes().len();
        if let Some(&(buffer_offset, cached_bytes)) = self.tensors.get(&source_offset) {
            if cached_bytes != bytes {
                bail!("mapped tensor changed length after entering the Vulkan weight cache");
            }
            return Ok((self.buffer.descriptor_at(buffer_offset, bytes)?, 0));
        }

        let buffer_offset = align_up(self.next_offset, self.alignment)
            .context("persistent weight-arena offset overflow")?;
        let end = buffer_offset
            .checked_add(bytes)
            .context("persistent weight-arena tensor overflow")?;
        if end > self.buffer.bytes {
            bail!("persistent weight arena is too small for the validated model tensors");
        }
        self.buffer.write_bytes_at(buffer_offset, tensor.bytes())?;
        self.tensors.insert(source_offset, (buffer_offset, bytes));
        self.next_offset = end;
        Ok((self.buffer.descriptor_at(buffer_offset, bytes)?, bytes))
    }
}

struct ImportedMapping {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: usize,
    _mapping: Arc<memmap2::Mmap>,
}

impl ImportedMapping {
    fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        external_host: &vk::ExtExternalMemoryHostFn,
        alignment: u64,
        mapping: Arc<memmap2::Mmap>,
    ) -> Result<Self> {
        let mut failures = Vec::new();
        for (label, handle_type) in [
            (
                "HOST_MAPPED_FOREIGN_MEMORY",
                vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT,
            ),
            (
                "HOST_ALLOCATION",
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            ),
        ] {
            match Self::try_new(
                instance,
                physical,
                device,
                external_host,
                alignment,
                Arc::clone(&mapping),
                handle_type,
            ) {
                Ok(imported) => return Ok(imported),
                Err(error) => failures.push(format!("{label}: {error:#}")),
            }
        }
        bail!(
            "external model mapping import failed for every supported host-pointer type: {}",
            failures.join("; ")
        )
    }

    fn try_new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        external_host: &vk::ExtExternalMemoryHostFn,
        alignment: u64,
        mapping: Arc<memmap2::Mmap>,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<Self> {
        let alignment = usize::try_from(alignment)
            .context("external-host alignment does not fit this platform")?;
        if alignment == 0
            || !alignment.is_power_of_two()
            || mapping.as_ptr() as usize % alignment != 0
        {
            bail!("model mapping does not meet Vulkan external-host alignment");
        }
        let allocation_bytes = mapping
            .len()
            .checked_add(alignment - 1)
            .map(|value| value & !(alignment - 1))
            .context("external-host allocation size overflow")?;

        let mut external_info = vk::ExternalMemoryBufferCreateInfo::builder()
            .handle_types(handle_type)
            .build();
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(mapping.len() as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }
            .map_err(|error| anyhow!("external model buffer creation failed: {error:?}"))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        if requirements.size > allocation_bytes as u64 {
            unsafe { device.destroy_buffer(buffer, None) };
            bail!(
                "external model buffer requires {} bytes but aligned mapping has {allocation_bytes}",
                requirements.size
            );
        }

        let mut host_properties = vk::MemoryHostPointerPropertiesEXT::default();
        let query = unsafe {
            (external_host.get_memory_host_pointer_properties_ext)(
                device.handle(),
                handle_type,
                mapping.as_ptr().cast(),
                &mut host_properties,
            )
        };
        if query != vk::Result::SUCCESS {
            unsafe { device.destroy_buffer(buffer, None) };
            bail!("model mapping cannot be imported by Vulkan: {query:?}");
        }
        let allowed_types = requirements.memory_type_bits & host_properties.memory_type_bits;
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_type = (0..memory_properties.memory_type_count)
            .filter(|index| allowed_types & (1 << index) != 0)
            .filter(|index| {
                !memory_properties.memory_types[*index as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::PROTECTED)
            })
            .filter(|index| memory_heap_size(&memory_properties, *index) >= allocation_bytes as u64)
            .max_by_key(|index| {
                (
                    memory_heap_size(&memory_properties, *index),
                    memory_properties.memory_types[*index as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                )
            });
        let Some(memory_type) = memory_type else {
            unsafe { device.destroy_buffer(buffer, None) };
            bail!("Vulkan reported no compatible memory type for the model mapping");
        };

        let mut import_info = vk::ImportMemoryHostPointerInfoEXT::builder()
            .handle_type(handle_type)
            .host_pointer(mapping.as_ptr().cast_mut().cast())
            .build();
        let allocation = vk::MemoryAllocateInfo::builder()
            .allocation_size(allocation_bytes as u64)
            .memory_type_index(memory_type)
            .push_next(&mut import_info);
        let memory = match unsafe { device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                bail!("external model mapping import failed: {error:?}");
            }
        };
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            bail!("external model buffer bind failed: {error:?}");
        }
        Ok(Self {
            device: device.clone(),
            buffer,
            memory,
            bytes: mapping.len(),
            _mapping: mapping,
        })
    }

    fn descriptor(&self, offset: usize, bytes: usize) -> Result<vk::DescriptorBufferInfo> {
        let end = offset
            .checked_add(bytes)
            .context("mapped tensor descriptor range overflow")?;
        if bytes == 0 || end > self.bytes {
            bail!("mapped tensor descriptor is outside the imported model buffer");
        }
        Ok(vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .offset(offset as u64)
            .range(bytes as u64)
            .build())
    }
}

impl Drop for ImportedMapping {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

struct Buffer {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped_address: Option<usize>,
    bytes: usize,
    allocation_bytes: u64,
    memory_class: BufferMemoryClass,
}

fn memory_heap_size(properties: &vk::PhysicalDeviceMemoryProperties, memory_type: u32) -> u64 {
    let heap = properties.memory_types[memory_type as usize].heap_index as usize;
    properties.memory_heaps[heap].size
}

impl Buffer {
    fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        bytes: usize,
    ) -> Result<Self> {
        Self::new_with_options(
            instance,
            physical,
            device,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferMemoryClass::HostVisible,
        )
    }

    fn new_staging(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        bytes: usize,
    ) -> Result<Self> {
        Self::new_with_options(
            instance,
            physical,
            device,
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemoryClass::HostVisible,
        )
    }

    fn new_device_local_storage(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        bytes: usize,
    ) -> Result<Self> {
        Self::new_with_options(
            instance,
            physical,
            device,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            BufferMemoryClass::DeviceLocal,
        )
    }

    fn new_with_options(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        bytes: usize,
        usage: vk::BufferUsageFlags,
        memory_class: BufferMemoryClass,
    ) -> Result<Self> {
        if bytes == 0 {
            bail!("Vulkan buffers cannot be empty");
        }
        let info = vk::BufferCreateInfo::builder()
            .size(bytes as u64)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None) }
            .map_err(|error| anyhow!("buffer creation failed: {error:?}"))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let required_flags = match memory_class {
            BufferMemoryClass::HostVisible => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
            BufferMemoryClass::DeviceLocal => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        };
        let memory_type = (0..properties.memory_type_count)
            .filter(|index| requirements.memory_type_bits & (1 << index) != 0)
            .filter(|index| {
                properties.memory_types[*index as usize]
                    .property_flags
                    .contains(required_flags)
            })
            .filter(|index| memory_heap_size(&properties, *index) >= requirements.size)
            .max_by_key(|index| {
                let flags = properties.memory_types[*index as usize].property_flags;
                match memory_class {
                    BufferMemoryClass::HostVisible => (
                        memory_heap_size(&properties, *index),
                        flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) as u64,
                    ),
                    BufferMemoryClass::DeviceLocal => (
                        (!flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)) as u64,
                        memory_heap_size(&properties, *index),
                    ),
                }
            });
        let Some(memory_type) = memory_type else {
            unsafe { device.destroy_buffer(buffer, None) };
            bail!("no {memory_class:?} Vulkan memory type supports the requested buffer");
        };
        let allocation = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = match unsafe { device.allocate_memory(&allocation, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                bail!("buffer allocation failed: {error:?}");
            }
        };
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            bail!("buffer bind failed: {error:?}");
        }
        let mapped_address = match memory_class {
            BufferMemoryClass::HostVisible => {
                match unsafe {
                    device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                } {
                    Ok(mapped) => Some(mapped as usize),
                    Err(error) => {
                        unsafe {
                            device.free_memory(memory, None);
                            device.destroy_buffer(buffer, None);
                        }
                        bail!("persistent buffer map failed: {error:?}");
                    }
                }
            }
            BufferMemoryClass::DeviceLocal => None,
        };
        Ok(Self {
            device: device.clone(),
            buffer,
            memory,
            mapped_address,
            bytes,
            allocation_bytes: requirements.size,
            memory_class,
        })
    }

    fn write_bytes(&self, bytes: &[u8]) -> Result<()> {
        self.write_bytes_at(0, bytes)
    }

    fn write_bytes_at(&self, offset: usize, bytes: &[u8]) -> Result<()> {
        let end = offset
            .checked_add(bytes.len())
            .context("Vulkan upload range overflow")?;
        if end > self.bytes {
            bail!("Vulkan upload exceeds buffer allocation");
        }
        let mapped_address = self
            .mapped_address
            .context("cannot write directly to device-local Vulkan memory")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (mapped_address as *mut u8).add(offset),
                bytes.len(),
            )
        };
        Ok(())
    }

    fn read_f32(&self, length: usize) -> Result<Vec<f32>> {
        if length * size_of::<f32>() > self.bytes {
            bail!("Vulkan read exceeds buffer allocation");
        }
        let mapped_address = self
            .mapped_address
            .context("cannot read device-local Vulkan memory without staging")?;
        let values =
            unsafe { std::slice::from_raw_parts(mapped_address as *const f32, length) }.to_vec();
        Ok(values)
    }

    fn descriptor(&self, bytes: usize) -> vk::DescriptorBufferInfo {
        debug_assert!(bytes <= self.bytes);
        vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .offset(0)
            .range(bytes as u64)
            .build()
    }

    fn descriptor_at(&self, offset: usize, bytes: usize) -> Result<vk::DescriptorBufferInfo> {
        let end = offset
            .checked_add(bytes)
            .context("Vulkan descriptor range overflow")?;
        if bytes == 0 || end > self.bytes {
            bail!("Vulkan descriptor is outside its buffer allocation");
        }
        Ok(vk::DescriptorBufferInfo::builder()
            .buffer(self.buffer)
            .offset(offset as u64)
            .range(bytes as u64)
            .build())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped_address.is_some() {
                self.device.unmap_memory(self.memory);
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment.checked_sub(1)?)
        .map(|value| value & !(alignment - 1))
}

fn convolution_tile_rows(dimensions: &[u32; 13]) -> usize {
    const MAX_COLUMN_BYTES: usize = 32 * 1024 * 1024;
    const MAX_SHADER_INVOCATIONS: usize = 65_535 * 256;
    let output_plane = dimensions[11] as usize * dimensions[12] as usize;
    let rows = dimensions[0] as usize * output_plane;
    let width = dimensions[1] as usize * dimensions[5] as usize * dimensions[6] as usize;
    let max_rows_by_bytes = MAX_COLUMN_BYTES / (width * size_of::<u16>());
    let max_rows_by_dispatch = MAX_SHADER_INVOCATIONS / width;
    rows.min(max_rows_by_bytes).min(max_rows_by_dispatch).max(1)
}

fn convolution_column_bytes(dimensions: &[u32; 13]) -> usize {
    let output_plane = dimensions[11] as u64 * dimensions[12] as u64;
    let rows = dimensions[0] as u64 * output_plane;
    let width = dimensions[1] as u64 * dimensions[5] as u64 * dimensions[6] as u64;
    let operations = rows
        .saturating_mul(width)
        .saturating_mul(dimensions[4] as u64);
    if width % 4 != 0 || operations < 50_000_000 {
        return 0;
    }
    convolution_tile_rows(dimensions) * width as usize * size_of::<u16>()
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = 14 - exponent;
        let mut half = (mantissa >> shift) as u16;
        if (mantissa >> (shift - 1)) & 1 != 0 {
            half = half.wrapping_add(1);
        }
        sign | half
    } else if exponent >= 31 {
        sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 }
    } else {
        let mut half = sign | ((exponent as u16) << 10) | (mantissa >> 13) as u16;
        if mantissa & 0x1000 != 0 {
            half = half.wrapping_add(1);
        }
        half
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion_round_trips_normal_values() {
        for value in [-2.0, -0.5, 0.0, 0.25, 1.0, 65_504.0] {
            assert_eq!(crate::dequant::f16_to_f32(f32_to_f16(value)), value);
        }
    }

    #[test]
    fn reports_selected_device_capabilities() {
        let profile = match device_profile() {
            Ok(profile) => profile,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping Vulkan device profile: {error:#}");
                return;
            }
            Err(error) => panic!("required Vulkan device profile failed: {error:#}"),
        };
        eprintln!("{profile}");
        assert!(!profile.device_name.is_empty());
        assert!(profile.device_local_memory_bytes > 0);
        assert!(profile.available_device_memory_bytes > 0);
        assert!(profile.max_storage_buffer_bytes > 0);
        assert!(profile.max_memory_allocation_bytes > 0);
        assert!(profile.max_workgroup_invocations >= 256);
        assert!(profile.max_workgroup_size[0] >= 256);
        assert!(profile.subgroup_size > 0);
        assert!(profile.fp16_supported);
        assert!(profile.storage_buffer_alignment > 0);
    }

    #[test]
    fn chunked_attention_preserves_query_row_offsets() {
        if with_runtime(|_| Ok(())).is_err() {
            eprintln!(
                "skipping chunked_attention_preserves_query_row_offsets: no usable Vulkan device here"
            );
            return;
        }
        let query = Tensor::new(
            vec![1, 1, 65, 4],
            (0..65).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(),
        )
        .unwrap();
        let key = Tensor::new(
            vec![1, 1, 3, 4],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let value = Tensor::new(
            vec![1, 1, 3, 4],
            vec![
                2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 30.0, 40.0, 3.0, 6.0, 9.0, 12.0,
            ],
        )
        .unwrap();
        let output = attention(&query, &key, &value, false).unwrap();
        let emphasized = 0.5f32.exp();
        let denominator = emphasized + 2.0;
        let expected = [
            (emphasized * 2.0 + 10.0 + 3.0) / denominator,
            (emphasized * 4.0 + 20.0 + 6.0) / denominator,
            (emphasized * 6.0 + 30.0 + 9.0) / denominator,
            (emphasized * 8.0 + 40.0 + 12.0) / denominator,
        ];
        assert_eq!(output.shape(), &[1, 1, 65, 4]);
        for (row_index, row) in output.data().chunks_exact(4).enumerate() {
            for (column, (&actual, &expected)) in row.iter().zip(&expected).enumerate() {
                assert!(
                    (actual - expected).abs() < 2e-3,
                    "row {row_index}, column {column}: {actual} != {expected}"
                );
            }
        }
    }

    fn write_linear_fixture(path: &std::path::Path, tensor_count: usize, out: usize, width: usize) {
        let elements = out * width;
        let bytes_len = elements * 2;
        let mut header_entries = Vec::new();
        let mut payload = Vec::new();
        for index in 0..tensor_count {
            let start = index * bytes_len;
            header_entries.push(format!(
                "\"w{index}\":{{\"dtype\":\"F16\",\"shape\":[{out},{width}],\"data_offsets\":[{start},{}]}}",
                start + bytes_len
            ));
            for element in 0..elements {
                let value = ((index * elements + element) as f32 * 0.01) - 1.0;
                payload.extend_from_slice(&f32_to_f16(value).to_le_bytes());
            }
        }
        let header = format!("{{{}}}", header_entries.join(","));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(path, bytes).unwrap();
    }

    /// Proves two things about the staged-loading mechanism added for the mobile SDXL
    /// work: (1) a bounded arena that evicts and reuses its buffer between
    /// `begin_weight_stage` calls produces bit-for-bit-equivalent results to the default
    /// whole-file cache, and (2) the byte cap is real — touching a second tensor without
    /// resetting a too-small arena fails loudly instead of silently overrunning it.
    #[test]
    fn staged_weight_loading_matches_whole_file_cache() {
        if with_runtime(|_| Ok(())).is_err() {
            eprintln!(
                "skipping staged_weight_loading_matches_whole_file_cache: no usable Vulkan device here"
            );
            return;
        }

        const TENSORS: usize = 6;
        const OUT: usize = 64;
        const WIDTH: usize = 256;
        let input = Tensor::new(
            vec![1, WIDTH],
            (0..WIDTH)
                .map(|index| ((index % 13) as f32 - 6.0) / 6.0)
                .collect(),
        )
        .unwrap();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        disable_staged_weight_loading();
        let baseline_path =
            std::env::temp_dir().join(format!("quartz-stage-baseline-{nonce}.safetensors"));
        write_linear_fixture(&baseline_path, TENSORS, OUT, WIDTH);
        let baseline_weights = crate::safetensors::SafeTensorFile::open(&baseline_path).unwrap();
        let mut baseline_outputs = Vec::new();
        for index in 0..TENSORS {
            let weight = baseline_weights.mapped(&format!("w{index}")).unwrap();
            baseline_outputs.push(linear(&input, weight, None).unwrap());
        }

        let one_tensor_bytes = OUT * WIDTH * 2;
        let staged_path =
            std::env::temp_dir().join(format!("quartz-stage-staged-{nonce}.safetensors"));
        write_linear_fixture(&staged_path, TENSORS, OUT, WIDTH);
        let staged_weights = crate::safetensors::SafeTensorFile::open(&staged_path).unwrap();
        enable_staged_weight_loading(staged_weights.mapping_key(), one_tensor_bytes, 1);
        let mut staged_outputs = Vec::new();
        for index in 0..TENSORS {
            begin_weight_stage().unwrap();
            let weight = staged_weights.mapped(&format!("w{index}")).unwrap();
            staged_outputs.push(linear(&input, weight, None).unwrap());
        }
        disable_staged_weight_loading();

        assert_eq!(baseline_outputs.len(), staged_outputs.len());
        for (index, (base, staged)) in baseline_outputs
            .iter()
            .zip(staged_outputs.iter())
            .enumerate()
        {
            assert_eq!(
                base.shape(),
                staged.shape(),
                "tensor {index} shape mismatch"
            );
            for (b, s) in base.data().iter().zip(staged.data().iter()) {
                assert!((b - s).abs() < 1e-3, "tensor {index}: {b} != {s}");
            }
        }

        let overflow_path =
            std::env::temp_dir().join(format!("quartz-stage-overflow-{nonce}.safetensors"));
        write_linear_fixture(&overflow_path, 2, OUT, WIDTH);
        let overflow_weights = crate::safetensors::SafeTensorFile::open(&overflow_path).unwrap();
        enable_staged_weight_loading(overflow_weights.mapping_key(), one_tensor_bytes, 1);
        linear(&input, overflow_weights.mapped("w0").unwrap(), None).unwrap();
        // No begin_weight_stage() here: touching a second tensor must overflow the
        // single-tensor arena rather than silently growing past its reserved bytes.
        let error = linear(&input, overflow_weights.mapped("w1").unwrap(), None).unwrap_err();
        assert!(error.to_string().contains("too small"), "{error}");
        disable_staged_weight_loading();

        // The runtime normally retains model arenas across requests. These mappings
        // belong to temporary test files, so keeping them alive past this test leaks
        // both the mmap and its Vulkan allocation into every later test in the same
        // process.
        release_sd_resources().unwrap();

        let _ = std::fs::remove_file(&baseline_path);
        let _ = std::fs::remove_file(&staged_path);
        let _ = std::fs::remove_file(&overflow_path);
    }
}
