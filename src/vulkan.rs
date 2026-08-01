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
static SD_ACCELERATION: AtomicBool = AtomicBool::new(false);
static RUNTIME: OnceLock<Result<Mutex<VulkanRuntime>, String>> = OnceLock::new();

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
    gemm: KernelStats,
    conv2d: KernelStats,
    attention: KernelStats,
    uploaded_bytes: u64,
    cached_weight_bytes: u64,
    peak_dispatch_bytes: u64,
}

#[derive(Clone, Copy)]
enum KernelKind {
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

pub struct BenchmarkResult {
    device: String,
    rows: u32,
    outputs: u32,
    width: u32,
    dispatch_milliseconds: f64,
    wall_milliseconds: f64,
    max_error: f32,
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
        "Vulkan profile: gemm={} calls/{:.3} dispatch/{:.3} wall ms conv={} calls/{:.3} dispatch/{:.3} wall ms attention={} calls/{:.3} dispatch/{:.3} wall ms uploaded={} bytes cached_weights={} bytes peak_dispatch={} bytes",
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
    );
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
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    device_name: String,
    stats: RuntimeStats,
    buffers: [Option<Buffer>; 8],
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
            let physical_devices = unsafe { instance.enumerate_physical_devices() }
                .map_err(|error| anyhow!("cannot enumerate Vulkan devices: {error:?}"))?;
            let physical = *physical_devices
                .first()
                .context("no Vulkan physical device")?;
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            const REQUIRED_WORKGROUP_INVOCATIONS: u32 = 256;
            const REQUIRED_SHARED_MEMORY: u32 = 4096 * 4 + 256 * 4;
            if properties.limits.max_compute_work_group_invocations < REQUIRED_WORKGROUP_INVOCATIONS
                || properties.limits.max_compute_work_group_size[0] < REQUIRED_WORKGROUP_INVOCATIONS
                || properties.limits.max_compute_shared_memory_size < REQUIRED_SHARED_MEMORY
            {
                bail!(
                    "Vulkan device {device_name:?} is below the Quartz SD compute limits: workgroup_invocations={} workgroup_x={} shared_memory={} (need 256, 256, {REQUIRED_SHARED_MEMORY})",
                    properties.limits.max_compute_work_group_invocations,
                    properties.limits.max_compute_work_group_size[0],
                    properties.limits.max_compute_shared_memory_size,
                );
            }
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(physical) };
            let queue_family = queue_families
                .iter()
                .position(|family| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .context("Vulkan device has no compute queue")?
                as u32;

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
            if std::env::var("QUARTZ_DEBUG_ARENA_SIZE").is_ok() {
                let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::default();
                let mut properties2 = vk::PhysicalDeviceProperties2::builder()
                    .push_next(&mut maintenance3)
                    .build();
                unsafe { instance.get_physical_device_properties2(physical, &mut properties2) };
                let memory_properties =
                    unsafe { instance.get_physical_device_memory_properties(physical) };
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
                device_name,
                external_host_supported,
                external_host_properties.min_imported_host_pointer_alignment,
                properties.limits.min_storage_buffer_offset_alignment,
            ))
        })();
        let (
            physical,
            device,
            queue_family,
            device_name,
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
            descriptor_pool: vk::DescriptorPool::null(),
            command_pool: vk::CommandPool::null(),
            device_name,
            stats: RuntimeStats::default(),
            buffers: std::array::from_fn(|_| None),
            external_host,
            external_host_alignment,
            storage_buffer_alignment,
            model_mappings: HashMap::new(),
        };
        runtime.initialize_resources()?;
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
    mapped_address: usize,
    bytes: usize,
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
        if bytes == 0 {
            bail!("Vulkan buffers cannot be empty");
        }
        let info = vk::BufferCreateInfo::builder()
            .size(bytes as u64)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None) }
            .map_err(|error| anyhow!("buffer creation failed: {error:?}"))?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_type = (0..properties.memory_type_count)
            .filter(|index| requirements.memory_type_bits & (1 << index) != 0)
            .filter(|index| {
                properties.memory_types[*index as usize]
                    .property_flags
                    .contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .filter(|index| memory_heap_size(&properties, *index) >= requirements.size)
            .max_by_key(|index| {
                (
                    memory_heap_size(&properties, *index),
                    properties.memory_types[*index as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                )
            });
        let Some(memory_type) = memory_type else {
            unsafe { device.destroy_buffer(buffer, None) };
            bail!("no host-visible coherent Vulkan memory type");
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
        let mapped = match unsafe {
            device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        } {
            Ok(mapped) => mapped,
            Err(error) => {
                unsafe {
                    device.free_memory(memory, None);
                    device.destroy_buffer(buffer, None);
                }
                bail!("persistent buffer map failed: {error:?}");
            }
        };
        Ok(Self {
            device: device.clone(),
            buffer,
            memory,
            mapped_address: mapped as usize,
            bytes,
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
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (self.mapped_address as *mut u8).add(offset),
                bytes.len(),
            )
        };
        Ok(())
    }

    fn read_f32(&self, length: usize) -> Result<Vec<f32>> {
        if length * size_of::<f32>() > self.bytes {
            bail!("Vulkan read exceeds buffer allocation");
        }
        let values =
            unsafe { std::slice::from_raw_parts(self.mapped_address as *const f32, length) }
                .to_vec();
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
            self.device.unmap_memory(self.memory);
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

        let _ = std::fs::remove_file(&baseline_path);
        let _ = std::fs::remove_file(&staged_path);
        let _ = std::fs::remove_file(&overflow_path);
    }
}
