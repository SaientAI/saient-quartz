//! Backend boundary for the verified tensor graph.
//!
//! The scalar implementation remains the reference. Operations are moved
//! behind this interface incrementally, with parity tests added before another
//! implementation is allowed to service them.

use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendKind {
    ScalarCpu,
    #[cfg(feature = "vulkan")]
    Vulkan,
}

#[derive(Clone)]
pub(crate) struct DeviceTensor {
    backend: BackendKind,
    shape: Vec<usize>,
    storage: DeviceTensorStorage,
}

#[derive(Clone)]
enum DeviceTensorStorage {
    Host(Arc<Tensor>),
    #[cfg(feature = "vulkan")]
    Vulkan(crate::vulkan::ResidentTensor),
}

impl DeviceTensorStorage {
    fn host(&self) -> Result<&Tensor> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => bail!("operation requires host tensor storage"),
        }
    }
}

impl DeviceTensor {
    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub(crate) fn byte_len(&self) -> usize {
        self.len() * std::mem::size_of::<f32>()
    }

    pub(crate) fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    pub(crate) fn remains_resident(&self) -> bool {
        match &self.storage {
            DeviceTensorStorage::Host(_) => false,
            #[cfg(feature = "vulkan")]
            DeviceTensorStorage::Vulkan(_) => true,
        }
    }

    pub(crate) fn is_device_local(&self) -> Result<bool> {
        match &self.storage {
            DeviceTensorStorage::Host(_) => Ok(false),
            #[cfg(feature = "vulkan")]
            DeviceTensorStorage::Vulkan(storage) => {
                crate::vulkan::resident_tensor_is_device_local(storage)
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct LinearWeightHandle {
    backend: BackendKind,
    input_width: usize,
    output_width: usize,
    storage: LinearWeightStorage,
}

#[derive(Clone)]
enum LinearWeightStorage {
    Host {
        weight: Arc<Tensor>,
        bias: Option<Arc<Tensor>>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan(crate::vulkan::ResidentLinearWeights),
}

#[derive(Clone)]
pub(crate) struct Conv3dWeightHandle {
    backend: BackendKind,
    input_channels: usize,
    output_channels: usize,
    kernel: [usize; 3],
    storage: Conv3dWeightStorage,
}

#[derive(Clone)]
enum Conv3dWeightStorage {
    Host {
        weight: Arc<Tensor>,
        bias: Option<Arc<Tensor>>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan(crate::vulkan::ResidentConv3dWeights),
}

impl Conv3dWeightStorage {
    fn host(&self) -> Result<(&Tensor, Option<&Tensor>)> {
        match self {
            Self::Host { weight, bias } => Ok((weight, bias.as_deref())),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => bail!("operation requires host Conv3D weight storage"),
        }
    }
}

impl LinearWeightStorage {
    fn host(&self) -> Result<(&Tensor, Option<&Tensor>)> {
        match self {
            Self::Host { weight, bias } => Ok((weight, bias.as_deref())),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => bail!("operation requires host weight storage"),
        }
    }
}

#[derive(Clone)]
pub(crate) struct PreparedVectorHandle {
    backend: BackendKind,
    length: usize,
    storage: PreparedVectorStorage,
}

#[derive(Clone)]
enum PreparedVectorStorage {
    Host(Arc<Tensor>),
    #[cfg(feature = "vulkan")]
    Vulkan(crate::vulkan::ResidentTensor),
}

impl PreparedVectorStorage {
    fn host(&self) -> Result<&Tensor> {
        match self {
            Self::Host(tensor) => Ok(tensor),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => bail!("operation requires host vector storage"),
        }
    }
}

pub(crate) trait TensorBackend: Send + Sync {
    fn kind(&self) -> BackendKind;
    fn name(&self) -> &'static str;

    /// Elementwise addition of equal-shaped FP32 tensors.
    fn add(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;

    fn multiply(&self, left: &Tensor, right: &Tensor) -> Result<Tensor>;
    fn scale(&self, input: &Tensor, value: f32) -> Result<Tensor>;
    fn silu(&self, input: &Tensor) -> Result<Tensor>;
    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor>;
    fn clamp(&self, input: &Tensor, minimum: f32, maximum: f32) -> Result<Tensor>;
    fn channel_rms_norm_3d(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor>;
    fn linear(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor>;

    /// Move a host tensor into storage owned by this backend. The default
    /// scalar implementation retains a shared host tensor.
    fn upload_tensor(&self, input: &Tensor) -> Result<DeviceTensor> {
        Ok(DeviceTensor {
            backend: self.kind(),
            shape: input.shape().to_vec(),
            storage: DeviceTensorStorage::Host(Arc::new(input.clone())),
        })
    }

    fn download_tensor(&self, input: &DeviceTensor) -> Result<Tensor> {
        self.require_tensor(input)?;
        match &input.storage {
            DeviceTensorStorage::Host(tensor) => Ok(tensor.as_ref().clone()),
            #[cfg(feature = "vulkan")]
            DeviceTensorStorage::Vulkan(_) => {
                bail!("non-Vulkan backend cannot download Vulkan storage")
            }
        }
    }

    fn ncthw_slice_time_device(
        &self,
        input: &DeviceTensor,
        start: usize,
        count: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let [batch, channels, time, height, width]: [usize; 5] = input
            .shape()
            .try_into()
            .context("temporal slice input must be NCTHW")?;
        if count == 0 {
            bail!("temporal slice cannot select zero frames");
        }
        let end = start
            .checked_add(count)
            .context("temporal slice range overflow")?;
        if end > time {
            bail!("temporal slice {start}..{end} exceeds input time {time}");
        }
        let input = input.storage.host()?;
        let plane = height * width;
        let mut data = Vec::with_capacity(batch * channels * count * plane);
        for sample in 0..batch {
            for channel in 0..channels {
                let source = ((sample * channels + channel) * time + start) * plane;
                data.extend_from_slice(&input.data()[source..source + count * plane]);
            }
        }
        self.upload_tensor(&Tensor::new(
            vec![batch, channels, count, height, width],
            data,
        )?)
    }

    fn ncthw_concat_time_device(&self, inputs: &[&DeviceTensor]) -> Result<DeviceTensor> {
        let first = inputs
            .first()
            .context("temporal concat requires at least one tensor")?;
        self.require_tensor(first)?;
        let [batch, channels, _, height, width]: [usize; 5] = first
            .shape()
            .try_into()
            .context("temporal concat input must be NCTHW")?;
        let mut total_time = 0usize;
        for input in inputs {
            self.require_tensor(input)?;
            let [
                input_batch,
                input_channels,
                input_time,
                input_height,
                input_width,
            ]: [usize; 5] = input
                .shape()
                .try_into()
                .context("temporal concat input must be NCTHW")?;
            if [input_batch, input_channels, input_height, input_width]
                != [batch, channels, height, width]
            {
                bail!(
                    "temporal concat non-time dimensions differ: {:?} vs {:?}",
                    first.shape(),
                    input.shape()
                );
            }
            total_time = total_time
                .checked_add(input_time)
                .context("temporal concat length overflow")?;
        }
        let plane = height * width;
        let mut data = Vec::with_capacity(batch * channels * total_time * plane);
        for sample in 0..batch {
            for channel in 0..channels {
                for input in inputs {
                    let input_time = input.shape()[2];
                    let input = input.storage.host()?;
                    let source = (sample * channels + channel) * input_time * plane;
                    data.extend_from_slice(&input.data()[source..source + input_time * plane]);
                }
            }
        }
        self.upload_tensor(&Tensor::new(
            vec![batch, channels, total_time, height, width],
            data,
        )?)
    }

    fn ncthw_prepend_zero_time_device(
        &self,
        input: &DeviceTensor,
        count: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        if count == 0 {
            return Ok(input.clone());
        }
        let [batch, channels, time, height, width]: [usize; 5] = input
            .shape()
            .try_into()
            .context("zero-time prepend input must be NCTHW")?;
        let output_time = time
            .checked_add(count)
            .context("zero-time prepend length overflow")?;
        let plane = height * width;
        let input = input.storage.host()?;
        let mut data = vec![0.0; batch * channels * output_time * plane];
        for sample in 0..batch {
            for channel in 0..channels {
                let source = (sample * channels + channel) * time * plane;
                let destination = ((sample * channels + channel) * output_time + count) * plane;
                data[destination..destination + time * plane]
                    .copy_from_slice(&input.data()[source..source + time * plane]);
            }
        }
        self.upload_tensor(&Tensor::new(
            vec![batch, channels, output_time, height, width],
            data,
        )?)
    }

    fn ncthw_channels_to_time_device(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let [batch, doubled_channels, time, height, width]: [usize; 5] =
            input
                .shape()
                .try_into()
                .context("channel-to-time input must be NCTHW")?;
        if doubled_channels % 2 != 0 {
            bail!("Wan channel-to-time shuffle needs an even channel count");
        }
        let channels = doubled_channels / 2;
        let output_time = time
            .checked_mul(2)
            .context("channel-to-time length overflow")?;
        let plane = height * width;
        let input = input.storage.host()?;
        let mut data = vec![0.0; input.len()];
        for sample in 0..batch {
            for channel in 0..channels {
                for input_time in 0..time {
                    for half in 0..2 {
                        // Exact Wan mapping:
                        // input[n, half*C+c, t, h, w] -> output[n, c, 2*t+half, h, w].
                        let input_channel = half * channels + channel;
                        let source = ((sample * doubled_channels + input_channel) * time
                            + input_time)
                            * plane;
                        let destination =
                            ((sample * channels + channel) * output_time + input_time * 2 + half)
                                * plane;
                        data[destination..destination + plane]
                            .copy_from_slice(&input.data()[source..source + plane]);
                    }
                }
            }
        }
        self.upload_tensor(&Tensor::new(
            vec![batch, channels, output_time, height, width],
            data,
        )?)
    }

    /// Prepare one dense projection. Matrix and optional bias ownership are
    /// grouped so an accelerated backend can upload them exactly once.
    fn prepare_linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<LinearWeightHandle> {
        let [output_width, input_width]: [usize; 2] = weight
            .shape()
            .try_into()
            .context("linear weight must be rank two")?;
        if let Some(bias) = bias
            && bias.shape() != [output_width]
        {
            bail!(
                "linear bias shape {:?} must be [{output_width}]",
                bias.shape()
            );
        }
        Ok(LinearWeightHandle {
            backend: self.kind(),
            input_width,
            output_width,
            storage: LinearWeightStorage::Host {
                weight: Arc::new(weight.clone()),
                bias: bias.cloned().map(Arc::new),
            },
        })
    }

    fn linear_prepared(
        &self,
        input: &DeviceTensor,
        weights: &LinearWeightHandle,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_weights(weights)?;
        if input.shape().last().copied() != Some(weights.input_width) {
            bail!(
                "prepared linear input shape {:?} does not end in {}",
                input.shape(),
                weights.input_width
            );
        }
        let input = input.storage.host()?;
        let (weight, bias) = weights.storage.host()?;
        let output = self.linear(input, weight, bias)?;
        if output.shape().last().copied() != Some(weights.output_width) {
            bail!(
                "prepared linear output shape {:?} does not end in {}",
                output.shape(),
                weights.output_width
            );
        }
        self.upload_tensor(&output)
    }

    fn prepare_conv3d(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<Conv3dWeightHandle> {
        let [
            output_channels,
            input_channels,
            kernel_time,
            kernel_height,
            kernel_width,
        ]: [usize; 5] = weight
            .shape()
            .try_into()
            .context("Conv3D weight must be [out,in,time,height,width]")?;
        if output_channels == 0
            || input_channels == 0
            || kernel_time == 0
            || kernel_height == 0
            || kernel_width == 0
        {
            bail!("Conv3D weight dimensions must be non-zero");
        }
        if let Some(bias) = bias
            && bias.shape() != [output_channels]
        {
            bail!(
                "Conv3D bias shape {:?} must be [{output_channels}]",
                bias.shape()
            );
        }
        Ok(Conv3dWeightHandle {
            backend: self.kind(),
            input_channels,
            output_channels,
            kernel: [kernel_time, kernel_height, kernel_width],
            storage: Conv3dWeightStorage::Host {
                weight: Arc::new(weight.clone()),
                bias: bias.cloned().map(Arc::new),
            },
        })
    }

    fn conv3d_prepared_device(
        &self,
        input: &DeviceTensor,
        weights: &Conv3dWeightHandle,
        padding_before: [usize; 3],
        padding_after: [usize; 3],
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_conv3d_weights(weights)?;
        let [batch, input_channels, input_time, input_height, input_width]: [usize; 5] = input
            .shape()
            .try_into()
            .context("Conv3D input must be [batch,channels,time,height,width]")?;
        if batch == 0 || input_channels != weights.input_channels {
            bail!("Conv3D input batch/channels do not match prepared weights");
        }
        for axis in 0..3 {
            let input_axis = [input_time, input_height, input_width][axis];
            let padded = input_axis
                .checked_add(padding_before[axis])
                .and_then(|value| value.checked_add(padding_after[axis]))
                .context("Conv3D padded dimension overflow")?;
            if padded < weights.kernel[axis] {
                bail!("Conv3D kernel is larger than the padded input");
            }
        }
        let input = input.storage.host()?;
        let (weight, bias) = weights.storage.host()?;
        let output = input.conv3d(
            weight,
            bias,
            [1, 1, 1],
            padding_before,
            padding_after,
            [1, 1, 1],
            1,
        )?;
        self.upload_tensor(&output)
    }

    fn silu_device(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let input = input.storage.host()?;
        let output = self.silu(input)?;
        self.upload_tensor(&output)
    }

    fn gelu_tanh_device(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let input = input.storage.host()?;
        let output = self.gelu_tanh(input)?;
        self.upload_tensor(&output)
    }

    fn add_device(&self, left: &DeviceTensor, right: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(left)?;
        self.require_tensor(right)?;
        if left.shape() != right.shape() {
            bail!(
                "resident add shape mismatch: {:?} vs {:?}",
                left.shape(),
                right.shape()
            );
        }
        let left = left.storage.host()?;
        let right = right.storage.host()?;
        self.upload_tensor(&self.add(left, right)?)
    }

    fn scale_device(&self, input: &DeviceTensor, value: f32) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        if !value.is_finite() {
            bail!("resident scale must be finite");
        }
        let input = input.storage.host()?;
        self.upload_tensor(&self.scale(input, value)?)
    }

    /// Prepare a persistent FP32 vector such as a normalization parameter or
    /// Wan block-modulation bias.
    fn prepare_vector(&self, vector: &Tensor) -> Result<PreparedVectorHandle> {
        let [length]: [usize; 1] = vector
            .shape()
            .try_into()
            .context("prepared vector must be rank one")?;
        if length == 0 {
            bail!("prepared vector cannot be empty");
        }
        Ok(PreparedVectorHandle {
            backend: self.kind(),
            length,
            storage: PreparedVectorStorage::Host(Arc::new(vector.clone())),
        })
    }

    fn add_vector_device(
        &self,
        input: &DeviceTensor,
        vector: &PreparedVectorHandle,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_vector(vector)?;
        if input.shape.iter().product::<usize>() != vector.length {
            bail!(
                "prepared vector length {} does not match tensor shape {:?}",
                vector.length,
                input.shape()
            );
        }
        let input = input.storage.host()?;
        let vector = vector.storage.host()?;
        let output = Tensor::new(
            input.shape().to_vec(),
            input
                .data()
                .iter()
                .zip(vector.data())
                .map(|(input, vector)| input + vector)
                .collect(),
        )?;
        self.upload_tensor(&output)
    }

    /// Normalize the final axis. Weight and bias must either both be absent
    /// (Wan norm1/norm2) or both be prepared vectors (Wan norm3).
    fn layer_norm_device(
        &self,
        input: &DeviceTensor,
        weight: Option<&PreparedVectorHandle>,
        bias: Option<&PreparedVectorHandle>,
        epsilon: f32,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("LayerNorm epsilon must be finite and positive");
        }
        if weight.is_some() != bias.is_some() {
            bail!("affine LayerNorm requires both weight and bias");
        }
        let width = input
            .shape()
            .last()
            .copied()
            .context("LayerNorm input must have at least one dimension")?;
        if width == 0 {
            bail!("LayerNorm width cannot be zero");
        }
        let affine = match (weight, bias) {
            (Some(weight), Some(bias)) => {
                self.require_vector(weight)?;
                self.require_vector(bias)?;
                if weight.length != width || bias.length != width {
                    bail!(
                        "LayerNorm weight/bias lengths {}/{} do not match width {width}",
                        weight.length,
                        bias.length
                    );
                }
                Some((weight.storage.host()?, bias.storage.host()?))
            }
            (None, None) => None,
            _ => unreachable!(),
        };
        let input = input.storage.host()?;
        let mut output = input.data().to_vec();
        for row in output.chunks_exact_mut(width) {
            let count = width as f32;
            let mean = row.iter().sum::<f32>() / count;
            let variance = row
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / count;
            let inverse_standard_deviation = 1.0 / (variance + epsilon).sqrt();
            for (column, value) in row.iter_mut().enumerate() {
                *value = (*value - mean) * inverse_standard_deviation;
                if let Some((weight, bias)) = affine {
                    *value = *value * weight.data()[column] + bias.data()[column];
                }
            }
        }
        self.upload_tensor(&Tensor::new(input.shape().to_vec(), output)?)
    }

    /// Normalize the final axis by its root mean square and apply one learned
    /// weight per channel. Wan applies this to Q and K before splitting heads.
    fn rms_norm_device(
        &self,
        input: &DeviceTensor,
        weight: &PreparedVectorHandle,
        epsilon: f32,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_vector(weight)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("RMSNorm epsilon must be finite and positive");
        }
        let width = input
            .shape()
            .last()
            .copied()
            .context("RMSNorm input must have at least one dimension")?;
        if width == 0 || weight.length != width {
            bail!(
                "RMSNorm weight length {} does not match input width {width}",
                weight.length
            );
        }
        let input = input.storage.host()?;
        let weight = weight.storage.host()?;
        let mut output = input.data().to_vec();
        for row in output.chunks_exact_mut(width) {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
            let inverse_rms = 1.0 / (mean_square + epsilon).sqrt();
            for (column, value) in row.iter_mut().enumerate() {
                *value = *value * inverse_rms * weight.data()[column];
            }
        }
        self.upload_tensor(&Tensor::new(input.shape().to_vec(), output)?)
    }

    /// Wan VAE RMSNorm reduces across channels independently at every NTHW
    /// location rather than over the final contiguous axis.
    fn channel_rms_norm_3d_device(
        &self,
        input: &DeviceTensor,
        weight: &PreparedVectorHandle,
        epsilon: f32,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_vector(weight)?;
        let [_, channels, _, _, _]: [usize; 5] = input
            .shape()
            .try_into()
            .context("channel RMSNorm input must be NCTHW")?;
        if channels == 0 || weight.length != channels {
            bail!(
                "channel RMSNorm weight length {} does not match {channels} channels",
                weight.length
            );
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("channel RMSNorm epsilon must be finite and positive");
        }
        let input = input.storage.host()?;
        let weight = weight.storage.host()?;
        self.upload_tensor(&input.channel_rms_norm_3d(weight, epsilon)?)
    }

    /// Apply Wan's position-local rotary matrix to every head. Positions are
    /// `[rows, head_dim / 2, 4]` and are shared across heads within a row.
    fn rope_device(
        &self,
        input: &DeviceTensor,
        positions: &DeviceTensor,
        heads: usize,
        head_dim: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_tensor(positions)?;
        if heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
            bail!("RoPE heads and even head dimension must be non-zero");
        }
        let width = heads.checked_mul(head_dim).context("RoPE width overflow")?;
        if input.shape().last().copied() != Some(width) {
            bail!(
                "RoPE input shape {:?} does not end in width {width}",
                input.shape()
            );
        }
        let rows = input.shape().iter().product::<usize>() / width;
        let position_values = rows
            .checked_mul(head_dim / 2)
            .and_then(|values| values.checked_mul(crate::wan_rope::PAIR_STRIDE))
            .context("RoPE position size overflow")?;
        let actual_position_values = positions.shape().iter().product::<usize>();
        if actual_position_values != position_values {
            bail!(
                "RoPE positions contain {} values, expected {position_values}",
                actual_position_values
            );
        }
        let input = input.storage.host()?;
        let positions = positions.storage.host()?;
        let mut output = input.data().to_vec();
        let position_stride = head_dim / 2 * crate::wan_rope::PAIR_STRIDE;
        for row in 0..rows {
            let position = &positions.data()[row * position_stride..(row + 1) * position_stride];
            for head in 0..heads {
                let offset = row * width + head * head_dim;
                crate::wan_rope::apply_rope(&mut output[offset..offset + head_dim], position);
            }
        }
        self.upload_tensor(&Tensor::new(input.shape().to_vec(), output)?)
    }

    /// Head-major `[heads, queries, keys]` scaled QK scores from row-major
    /// `[queries, heads * head_dim]` and `[keys, heads * head_dim]` inputs.
    fn attention_scores_device(
        &self,
        query: &DeviceTensor,
        key: &DeviceTensor,
        heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<DeviceTensor> {
        self.require_tensor(query)?;
        self.require_tensor(key)?;
        if heads == 0 || head_dim == 0 || !scale.is_finite() {
            bail!("attention score dimensions and scale must be valid");
        }
        let width = heads
            .checked_mul(head_dim)
            .context("attention score width overflow")?;
        let [queries, query_width]: [usize; 2] = query
            .shape()
            .try_into()
            .context("attention query must be rank two")?;
        let [keys, key_width]: [usize; 2] = key
            .shape()
            .try_into()
            .context("attention key must be rank two")?;
        if queries == 0 || keys == 0 || query_width != width || key_width != width {
            bail!("attention Q/K shapes do not match the requested head layout");
        }
        let query = query.storage.host()?;
        let key = key.storage.host()?;
        let mut scores = vec![0.0; heads * queries * keys];
        for head in 0..heads {
            let head_offset = head * head_dim;
            for query_row in 0..queries {
                for key_row in 0..keys {
                    let mut dot = 0.0;
                    for channel in 0..head_dim {
                        dot += query.data()[query_row * width + head_offset + channel]
                            * key.data()[key_row * width + head_offset + channel];
                    }
                    scores[(head * queries + query_row) * keys + key_row] = dot * scale;
                }
            }
        }
        self.upload_tensor(&Tensor::new(vec![heads, queries, keys], scores)?)
    }

    fn softmax_device(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let width = input
            .shape()
            .last()
            .copied()
            .context("softmax input must have at least one dimension")?;
        if width == 0 {
            bail!("softmax width cannot be zero");
        }
        let input = input.storage.host()?;
        let mut output = input.data().to_vec();
        for row in output.chunks_exact_mut(width) {
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for value in row.iter_mut() {
                *value = (*value - maximum).exp();
                sum += *value;
            }
            let inverse_sum = 1.0 / sum;
            for value in row.iter_mut() {
                *value *= inverse_sum;
            }
        }
        self.upload_tensor(&Tensor::new(input.shape().to_vec(), output)?)
    }

    /// Multiply head-major probabilities by row-major values and merge heads
    /// back into `[queries, heads * head_dim]`.
    fn attention_values_device(
        &self,
        probabilities: &DeviceTensor,
        value: &DeviceTensor,
        heads: usize,
        head_dim: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(probabilities)?;
        self.require_tensor(value)?;
        let [probability_heads, queries, keys]: [usize; 3] = probabilities
            .shape()
            .try_into()
            .context("attention probabilities must be rank three")?;
        let [value_rows, value_width]: [usize; 2] = value
            .shape()
            .try_into()
            .context("attention value must be rank two")?;
        let width = heads
            .checked_mul(head_dim)
            .context("attention value width overflow")?;
        if heads == 0
            || head_dim == 0
            || probability_heads != heads
            || queries == 0
            || keys == 0
            || value_rows != keys
            || value_width != width
        {
            bail!("attention probability/value shapes do not match the head layout");
        }
        let probabilities = probabilities.storage.host()?;
        let value = value.storage.host()?;
        let mut output = vec![0.0; queries * width];
        for query in 0..queries {
            for head in 0..heads {
                for key in 0..keys {
                    let probability = probabilities.data()[(head * queries + query) * keys + key];
                    for channel in 0..head_dim {
                        output[query * width + head * head_dim + channel] +=
                            probability * value.data()[key * width + head * head_dim + channel];
                    }
                }
            }
        }
        self.upload_tensor(&Tensor::new(vec![queries, width], output)?)
    }

    fn wan_modulate_device(
        &self,
        input: &DeviceTensor,
        modulation: &DeviceTensor,
        shift_chunk: usize,
        scale_chunk: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_tensor(modulation)?;
        let width = input
            .shape()
            .last()
            .copied()
            .context("Wan modulation input must have at least one dimension")?;
        let required_chunks = shift_chunk.max(scale_chunk) + 1;
        let required_values = required_chunks
            .checked_mul(width)
            .context("Wan modulation size overflow")?;
        let modulation = modulation.storage.host()?;
        if modulation.len() < required_values {
            bail!(
                "Wan modulation has {} values, requires at least {required_values}",
                modulation.len()
            );
        }
        let input = input.storage.host()?;
        let output = input
            .data()
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let channel = index % width;
                let shift = modulation.data()[shift_chunk * width + channel];
                let scale = modulation.data()[scale_chunk * width + channel];
                value * (1.0 + scale) + shift
            })
            .collect();
        self.upload_tensor(&Tensor::new(input.shape().to_vec(), output)?)
    }

    fn wan_head_modulate_device(
        &self,
        input: &DeviceTensor,
        timestep: &DeviceTensor,
        modulation: &PreparedVectorHandle,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_tensor(timestep)?;
        self.require_vector(modulation)?;
        let width = input
            .shape()
            .last()
            .copied()
            .context("Wan head modulation input must have at least one dimension")?;
        if width == 0
            || timestep.shape().iter().product::<usize>() != width
            || modulation.length != 2 * width
        {
            bail!("Wan head timestep or modulation shape is invalid");
        }
        let input = input.storage.host()?;
        let timestep = timestep.storage.host()?;
        let modulation = modulation.storage.host()?;
        let output = input
            .data()
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let channel = index % width;
                let timestep = timestep.data()[channel];
                let shift = modulation.data()[channel];
                let scale = modulation.data()[width + channel];
                value * (1.0 + timestep + scale) + timestep + shift
            })
            .collect();
        self.upload_tensor(&Tensor::new(input.shape().to_vec(), output)?)
    }

    fn patchify_device(
        &self,
        input: &DeviceTensor,
        patch: (usize, usize, usize),
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let [channels, time, height, width]: [usize; 4] = input
            .shape()
            .try_into()
            .context("patchify input must be [channels,time,height,width]")?;
        let (patch_time, patch_height, patch_width) = patch;
        if channels == 0
            || patch_time == 0
            || patch_height == 0
            || patch_width == 0
            || time % patch_time != 0
            || height % patch_height != 0
            || width % patch_width != 0
        {
            bail!("patchify input dimensions must be non-zero and divisible by the patch");
        }
        let token_time = time / patch_time;
        let token_height = height / patch_height;
        let token_width = width / patch_width;
        let patch_volume = patch_time * patch_height * patch_width;
        let tokens = token_time * token_height * token_width;
        let feature_width = channels * patch_volume;
        let input = input.storage.host()?;
        let mut output = vec![0.0; tokens * feature_width];
        for token_t in 0..token_time {
            for token_h in 0..token_height {
                for token_w in 0..token_width {
                    let token = (token_t * token_height + token_h) * token_width + token_w;
                    for channel in 0..channels {
                        for patch_t in 0..patch_time {
                            for patch_h in 0..patch_height {
                                for patch_w in 0..patch_width {
                                    let patch_index =
                                        (patch_t * patch_height + patch_h) * patch_width + patch_w;
                                    let source =
                                        ((channel * time + token_t * patch_time + patch_t)
                                            * height
                                            + token_h * patch_height
                                            + patch_h)
                                            * width
                                            + token_w * patch_width
                                            + patch_w;
                                    output[token * feature_width
                                        + channel * patch_volume
                                        + patch_index] = input.data()[source];
                                }
                            }
                        }
                    }
                }
            }
        }
        self.upload_tensor(&Tensor::new(vec![tokens, feature_width], output)?)
    }

    fn unpatchify_device(
        &self,
        input: &DeviceTensor,
        output_channels: usize,
        output: (usize, usize, usize),
        patch: (usize, usize, usize),
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let [tokens, feature_width]: [usize; 2] = input
            .shape()
            .try_into()
            .context("unpatchify input must be [tokens,features]")?;
        let (time, height, width) = output;
        let (patch_time, patch_height, patch_width) = patch;
        if output_channels == 0
            || patch_time == 0
            || patch_height == 0
            || patch_width == 0
            || time % patch_time != 0
            || height % patch_height != 0
            || width % patch_width != 0
        {
            bail!("unpatchify output dimensions must be non-zero and divisible by the patch");
        }
        let patch_volume = patch_time * patch_height * patch_width;
        let token_time = time / patch_time;
        let token_height = height / patch_height;
        let token_width = width / patch_width;
        if tokens != token_time * token_height * token_width
            || feature_width != output_channels * patch_volume
        {
            bail!("unpatchify token shape does not match the requested output");
        }
        let input = input.storage.host()?;
        let mut values = vec![0.0; output_channels * time * height * width];
        for channel in 0..output_channels {
            for output_t in 0..time {
                for output_h in 0..height {
                    for output_w in 0..width {
                        let token = ((output_t / patch_time) * token_height
                            + output_h / patch_height)
                            * token_width
                            + output_w / patch_width;
                        let patch_index = ((output_t % patch_time) * patch_height
                            + output_h % patch_height)
                            * patch_width
                            + output_w % patch_width;
                        let source = token * output_channels * patch_volume
                            + patch_index * output_channels
                            + channel;
                        let destination =
                            ((channel * time + output_t) * height + output_h) * width + output_w;
                        values[destination] = input.data()[source];
                    }
                }
            }
        }
        self.upload_tensor(&Tensor::new(
            vec![output_channels, time, height, width],
            values,
        )?)
    }

    fn multiply_vector_chunk_device(
        &self,
        input: &DeviceTensor,
        vector: &DeviceTensor,
        chunk: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_tensor(vector)?;
        let width = input
            .shape()
            .last()
            .copied()
            .context("vector multiply input must have at least one dimension")?;
        let required_values = (chunk + 1)
            .checked_mul(width)
            .context("vector multiply size overflow")?;
        if width == 0 || vector.shape().iter().product::<usize>() < required_values {
            bail!("vector multiply tensor does not contain the requested chunk");
        }
        let input = input.storage.host()?;
        let vector = vector.storage.host()?;
        let output = input
            .data()
            .iter()
            .enumerate()
            .map(|(index, value)| value * vector.data()[chunk * width + index % width])
            .collect();
        self.upload_tensor(&Tensor::new(input.shape().to_vec(), output)?)
    }

    fn require_tensor(&self, tensor: &DeviceTensor) -> Result<()> {
        if tensor.backend != self.kind() {
            bail!(
                "backend {:?} cannot consume tensor owned by {:?}",
                self.kind(),
                tensor.backend
            );
        }
        Ok(())
    }

    fn require_weights(&self, weights: &LinearWeightHandle) -> Result<()> {
        if weights.backend != self.kind() {
            bail!(
                "backend {:?} cannot consume weights owned by {:?}",
                self.kind(),
                weights.backend
            );
        }
        Ok(())
    }

    fn require_conv3d_weights(&self, weights: &Conv3dWeightHandle) -> Result<()> {
        if weights.backend != self.kind() {
            bail!(
                "backend {:?} cannot consume Conv3D weights owned by {:?}",
                self.kind(),
                weights.backend
            );
        }
        Ok(())
    }

    fn require_vector(&self, vector: &PreparedVectorHandle) -> Result<()> {
        if vector.backend != self.kind() {
            bail!(
                "backend {:?} cannot consume vector owned by {:?}",
                self.kind(),
                vector.backend
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScalarBackend;

pub(crate) static SCALAR_BACKEND: ScalarBackend = ScalarBackend;

impl TensorBackend for ScalarBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::ScalarCpu
    }

    fn name(&self) -> &'static str {
        "scalar-cpu"
    }

    fn add(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        left.add(right)
    }

    fn multiply(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        if left.shape() != right.shape() {
            bail!(
                "multiply shape mismatch: {:?} vs {:?}",
                left.shape(),
                right.shape()
            );
        }
        Tensor::new(
            left.shape().to_vec(),
            left.data()
                .iter()
                .zip(right.data())
                .map(|(left, right)| left * right)
                .collect(),
        )
    }

    fn scale(&self, input: &Tensor, value: f32) -> Result<Tensor> {
        Tensor::new(
            input.shape().to_vec(),
            input.data().iter().map(|input| input * value).collect(),
        )
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone().silu())
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        const SQRT_TWO_OVER_PI: f32 = 0.797_884_56;
        Tensor::new(
            input.shape().to_vec(),
            input
                .data()
                .iter()
                .map(|&input| {
                    0.5 * input
                        * (1.0
                            + (SQRT_TWO_OVER_PI * (input + 0.044715 * input * input * input))
                                .tanh())
                })
                .collect(),
        )
    }

    fn clamp(&self, input: &Tensor, minimum: f32, maximum: f32) -> Result<Tensor> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            bail!("clamp bounds must be finite and ordered");
        }
        Tensor::new(
            input.shape().to_vec(),
            input
                .data()
                .iter()
                .map(|value| value.clamp(minimum, maximum))
                .collect(),
        )
    }

    fn channel_rms_norm_3d(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
        input.channel_rms_norm_3d(weight, epsilon)
    }

    fn linear(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        input.linear(weight, bias)
    }
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VulkanBackend;

#[cfg(feature = "vulkan")]
pub(crate) static VULKAN_BACKEND: VulkanBackend = VulkanBackend;

#[cfg(feature = "vulkan")]
impl TensorBackend for VulkanBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    fn name(&self) -> &'static str {
        "vulkan"
    }

    fn add(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        crate::vulkan::add(left, right)
    }

    fn multiply(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        crate::vulkan::multiply(left, right)
    }

    fn scale(&self, input: &Tensor, value: f32) -> Result<Tensor> {
        crate::vulkan::scale(input, value)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        crate::vulkan::silu(input)
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        crate::vulkan::gelu_tanh(input)
    }

    fn clamp(&self, input: &Tensor, minimum: f32, maximum: f32) -> Result<Tensor> {
        crate::vulkan::clamp(input, minimum, maximum)
    }

    fn channel_rms_norm_3d(&self, input: &Tensor, weight: &Tensor, epsilon: f32) -> Result<Tensor> {
        crate::vulkan::channel_rms_norm_3d(input, weight, epsilon)
    }

    fn linear(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        crate::vulkan::linear_tensor(input, weight, bias)
    }

    fn upload_tensor(&self, input: &Tensor) -> Result<DeviceTensor> {
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape().to_vec(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::upload_resident_tensor(input)?),
        })
    }

    fn download_tensor(&self, input: &DeviceTensor) -> Result<Tensor> {
        self.require_tensor(input)?;
        let DeviceTensorStorage::Vulkan(storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan input storage");
        };
        crate::vulkan::download_resident_tensor(storage, &input.shape)
    }

    fn ncthw_slice_time_device(
        &self,
        input: &DeviceTensor,
        start: usize,
        count: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let input_shape: [usize; 5] = input
            .shape()
            .try_into()
            .context("Vulkan temporal slice input must be NCTHW")?;
        if count == 0 {
            bail!("Vulkan temporal slice cannot select zero frames");
        }
        let end = start
            .checked_add(count)
            .context("Vulkan temporal slice range overflow")?;
        if end > input_shape[2] {
            bail!(
                "Vulkan temporal slice {start}..{end} exceeds input time {}",
                input_shape[2]
            );
        }
        let DeviceTensorStorage::Vulkan(storage) = &input.storage else {
            bail!("Vulkan temporal slice received non-Vulkan storage");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: vec![
                input_shape[0],
                input_shape[1],
                count,
                input_shape[3],
                input_shape[4],
            ],
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_ncthw_slice_time(
                storage,
                input_shape,
                start,
                count,
            )?),
        })
    }

    fn ncthw_concat_time_device(&self, inputs: &[&DeviceTensor]) -> Result<DeviceTensor> {
        let first = inputs
            .first()
            .context("Vulkan temporal concat requires at least one tensor")?;
        self.require_tensor(first)?;
        let mut output = (*first).clone();
        for right in &inputs[1..] {
            self.require_tensor(right)?;
            let left_shape: [usize; 5] = output
                .shape()
                .try_into()
                .context("Vulkan temporal concat input must be NCTHW")?;
            let right_shape: [usize; 5] = right
                .shape()
                .try_into()
                .context("Vulkan temporal concat input must be NCTHW")?;
            if [left_shape[0], left_shape[1], left_shape[3], left_shape[4]]
                != [
                    right_shape[0],
                    right_shape[1],
                    right_shape[3],
                    right_shape[4],
                ]
            {
                bail!(
                    "Vulkan temporal concat non-time dimensions differ: {:?} vs {:?}",
                    output.shape(),
                    right.shape()
                );
            }
            let output_time = left_shape[2]
                .checked_add(right_shape[2])
                .context("Vulkan temporal concat length overflow")?;
            let DeviceTensorStorage::Vulkan(left_storage) = &output.storage else {
                bail!("Vulkan temporal concat received non-Vulkan left storage");
            };
            let DeviceTensorStorage::Vulkan(right_storage) = &right.storage else {
                bail!("Vulkan temporal concat received non-Vulkan right storage");
            };
            let storage = crate::vulkan::resident_ncthw_concat_time(
                left_storage,
                right_storage,
                left_shape,
                right_shape[2],
            )?;
            output = DeviceTensor {
                backend: BackendKind::Vulkan,
                shape: vec![
                    left_shape[0],
                    left_shape[1],
                    output_time,
                    left_shape[3],
                    left_shape[4],
                ],
                storage: DeviceTensorStorage::Vulkan(storage),
            };
        }
        Ok(output)
    }

    fn ncthw_prepend_zero_time_device(
        &self,
        input: &DeviceTensor,
        count: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        if count == 0 {
            return Ok(input.clone());
        }
        let input_shape: [usize; 5] = input
            .shape()
            .try_into()
            .context("Vulkan zero-time prepend input must be NCTHW")?;
        let output_time = input_shape[2]
            .checked_add(count)
            .context("Vulkan zero-time prepend length overflow")?;
        let DeviceTensorStorage::Vulkan(storage) = &input.storage else {
            bail!("Vulkan zero-time prepend received non-Vulkan storage");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: vec![
                input_shape[0],
                input_shape[1],
                output_time,
                input_shape[3],
                input_shape[4],
            ],
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_ncthw_prepend_zero_time(
                storage,
                input_shape,
                count,
            )?),
        })
    }

    fn ncthw_channels_to_time_device(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let input_shape: [usize; 5] = input
            .shape()
            .try_into()
            .context("Vulkan channel-to-time input must be NCTHW")?;
        if input_shape[1] % 2 != 0 {
            bail!("Vulkan Wan channel-to-time shuffle needs an even channel count");
        }
        let output_time = input_shape[2]
            .checked_mul(2)
            .context("Vulkan channel-to-time output length overflow")?;
        let DeviceTensorStorage::Vulkan(storage) = &input.storage else {
            bail!("Vulkan channel-to-time received non-Vulkan storage");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: vec![
                input_shape[0],
                input_shape[1] / 2,
                output_time,
                input_shape[3],
                input_shape[4],
            ],
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_ncthw_channels_to_time(
                storage,
                input_shape,
            )?),
        })
    }

    fn prepare_linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<LinearWeightHandle> {
        let [output_width, input_width]: [usize; 2] = weight
            .shape()
            .try_into()
            .context("Vulkan linear weight must be rank two")?;
        if let Some(bias) = bias
            && bias.shape() != [output_width]
        {
            bail!(
                "Vulkan linear bias shape {:?} must be [{output_width}]",
                bias.shape()
            );
        }
        let storage = crate::vulkan::prepare_resident_linear(weight, bias)?;
        Ok(LinearWeightHandle {
            backend: BackendKind::Vulkan,
            input_width,
            output_width,
            storage: LinearWeightStorage::Vulkan(storage),
        })
    }

    fn linear_prepared(
        &self,
        input: &DeviceTensor,
        weights: &LinearWeightHandle,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_weights(weights)?;
        if input.shape().last().copied() != Some(weights.input_width) {
            bail!(
                "prepared Vulkan linear input shape {:?} does not end in {}",
                input.shape(),
                weights.input_width
            );
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan input storage");
        };
        let LinearWeightStorage::Vulkan(weight_storage) = &weights.storage else {
            bail!("Vulkan backend received non-Vulkan weight storage");
        };
        let storage = crate::vulkan::resident_linear(input_storage, weight_storage)?;
        let mut shape = input.shape.clone();
        *shape
            .last_mut()
            .context("prepared Vulkan linear input has no dimensions")? = weights.output_width;
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape,
            storage: DeviceTensorStorage::Vulkan(storage),
        })
    }

    fn prepare_conv3d(&self, weight: &Tensor, bias: Option<&Tensor>) -> Result<Conv3dWeightHandle> {
        let [
            output_channels,
            input_channels,
            kernel_time,
            kernel_height,
            kernel_width,
        ]: [usize; 5] = weight
            .shape()
            .try_into()
            .context("Vulkan Conv3D weight must be [out,in,time,height,width]")?;
        if output_channels == 0
            || input_channels == 0
            || kernel_time == 0
            || kernel_height == 0
            || kernel_width == 0
        {
            bail!("Vulkan Conv3D weight dimensions must be non-zero");
        }
        if let Some(bias) = bias
            && bias.shape() != [output_channels]
        {
            bail!(
                "Vulkan Conv3D bias shape {:?} must be [{output_channels}]",
                bias.shape()
            );
        }
        Ok(Conv3dWeightHandle {
            backend: BackendKind::Vulkan,
            input_channels,
            output_channels,
            kernel: [kernel_time, kernel_height, kernel_width],
            storage: Conv3dWeightStorage::Vulkan(crate::vulkan::prepare_resident_conv3d(
                weight, bias,
            )?),
        })
    }

    fn conv3d_prepared_device(
        &self,
        input: &DeviceTensor,
        weights: &Conv3dWeightHandle,
        padding_before: [usize; 3],
        padding_after: [usize; 3],
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_conv3d_weights(weights)?;
        let [batch, input_channels, input_time, input_height, input_width]: [usize; 5] = input
            .shape()
            .try_into()
            .context("Vulkan Conv3D input must be [batch,channels,time,height,width]")?;
        if batch == 0 || input_channels != weights.input_channels {
            bail!("Vulkan Conv3D input batch/channels do not match prepared weights");
        }
        let input_axes = [input_time, input_height, input_width];
        let mut output_axes = [0; 3];
        for axis in 0..3 {
            let padded = input_axes[axis]
                .checked_add(padding_before[axis])
                .and_then(|value| value.checked_add(padding_after[axis]))
                .context("Vulkan Conv3D padded dimension overflow")?;
            if padded < weights.kernel[axis] {
                bail!("Vulkan Conv3D kernel is larger than the padded input");
            }
            output_axes[axis] = padded - weights.kernel[axis] + 1;
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan Conv3D input");
        };
        let Conv3dWeightStorage::Vulkan(weight_storage) = &weights.storage else {
            bail!("Vulkan backend received non-Vulkan Conv3D weights");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: vec![
                batch,
                weights.output_channels,
                output_axes[0],
                output_axes[1],
                output_axes[2],
            ],
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_conv3d(
                input_storage,
                weight_storage,
                [batch, input_channels, input_time, input_height, input_width],
                padding_before,
                padding_after,
            )?),
        })
    }

    fn silu_device(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan input storage");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_silu(input_storage)?),
        })
    }

    fn gelu_tanh_device(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan GELU input");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_gelu_tanh(input_storage)?),
        })
    }

    fn add_device(&self, left: &DeviceTensor, right: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(left)?;
        self.require_tensor(right)?;
        if left.shape() != right.shape() {
            bail!(
                "resident Vulkan add shape mismatch: {:?} vs {:?}",
                left.shape(),
                right.shape()
            );
        }
        let DeviceTensorStorage::Vulkan(left_storage) = &left.storage else {
            bail!("Vulkan backend received non-Vulkan add input");
        };
        let DeviceTensorStorage::Vulkan(right_storage) = &right.storage else {
            bail!("Vulkan backend received non-Vulkan add input");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: left.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_add(
                left_storage,
                right_storage,
            )?),
        })
    }

    fn scale_device(&self, input: &DeviceTensor, value: f32) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        if !value.is_finite() {
            bail!("resident Vulkan scale must be finite");
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan scale input");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_scale(
                input_storage,
                value,
            )?),
        })
    }

    fn prepare_vector(&self, vector: &Tensor) -> Result<PreparedVectorHandle> {
        let [length]: [usize; 1] = vector
            .shape()
            .try_into()
            .context("prepared Vulkan vector must be rank one")?;
        if length == 0 {
            bail!("prepared Vulkan vector cannot be empty");
        }
        Ok(PreparedVectorHandle {
            backend: BackendKind::Vulkan,
            length,
            storage: PreparedVectorStorage::Vulkan(crate::vulkan::prepare_resident_vector(vector)?),
        })
    }

    fn add_vector_device(
        &self,
        input: &DeviceTensor,
        vector: &PreparedVectorHandle,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_vector(vector)?;
        if input.shape.iter().product::<usize>() != vector.length {
            bail!(
                "prepared Vulkan vector length {} does not match tensor shape {:?}",
                vector.length,
                input.shape()
            );
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan input storage");
        };
        let PreparedVectorStorage::Vulkan(vector_storage) = &vector.storage else {
            bail!("Vulkan backend received non-Vulkan vector storage");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_add_vector(
                input_storage,
                vector_storage,
            )?),
        })
    }

    fn layer_norm_device(
        &self,
        input: &DeviceTensor,
        weight: Option<&PreparedVectorHandle>,
        bias: Option<&PreparedVectorHandle>,
        epsilon: f32,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("Vulkan LayerNorm epsilon must be finite and positive");
        }
        if weight.is_some() != bias.is_some() {
            bail!("affine Vulkan LayerNorm requires both weight and bias");
        }
        let width = input
            .shape()
            .last()
            .copied()
            .context("Vulkan LayerNorm input must have at least one dimension")?;
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan input storage");
        };
        let affine = match (weight, bias) {
            (Some(weight), Some(bias)) => {
                self.require_vector(weight)?;
                self.require_vector(bias)?;
                if weight.length != width || bias.length != width {
                    bail!(
                        "Vulkan LayerNorm weight/bias lengths {}/{} do not match width {width}",
                        weight.length,
                        bias.length
                    );
                }
                let PreparedVectorStorage::Vulkan(weight) = &weight.storage else {
                    bail!("Vulkan backend received non-Vulkan LayerNorm weight");
                };
                let PreparedVectorStorage::Vulkan(bias) = &bias.storage else {
                    bail!("Vulkan backend received non-Vulkan LayerNorm bias");
                };
                Some((weight, bias))
            }
            (None, None) => None,
            _ => unreachable!(),
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_layer_norm(
                input_storage,
                affine,
                width,
                epsilon,
            )?),
        })
    }

    fn wan_modulate_device(
        &self,
        input: &DeviceTensor,
        modulation: &DeviceTensor,
        shift_chunk: usize,
        scale_chunk: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_tensor(modulation)?;
        let width = input
            .shape()
            .last()
            .copied()
            .context("Vulkan Wan modulation input must have at least one dimension")?;
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan modulation input");
        };
        let DeviceTensorStorage::Vulkan(modulation_storage) = &modulation.storage else {
            bail!("Vulkan backend received non-Vulkan modulation vector");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_wan_modulate(
                input_storage,
                modulation_storage,
                width,
                shift_chunk,
                scale_chunk,
            )?),
        })
    }

    fn wan_head_modulate_device(
        &self,
        input: &DeviceTensor,
        timestep: &DeviceTensor,
        modulation: &PreparedVectorHandle,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_tensor(timestep)?;
        self.require_vector(modulation)?;
        let width = input
            .shape()
            .last()
            .copied()
            .context("Vulkan Wan head input must have at least one dimension")?;
        if width == 0
            || timestep.shape().iter().product::<usize>() != width
            || modulation.length != 2 * width
        {
            bail!("Vulkan Wan head timestep or modulation shape is invalid");
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan Wan head input");
        };
        let DeviceTensorStorage::Vulkan(timestep_storage) = &timestep.storage else {
            bail!("Vulkan backend received non-Vulkan Wan head timestep");
        };
        let PreparedVectorStorage::Vulkan(modulation_storage) = &modulation.storage else {
            bail!("Vulkan backend received non-Vulkan Wan head modulation");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_wan_head_modulate(
                input_storage,
                timestep_storage,
                modulation_storage,
                width,
            )?),
        })
    }

    fn patchify_device(
        &self,
        input: &DeviceTensor,
        patch: (usize, usize, usize),
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let [channels, time, height, width]: [usize; 4] = input
            .shape()
            .try_into()
            .context("Vulkan patchify input must be [channels,time,height,width]")?;
        let (patch_time, patch_height, patch_width) = patch;
        if channels == 0
            || patch_time == 0
            || patch_height == 0
            || patch_width == 0
            || time % patch_time != 0
            || height % patch_height != 0
            || width % patch_width != 0
        {
            bail!("Vulkan patchify dimensions must be non-zero and divisible by the patch");
        }
        let tokens = (time / patch_time) * (height / patch_height) * (width / patch_width);
        let feature_width = channels * patch_time * patch_height * patch_width;
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan patchify input");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: vec![tokens, feature_width],
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_patchify(
                input_storage,
                channels,
                time,
                height,
                width,
                patch,
            )?),
        })
    }

    fn unpatchify_device(
        &self,
        input: &DeviceTensor,
        output_channels: usize,
        output: (usize, usize, usize),
        patch: (usize, usize, usize),
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let [tokens, feature_width]: [usize; 2] = input
            .shape()
            .try_into()
            .context("Vulkan unpatchify input must be [tokens,features]")?;
        let (time, height, width) = output;
        let (patch_time, patch_height, patch_width) = patch;
        if output_channels == 0
            || patch_time == 0
            || patch_height == 0
            || patch_width == 0
            || time % patch_time != 0
            || height % patch_height != 0
            || width % patch_width != 0
        {
            bail!("Vulkan unpatchify dimensions must be non-zero and divisible by the patch");
        }
        let patch_volume = patch_time * patch_height * patch_width;
        let expected_tokens = (time / patch_time) * (height / patch_height) * (width / patch_width);
        if tokens != expected_tokens || feature_width != output_channels * patch_volume {
            bail!("Vulkan unpatchify input shape does not match the requested output");
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan unpatchify input");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: vec![output_channels, time, height, width],
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_unpatchify(
                input_storage,
                output_channels,
                time,
                height,
                width,
                patch,
            )?),
        })
    }

    fn rms_norm_device(
        &self,
        input: &DeviceTensor,
        weight: &PreparedVectorHandle,
        epsilon: f32,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_vector(weight)?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("Vulkan RMSNorm epsilon must be finite and positive");
        }
        let width = input
            .shape()
            .last()
            .copied()
            .context("Vulkan RMSNorm input must have at least one dimension")?;
        if width == 0 || weight.length != width {
            bail!(
                "Vulkan RMSNorm weight length {} does not match input width {width}",
                weight.length
            );
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan RMSNorm input");
        };
        let PreparedVectorStorage::Vulkan(weight_storage) = &weight.storage else {
            bail!("Vulkan backend received non-Vulkan RMSNorm weight");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_rms_norm(
                input_storage,
                weight_storage,
                width,
                epsilon,
            )?),
        })
    }

    fn channel_rms_norm_3d_device(
        &self,
        input: &DeviceTensor,
        weight: &PreparedVectorHandle,
        epsilon: f32,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_vector(weight)?;
        let [batch, channels, time, height, width]: [usize; 5] = input
            .shape()
            .try_into()
            .context("Vulkan channel RMSNorm input must be NCTHW")?;
        if channels == 0 || weight.length != channels {
            bail!(
                "Vulkan channel RMSNorm weight length {} does not match {channels} channels",
                weight.length
            );
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("Vulkan channel RMSNorm epsilon must be finite and positive");
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan channel RMSNorm received non-Vulkan input storage");
        };
        let PreparedVectorStorage::Vulkan(weight_storage) = &weight.storage else {
            bail!("Vulkan channel RMSNorm received non-Vulkan weight storage");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_channel_rms_norm_3d(
                input_storage,
                weight_storage,
                [batch, channels, time, height, width],
                epsilon,
            )?),
        })
    }

    fn rope_device(
        &self,
        input: &DeviceTensor,
        positions: &DeviceTensor,
        heads: usize,
        head_dim: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_tensor(positions)?;
        if heads == 0 || head_dim == 0 || head_dim % 2 != 0 {
            bail!("Vulkan RoPE heads and even head dimension must be non-zero");
        }
        let width = heads
            .checked_mul(head_dim)
            .context("Vulkan RoPE width overflow")?;
        if input.shape().last().copied() != Some(width) {
            bail!(
                "Vulkan RoPE input shape {:?} does not end in width {width}",
                input.shape()
            );
        }
        let rows = input.shape().iter().product::<usize>() / width;
        let expected_positions = rows
            .checked_mul(head_dim / 2)
            .and_then(|values| values.checked_mul(crate::wan_rope::PAIR_STRIDE))
            .context("Vulkan RoPE position size overflow")?;
        if positions.shape().iter().product::<usize>() != expected_positions {
            bail!(
                "Vulkan RoPE position shape {:?} has the wrong size; expected {expected_positions} values",
                positions.shape()
            );
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan RoPE input");
        };
        let DeviceTensorStorage::Vulkan(position_storage) = &positions.storage else {
            bail!("Vulkan backend received non-Vulkan RoPE positions");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_rope(
                input_storage,
                position_storage,
                rows,
                heads,
                head_dim,
            )?),
        })
    }

    fn attention_scores_device(
        &self,
        query: &DeviceTensor,
        key: &DeviceTensor,
        heads: usize,
        head_dim: usize,
        scale: f32,
    ) -> Result<DeviceTensor> {
        self.require_tensor(query)?;
        self.require_tensor(key)?;
        let width = heads
            .checked_mul(head_dim)
            .context("Vulkan attention score width overflow")?;
        let [queries, query_width]: [usize; 2] = query
            .shape()
            .try_into()
            .context("Vulkan attention query must be rank two")?;
        let [keys, key_width]: [usize; 2] = key
            .shape()
            .try_into()
            .context("Vulkan attention key must be rank two")?;
        if heads == 0
            || head_dim == 0
            || !scale.is_finite()
            || queries == 0
            || keys == 0
            || query_width != width
            || key_width != width
        {
            bail!("Vulkan attention Q/K shapes do not match the head layout");
        }
        let DeviceTensorStorage::Vulkan(query_storage) = &query.storage else {
            bail!("Vulkan backend received non-Vulkan attention query");
        };
        let DeviceTensorStorage::Vulkan(key_storage) = &key.storage else {
            bail!("Vulkan backend received non-Vulkan attention key");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: vec![heads, queries, keys],
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_attention_scores(
                query_storage,
                key_storage,
                queries,
                keys,
                heads,
                head_dim,
                scale,
            )?),
        })
    }

    fn softmax_device(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        let width = input
            .shape()
            .last()
            .copied()
            .context("Vulkan softmax input must have at least one dimension")?;
        if width == 0 {
            bail!("Vulkan softmax width cannot be zero");
        }
        let rows = input.shape().iter().product::<usize>() / width;
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan softmax input");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_softmax(
                input_storage,
                rows,
                width,
            )?),
        })
    }

    fn attention_values_device(
        &self,
        probabilities: &DeviceTensor,
        value: &DeviceTensor,
        heads: usize,
        head_dim: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(probabilities)?;
        self.require_tensor(value)?;
        let [probability_heads, queries, keys]: [usize; 3] = probabilities
            .shape()
            .try_into()
            .context("Vulkan attention probabilities must be rank three")?;
        let [value_rows, value_width]: [usize; 2] = value
            .shape()
            .try_into()
            .context("Vulkan attention value must be rank two")?;
        let width = heads
            .checked_mul(head_dim)
            .context("Vulkan attention value width overflow")?;
        if heads == 0
            || head_dim == 0
            || probability_heads != heads
            || queries == 0
            || keys == 0
            || value_rows != keys
            || value_width != width
        {
            bail!("Vulkan attention probability/value shapes do not match the head layout");
        }
        let DeviceTensorStorage::Vulkan(probability_storage) = &probabilities.storage else {
            bail!("Vulkan backend received non-Vulkan attention probabilities");
        };
        let DeviceTensorStorage::Vulkan(value_storage) = &value.storage else {
            bail!("Vulkan backend received non-Vulkan attention values");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: vec![queries, width],
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_attention_values(
                probability_storage,
                value_storage,
                queries,
                keys,
                heads,
                head_dim,
            )?),
        })
    }

    fn multiply_vector_chunk_device(
        &self,
        input: &DeviceTensor,
        vector: &DeviceTensor,
        chunk: usize,
    ) -> Result<DeviceTensor> {
        self.require_tensor(input)?;
        self.require_tensor(vector)?;
        let width = input
            .shape()
            .last()
            .copied()
            .context("Vulkan vector multiply input must have at least one dimension")?;
        let required_values = (chunk + 1)
            .checked_mul(width)
            .context("Vulkan vector multiply size overflow")?;
        if width == 0 || vector.shape().iter().product::<usize>() < required_values {
            bail!("Vulkan vector multiply tensor does not contain the requested chunk");
        }
        let DeviceTensorStorage::Vulkan(input_storage) = &input.storage else {
            bail!("Vulkan backend received non-Vulkan vector multiply input");
        };
        let DeviceTensorStorage::Vulkan(vector_storage) = &vector.storage else {
            bail!("Vulkan backend received non-Vulkan vector multiply tensor");
        };
        Ok(DeviceTensor {
            backend: BackendKind::Vulkan,
            shape: input.shape.clone(),
            storage: DeviceTensorStorage::Vulkan(crate::vulkan::resident_multiply_vector_chunk(
                input_storage,
                vector_storage,
                width,
                chunk,
            )?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_add_preserves_shape_and_values() {
        let left = Tensor::new(vec![1, 2, 3], vec![1.0, -2.0, 3.5, 4.0, 0.0, -7.0]).unwrap();
        let right = Tensor::new(vec![1, 2, 3], vec![0.5, 2.0, -1.5, 4.0, -3.0, 10.0]).unwrap();
        let output = SCALAR_BACKEND.add(&left, &right).unwrap();
        assert_eq!(output.shape(), &[1, 2, 3]);
        assert_eq!(output.data(), &[1.5, 0.0, 2.0, 8.0, -3.0, 3.0]);
        assert_eq!(SCALAR_BACKEND.kind(), BackendKind::ScalarCpu);
        assert_eq!(SCALAR_BACKEND.name(), "scalar-cpu");
    }

    #[test]
    fn scalar_add_rejects_shape_mismatch() {
        let left = Tensor::zeros(vec![2, 3]).unwrap();
        let right = Tensor::zeros(vec![3, 2]).unwrap();
        assert!(SCALAR_BACKEND.add(&left, &right).is_err());
    }

    #[test]
    fn scalar_elementwise_and_channel_rmsnorm_cover_the_backend_surface() {
        let input = Tensor::new(vec![1, 1, 1, 1, 5], vec![-2.0, -0.5, 0.0, 1.0, 3.0]).unwrap();
        let other = Tensor::new(vec![1, 1, 1, 1, 5], vec![2.0, -4.0, 5.0, 0.5, -1.0]).unwrap();
        assert_eq!(
            SCALAR_BACKEND.multiply(&input, &other).unwrap().data(),
            &[-4.0, 2.0, 0.0, 0.5, -3.0]
        );
        assert_eq!(
            SCALAR_BACKEND.scale(&input, 0.25).unwrap().data(),
            &[-0.5, -0.125, 0.0, 0.25, 0.75]
        );
        assert_eq!(
            SCALAR_BACKEND.clamp(&input, -1.0, 1.0).unwrap().data(),
            &[-1.0, -0.5, 0.0, 1.0, 1.0]
        );
        assert!(SCALAR_BACKEND.silu(&input).unwrap().data()[3] > 0.73);
        assert!(SCALAR_BACKEND.gelu_tanh(&input).unwrap().data()[3] > 0.84);

        let weight = Tensor::new(vec![1], vec![1.5]).unwrap();
        let normalized = SCALAR_BACKEND
            .channel_rms_norm_3d(&input, &weight, 1e-12)
            .unwrap();
        for (&actual, &source) in normalized.data().iter().zip(input.data()) {
            let expected = if source == 0.0 {
                0.0
            } else {
                source.signum() * 1.5
            };
            assert!((actual - expected).abs() < 1e-6);
        }

        let linear_input =
            Tensor::new(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, 0.0]).unwrap();
        let linear_weight = Tensor::new(
            vec![3, 4],
            vec![
                1.0, 0.0, -1.0, 0.5, 0.25, 0.5, 0.75, 1.0, -2.0, 1.0, 0.0, 0.5,
            ],
        )
        .unwrap();
        let linear_bias = Tensor::new(vec![3], vec![0.5, -1.0, 2.0]).unwrap();
        assert_eq!(
            SCALAR_BACKEND
                .linear(&linear_input, &linear_weight, Some(&linear_bias))
                .unwrap()
                .shape(),
            &[2, 3]
        );

        let prepared = SCALAR_BACKEND
            .prepare_linear(&linear_weight, Some(&linear_bias))
            .unwrap();
        let device_input = SCALAR_BACKEND.upload_tensor(&linear_input).unwrap();
        assert_eq!(device_input.shape(), &[2, 4]);
        let device_output = SCALAR_BACKEND
            .linear_prepared(&device_input, &prepared)
            .unwrap();
        let device_output = SCALAR_BACKEND.silu_device(&device_output).unwrap();
        let device_output = SCALAR_BACKEND.download_tensor(&device_output).unwrap();
        assert_eq!(device_output.shape(), &[2, 3]);
        assert!(device_output.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn scalar_ncthw_temporal_primitives_preserve_exact_layout() {
        let input = Tensor::new(
            vec![2, 3, 5, 2, 3],
            (0..180).map(|index| index as f32 - 90.0).collect(),
        )
        .unwrap();
        let device = SCALAR_BACKEND.upload_tensor(&input).unwrap();
        for (start, count) in [(0, 1), (4, 1), (1, 3), (0, 5)] {
            let output = SCALAR_BACKEND
                .ncthw_slice_time_device(&device, start, count)
                .unwrap();
            assert_eq!(output.shape(), &[2, 3, count, 2, 3]);
        }
        assert!(
            SCALAR_BACKEND
                .ncthw_slice_time_device(&device, 0, 0)
                .is_err()
        );
        assert!(
            SCALAR_BACKEND
                .ncthw_slice_time_device(&device, 5, 1)
                .is_err()
        );

        let first = SCALAR_BACKEND
            .ncthw_slice_time_device(&device, 0, 1)
            .unwrap();
        let middle = SCALAR_BACKEND
            .ncthw_slice_time_device(&device, 1, 2)
            .unwrap();
        let last = SCALAR_BACKEND
            .ncthw_slice_time_device(&device, 3, 2)
            .unwrap();
        let concatenated = SCALAR_BACKEND
            .ncthw_concat_time_device(&[&first, &middle, &last])
            .unwrap();
        assert_eq!(
            SCALAR_BACKEND.download_tensor(&concatenated).unwrap(),
            input
        );

        let prepended = SCALAR_BACKEND
            .ncthw_prepend_zero_time_device(&middle, 2)
            .unwrap();
        let prepended = SCALAR_BACKEND.download_tensor(&prepended).unwrap();
        assert_eq!(prepended.shape(), &[2, 3, 4, 2, 3]);
        let plane = 2 * 3;
        for sample in 0..2 {
            for channel in 0..3 {
                let base = (sample * 3 + channel) * 4 * plane;
                assert_eq!(&prepended.data()[base..base + 2 * plane], &[0.0; 12]);
                let source = (sample * 3 + channel) * 2 * plane;
                let middle = SCALAR_BACKEND.download_tensor(&middle).unwrap();
                assert_eq!(
                    &prepended.data()[base + 2 * plane..base + 4 * plane],
                    &middle.data()[source..source + 2 * plane]
                );
            }
        }

        let shuffle_input = Tensor::new(
            vec![2, 4, 2, 2, 3],
            (0..96).map(|index| index as f32 + 0.25).collect(),
        )
        .unwrap();
        let shuffle_device = SCALAR_BACKEND.upload_tensor(&shuffle_input).unwrap();
        let shuffled = SCALAR_BACKEND
            .ncthw_channels_to_time_device(&shuffle_device)
            .unwrap();
        let shuffled = SCALAR_BACKEND.download_tensor(&shuffled).unwrap();
        assert_eq!(shuffled.shape(), &[2, 2, 4, 2, 3]);
        for sample in 0..2 {
            for channel in 0..2 {
                for input_time in 0..2 {
                    for channel_half in 0..2 {
                        for spatial in 0..6 {
                            let source =
                                (((sample * 4 + channel_half * 2 + channel) * 2 + input_time) * 6)
                                    + spatial;
                            let destination =
                                (((sample * 2 + channel) * 4 + input_time * 2 + channel_half) * 6)
                                    + spatial;
                            assert_eq!(shuffled.data()[destination], shuffle_input.data()[source]);
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_resident_ncthw_temporal_primitives_match_scalar_exactly() {
        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident NCTHW temporal parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident NCTHW temporal parity failed: {error:#}"),
        };

        let input = Tensor::new(
            vec![2, 3, 5, 2, 3],
            (0..180).map(|index| index as f32 - 90.0).collect(),
        )
        .unwrap();
        let scalar_input = SCALAR_BACKEND.upload_tensor(&input).unwrap();
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let mut resident_outputs = Vec::new();
        for (start, count) in [(0, 1), (4, 1), (1, 3), (0, 5)] {
            let expected = SCALAR_BACKEND
                .ncthw_slice_time_device(&scalar_input, start, count)
                .and_then(|tensor| SCALAR_BACKEND.download_tensor(&tensor))
                .unwrap();
            let output = VULKAN_BACKEND
                .ncthw_slice_time_device(&device_input, start, count)
                .unwrap();
            assert!(output.remains_resident());
            assert!(!output.is_device_local().unwrap());
            assert_eq!(VULKAN_BACKEND.download_tensor(&output).unwrap(), expected);
            resident_outputs.push(output);
        }
        assert!(
            VULKAN_BACKEND
                .ncthw_slice_time_device(&device_input, 0, 0)
                .is_err()
        );
        assert!(
            VULKAN_BACKEND
                .ncthw_slice_time_device(&device_input, 5, 1)
                .is_err()
        );

        let first = VULKAN_BACKEND
            .ncthw_slice_time_device(&device_input, 0, 1)
            .unwrap();
        let middle = VULKAN_BACKEND
            .ncthw_slice_time_device(&device_input, 1, 2)
            .unwrap();
        let last = VULKAN_BACKEND
            .ncthw_slice_time_device(&device_input, 3, 2)
            .unwrap();
        let concatenated = VULKAN_BACKEND
            .ncthw_concat_time_device(&[&first, &middle, &last])
            .unwrap();
        assert_eq!(
            VULKAN_BACKEND.download_tensor(&concatenated).unwrap(),
            input
        );

        let expected_prepend = SCALAR_BACKEND
            .ncthw_slice_time_device(&scalar_input, 1, 2)
            .and_then(|tensor| SCALAR_BACKEND.ncthw_prepend_zero_time_device(&tensor, 2))
            .and_then(|tensor| SCALAR_BACKEND.download_tensor(&tensor))
            .unwrap();
        let prepended = VULKAN_BACKEND
            .ncthw_prepend_zero_time_device(&middle, 2)
            .unwrap();
        let prepended_host = VULKAN_BACKEND.download_tensor(&prepended).unwrap();
        assert_eq!(prepended_host, expected_prepend);
        for channel_plane in prepended_host.data().chunks_exact(4 * 2 * 3) {
            assert_eq!(&channel_plane[..2 * 2 * 3], &[0.0; 12]);
        }

        let shuffle_input = Tensor::new(
            vec![2, 4, 2, 2, 3],
            (0..96).map(|index| index as f32 + 0.25).collect(),
        )
        .unwrap();
        let scalar_shuffle = SCALAR_BACKEND.upload_tensor(&shuffle_input).unwrap();
        let expected_shuffle = SCALAR_BACKEND
            .ncthw_channels_to_time_device(&scalar_shuffle)
            .and_then(|tensor| SCALAR_BACKEND.download_tensor(&tensor))
            .unwrap();
        let device_shuffle = VULKAN_BACKEND.upload_tensor(&shuffle_input).unwrap();
        let shuffled = VULKAN_BACKEND
            .ncthw_channels_to_time_device(&device_shuffle)
            .unwrap();
        let shuffled_host = VULKAN_BACKEND.download_tensor(&shuffled).unwrap();
        assert_eq!(shuffled_host, expected_shuffle);
        assert_eq!(shuffled_host.shape(), &[2, 2, 4, 2, 3]);

        let mismatched = Tensor::zeros(vec![1, 3, 1, 2, 3]).unwrap();
        let mismatched = VULKAN_BACKEND.upload_tensor(&mismatched).unwrap();
        assert!(
            VULKAN_BACKEND
                .ncthw_concat_time_device(&[&first, &mismatched])
                .is_err()
        );

        let after = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            3
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 7);
        println!(
            "resident NCTHW temporal primitives: slice_input={:?} slice_cases={:?} concat_inputs=3 prepend=2 shuffle_input={:?} shuffle_output={:?} host_uploads={} downloads={} cached_device_local={}",
            input.shape(),
            [(0, 1), (4, 1), (1, 3), (0, 5)],
            shuffle_input.shape(),
            shuffled_host.shape(),
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_downloads - before.resident_downloads,
            shuffled.is_device_local().unwrap(),
        );

        drop(mismatched);
        drop(shuffled);
        drop(device_shuffle);
        drop(prepended);
        drop(concatenated);
        drop(last);
        drop(middle);
        drop(first);
        drop(resident_outputs);
        drop(device_input);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_add_matches_scalar_on_an_awkward_shape() {
        use crate::parity::{ParityTolerance, compare_backends};

        let left = Tensor::new(
            vec![2, 3, 5],
            (0..30).map(|index| index as f32 * 0.125 - 1.5).collect(),
        )
        .unwrap();
        let right = Tensor::new(
            vec![2, 3, 5],
            (0..30)
                .map(|index| ((index * 7) % 13) as f32 * -0.0625)
                .collect(),
        )
        .unwrap();
        let parity = compare_backends(&SCALAR_BACKEND, &VULKAN_BACKEND, |backend| {
            backend.add(&left, &right)
        });
        let parity = match parity {
            Ok(parity) => parity,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping Vulkan add parity: {error:#}");
                return;
            }
            Err(error) => panic!("required Vulkan add parity failed: {error:#}"),
        };
        eprintln!(
            "backend={} shape={:?} reference_us={} candidate_us={} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            parity.candidate_backend,
            parity.metrics.shape,
            parity.reference_runtime.as_micros(),
            parity.candidate_runtime.as_micros(),
            parity.metrics.cosine_similarity,
            parity.metrics.maximum_absolute_error,
            parity.metrics.mean_absolute_error,
        );
        parity
            .metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 1.0,
                maximum_absolute_error: 0.0,
                maximum_mean_absolute_error: 0.0,
            })
            .unwrap();
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_elementwise_and_channel_rmsnorm_match_scalar() {
        use crate::parity::{BackendParity, ParityTolerance, compare_backends};

        fn required(parity: Result<BackendParity>) -> Option<BackendParity> {
            match parity {
                Ok(parity) => Some(parity),
                Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                    eprintln!("skipping Vulkan backend parity: {error:#}");
                    None
                }
                Err(error) => panic!("required Vulkan backend parity failed: {error:#}"),
            }
        }

        fn verify(label: &str, parity: BackendParity, maximum_error: f32) {
            eprintln!(
                "{label}: shape={:?} scalar_us={} vulkan_us={} cosine={:.9} max_abs={:.9} mean_abs={:.9} nan={}/{} inf={}/{}",
                parity.metrics.shape,
                parity.reference_runtime.as_micros(),
                parity.candidate_runtime.as_micros(),
                parity.metrics.cosine_similarity,
                parity.metrics.maximum_absolute_error,
                parity.metrics.mean_absolute_error,
                parity.metrics.actual_nan_count,
                parity.metrics.expected_nan_count,
                parity.metrics.actual_infinity_count,
                parity.metrics.expected_infinity_count,
            );
            parity
                .metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 0.999_999,
                    maximum_absolute_error: maximum_error,
                    maximum_mean_absolute_error: maximum_error as f64,
                })
                .unwrap();
        }

        let input = Tensor::new(
            vec![2, 3, 5],
            (0..30)
                .map(|index| ((index * 11) % 37) as f32 * 0.125 - 2.0)
                .collect(),
        )
        .unwrap();
        let other = Tensor::new(
            vec![2, 3, 5],
            (0..30)
                .map(|index| ((index * 7) % 23) as f32 * -0.0625 + 0.25)
                .collect(),
        )
        .unwrap();

        let Some(parity) = required(compare_backends(
            &SCALAR_BACKEND,
            &VULKAN_BACKEND,
            |backend| backend.multiply(&input, &other),
        )) else {
            return;
        };
        verify("multiply", parity, 0.0);
        verify(
            "scale",
            required(compare_backends(
                &SCALAR_BACKEND,
                &VULKAN_BACKEND,
                |backend| backend.scale(&input, -0.375),
            ))
            .unwrap(),
            0.0,
        );
        verify(
            "silu",
            required(compare_backends(
                &SCALAR_BACKEND,
                &VULKAN_BACKEND,
                |backend| backend.silu(&input),
            ))
            .unwrap(),
            2e-6,
        );
        verify(
            "gelu_tanh",
            required(compare_backends(
                &SCALAR_BACKEND,
                &VULKAN_BACKEND,
                |backend| backend.gelu_tanh(&input),
            ))
            .unwrap(),
            2e-6,
        );
        verify(
            "clamp",
            required(compare_backends(
                &SCALAR_BACKEND,
                &VULKAN_BACKEND,
                |backend| backend.clamp(&input, -0.75, 1.25),
            ))
            .unwrap(),
            0.0,
        );

        let rms_input = Tensor::new(
            vec![1, 7, 3, 2, 5],
            (0..210)
                .map(|index| ((index * 13) % 41) as f32 * 0.03125 - 0.625)
                .collect(),
        )
        .unwrap();
        let weight = Tensor::new(vec![7], vec![0.75, 1.0, 1.25, -0.5, 0.125, 2.0, -1.5]).unwrap();
        verify(
            "channel_rmsnorm",
            required(compare_backends(
                &SCALAR_BACKEND,
                &VULKAN_BACKEND,
                |backend| backend.channel_rms_norm_3d(&rms_input, &weight, 1e-12),
            ))
            .unwrap(),
            2e-6,
        );

        let linear_input = Tensor::new(
            vec![7, 12],
            (0..84)
                .map(|index| ((index * 5) % 29) as f32 / 16.0 - 0.75)
                .collect(),
        )
        .unwrap();
        let linear_weight = Tensor::new(
            vec![13, 12],
            (0..156)
                .map(|index| ((index * 11) % 31) as f32 / 32.0 - 0.4375)
                .collect(),
        )
        .unwrap();
        let linear_bias = Tensor::new(
            vec![13],
            (0..13).map(|index| index as f32 / 64.0 - 0.125).collect(),
        )
        .unwrap();
        verify(
            "linear",
            required(compare_backends(
                &SCALAR_BACKEND,
                &VULKAN_BACKEND,
                |backend| backend.linear(&linear_input, &linear_weight, Some(&linear_bias)),
            ))
            .unwrap(),
            2e-6,
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_resident_linear_reuses_prepared_weight() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        let input = Tensor::new(
            vec![7, 12],
            (0..84)
                .map(|index| ((index * 5) % 29) as f32 / 16.0 - 0.75)
                .collect(),
        )
        .unwrap();
        let weight = Tensor::new(
            vec![13, 12],
            (0..156)
                .map(|index| ((index * 11) % 31) as f32 / 32.0 - 0.4375)
                .collect(),
        )
        .unwrap();
        let bias = Tensor::new(
            vec![13],
            (0..13).map(|index| index as f32 / 64.0 - 0.125).collect(),
        )
        .unwrap();
        let expected = SCALAR_BACKEND
            .silu(&SCALAR_BACKEND.linear(&input, &weight, Some(&bias)).unwrap())
            .unwrap();

        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident Vulkan parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident Vulkan parity failed: {error:#}"),
        };
        let prepared = VULKAN_BACKEND.prepare_linear(&weight, Some(&bias)).unwrap();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let first = VULKAN_BACKEND
            .linear_prepared(&device_input, &prepared)
            .and_then(|output| VULKAN_BACKEND.silu_device(&output))
            .unwrap();
        let second = VULKAN_BACKEND
            .linear_prepared(&device_input, &prepared)
            .and_then(|output| VULKAN_BACKEND.silu_device(&output))
            .unwrap();
        let first = VULKAN_BACKEND.download_tensor(&first).unwrap();
        let second = VULKAN_BACKEND.download_tensor(&second).unwrap();
        let after_execute = crate::vulkan::persistence_stats().unwrap();

        let first_metrics = compare_tensors(&first, &expected).unwrap();
        let repeat_metrics = compare_tensors(&second, &first).unwrap();
        for metrics in [&first_metrics, &repeat_metrics] {
            metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 0.999_999,
                    maximum_absolute_error: 2e-6,
                    maximum_mean_absolute_error: 2e-6,
                })
                .unwrap();
        }
        assert_eq!(first.shape(), &[7, 13]);
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            1
        );
        let expected_weight_bytes = (weight.len() * std::mem::size_of::<u16>()
            + bias.len() * std::mem::size_of::<f32>()) as u64;
        assert_eq!(
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            expected_weight_bytes,
            "the prepared matrix and bias must reside in DEVICE_LOCAL memory"
        );
        assert!(
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes
                >= expected_weight_bytes,
            "device-local Vulkan allocation requirements must cover the logical weights"
        );
        assert_eq!(
            after_execute.resident_weight_uploads, after_prepare.resident_weight_uploads,
            "prepared weights must not be uploaded during execution"
        );
        assert_eq!(
            after_execute.resident_device_local_bytes, after_prepare.resident_device_local_bytes,
            "execution must reuse the prepared device-local buffers"
        );
        assert_eq!(
            after_execute.resident_tensor_uploads - before.resident_tensor_uploads,
            1,
            "the input tensor is uploaded once and reused"
        );
        assert_eq!(
            after_execute.resident_downloads - before.resident_downloads,
            2
        );
        eprintln!(
            "resident linear: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9} repeat_max_abs={:.9} weight_uploads={} device_local_bytes={} device_local_allocation_bytes={}",
            first.shape(),
            first_metrics.cosine_similarity,
            first_metrics.maximum_absolute_error,
            first_metrics.mean_absolute_error,
            repeat_metrics.maximum_absolute_error,
            after_execute.resident_weight_uploads - before.resident_weight_uploads,
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes,
        );
        drop(prepared);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_device_local_bytes, before.resident_device_local_bytes,
            "dropping prepared weights must release logical device-local residency"
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes,
            "dropping prepared weights must free device-local allocations"
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_resident_layernorm_and_wan_modulation_match_scalar() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        let input = Tensor::new(
            vec![5, 7],
            (0..35)
                .map(|index| ((index * 13) % 37) as f32 / 11.0 - 1.5)
                .collect(),
        )
        .unwrap();
        let weight = Tensor::new(vec![7], vec![0.75, -1.25, 0.5, 1.75, -0.25, 0.125, 2.0]).unwrap();
        let bias = Tensor::new(vec![7], vec![-0.5, 0.25, 1.0, -0.75, 0.125, 0.5, -1.0]).unwrap();
        let shared_modulation = Tensor::new(
            vec![1, 42],
            (0..42)
                .map(|index| ((index * 7) % 23) as f32 / 32.0 - 0.25)
                .collect(),
        )
        .unwrap();
        let block_modulation = Tensor::new(
            vec![42],
            (0..42)
                .map(|index| ((index * 5) % 19) as f32 / 64.0 - 0.125)
                .collect(),
        )
        .unwrap();

        let scalar_input = SCALAR_BACKEND.upload_tensor(&input).unwrap();
        let scalar_weight = SCALAR_BACKEND.prepare_vector(&weight).unwrap();
        let scalar_bias = SCALAR_BACKEND.prepare_vector(&bias).unwrap();
        let expected_affine = SCALAR_BACKEND
            .layer_norm_device(
                &scalar_input,
                Some(&scalar_weight),
                Some(&scalar_bias),
                1e-6,
            )
            .and_then(|tensor| SCALAR_BACKEND.download_tensor(&tensor))
            .unwrap();
        let scalar_block = SCALAR_BACKEND.prepare_vector(&block_modulation).unwrap();
        let scalar_modulation = SCALAR_BACKEND
            .upload_tensor(&shared_modulation)
            .and_then(|tensor| SCALAR_BACKEND.add_vector_device(&tensor, &scalar_block))
            .unwrap();
        let expected_modulated = SCALAR_BACKEND
            .layer_norm_device(&scalar_input, None, None, 1e-6)
            .and_then(|tensor| {
                SCALAR_BACKEND.wan_modulate_device(&tensor, &scalar_modulation, 0, 1)
            })
            .and_then(|tensor| SCALAR_BACKEND.download_tensor(&tensor))
            .unwrap();

        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident norm/modulation parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident norm/modulation parity failed: {error:#}"),
        };
        let vulkan_weight = VULKAN_BACKEND.prepare_vector(&weight).unwrap();
        let vulkan_bias = VULKAN_BACKEND.prepare_vector(&bias).unwrap();
        let vulkan_block = VULKAN_BACKEND.prepare_vector(&block_modulation).unwrap();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let device_shared = VULKAN_BACKEND.upload_tensor(&shared_modulation).unwrap();
        let device_modulation = VULKAN_BACKEND
            .add_vector_device(&device_shared, &vulkan_block)
            .unwrap();
        let affine = VULKAN_BACKEND
            .layer_norm_device(
                &device_input,
                Some(&vulkan_weight),
                Some(&vulkan_bias),
                1e-6,
            )
            .unwrap();
        let modulated = VULKAN_BACKEND
            .layer_norm_device(&device_input, None, None, 1e-6)
            .and_then(|tensor| {
                VULKAN_BACKEND.wan_modulate_device(&tensor, &device_modulation, 0, 1)
            })
            .unwrap();
        let repeated = VULKAN_BACKEND
            .layer_norm_device(&device_input, None, None, 1e-6)
            .and_then(|tensor| {
                VULKAN_BACKEND.wan_modulate_device(&tensor, &device_modulation, 0, 1)
            })
            .unwrap();
        let affine = VULKAN_BACKEND.download_tensor(&affine).unwrap();
        let modulated = VULKAN_BACKEND.download_tensor(&modulated).unwrap();
        let repeated = VULKAN_BACKEND.download_tensor(&repeated).unwrap();
        let after_execute = crate::vulkan::persistence_stats().unwrap();

        let affine_metrics = compare_tensors(&affine, &expected_affine).unwrap();
        let modulation_metrics = compare_tensors(&modulated, &expected_modulated).unwrap();
        let repeat_metrics = compare_tensors(&repeated, &modulated).unwrap();
        for metrics in [&affine_metrics, &modulation_metrics] {
            metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 0.999_999,
                    maximum_absolute_error: 3e-6,
                    maximum_mean_absolute_error: 1e-6,
                })
                .unwrap();
        }
        repeat_metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 1.0,
                maximum_absolute_error: 0.0,
                maximum_mean_absolute_error: 0.0,
            })
            .unwrap();
        assert_eq!(affine.shape(), &[5, 7]);
        assert_eq!(modulated.shape(), &[5, 7]);
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            3,
            "weight, bias, and block modulation each use one staging submission"
        );
        let expected_device_local_bytes = ((7 + 7 + 42) * std::mem::size_of::<f32>()) as u64;
        assert_eq!(
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            expected_device_local_bytes
        );
        assert_eq!(
            after_execute.resident_weight_uploads, after_prepare.resident_weight_uploads,
            "resident norm and modulation execution must not upload weights"
        );
        assert_eq!(
            after_execute.resident_device_local_bytes,
            after_prepare.resident_device_local_bytes
        );
        println!(
            "resident LayerNorm/modulation: shape={:?} affine_cosine={:.9} affine_max={:.9} affine_mean={:.9} modulation_cosine={:.9} modulation_max={:.9} modulation_mean={:.9} repeat_max={:.9} device_local_bytes={} device_local_allocation_bytes={}",
            affine.shape(),
            affine_metrics.cosine_similarity,
            affine_metrics.maximum_absolute_error,
            affine_metrics.mean_absolute_error,
            modulation_metrics.cosine_similarity,
            modulation_metrics.maximum_absolute_error,
            modulation_metrics.mean_absolute_error,
            repeat_metrics.maximum_absolute_error,
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes,
        );

        drop(vulkan_weight);
        drop(vulkan_bias);
        drop(vulkan_block);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_resident_rmsnorm_and_rope_match_scalar() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const ROWS: usize = 3;
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const WIDTH: usize = HEADS * HEAD_DIM;
        let input = Tensor::new(
            vec![ROWS, WIDTH],
            (0..ROWS * WIDTH)
                .map(|index| ((index * 11) % 29) as f32 / 9.0 - 1.25)
                .collect(),
        )
        .unwrap();
        let weight = Tensor::new(
            vec![WIDTH],
            vec![0.75, -1.25, 0.5, 1.75, -0.25, 0.125, 2.0, -0.625],
        )
        .unwrap();
        let positions = Tensor::new(
            vec![ROWS, HEAD_DIM / 2, crate::wan_rope::PAIR_STRIDE],
            (0..ROWS)
                .flat_map(|row| {
                    (0..HEAD_DIM / 2).flat_map(move |pair| {
                        let angle = row as f32 * 0.37 + pair as f32 * 0.19;
                        let (sin, cos) = angle.sin_cos();
                        [cos, -sin, sin, cos]
                    })
                })
                .collect(),
        )
        .unwrap();

        let scalar_input = SCALAR_BACKEND.upload_tensor(&input).unwrap();
        let scalar_weight = SCALAR_BACKEND.prepare_vector(&weight).unwrap();
        let scalar_positions = SCALAR_BACKEND.upload_tensor(&positions).unwrap();
        let expected = SCALAR_BACKEND
            .rms_norm_device(&scalar_input, &scalar_weight, 1e-6)
            .and_then(|normalized| {
                SCALAR_BACKEND.rope_device(&normalized, &scalar_positions, HEADS, HEAD_DIM)
            })
            .and_then(|output| SCALAR_BACKEND.download_tensor(&output))
            .unwrap();

        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident RMSNorm/RoPE parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident RMSNorm/RoPE parity failed: {error:#}"),
        };
        let vulkan_weight = VULKAN_BACKEND.prepare_vector(&weight).unwrap();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let device_positions = VULKAN_BACKEND.upload_tensor(&positions).unwrap();
        let actual_device = VULKAN_BACKEND
            .rms_norm_device(&device_input, &vulkan_weight, 1e-6)
            .and_then(|normalized| {
                VULKAN_BACKEND.rope_device(&normalized, &device_positions, HEADS, HEAD_DIM)
            })
            .unwrap();
        let actual = VULKAN_BACKEND.download_tensor(&actual_device).unwrap();
        let after_execute = crate::vulkan::persistence_stats().unwrap();

        let metrics = compare_tensors(&actual, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-6,
                maximum_mean_absolute_error: 1e-6,
            })
            .unwrap();
        assert_eq!(actual.shape(), &[ROWS, WIDTH]);
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            1
        );
        assert_eq!(
            after_execute.resident_tensor_uploads - before.resident_tensor_uploads,
            2,
            "input and position matrices are uploaded exactly once"
        );
        assert_eq!(
            after_execute.resident_downloads - before.resident_downloads,
            1
        );
        println!(
            "resident RMSNorm/RoPE: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9} device_local_bytes={} device_local_allocation_bytes={}",
            actual.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            after_prepare.resident_device_local_allocation_bytes
                - before.resident_device_local_allocation_bytes,
        );

        drop(vulkan_weight);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_allocation_bytes,
            before.resident_device_local_allocation_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_resident_unfused_attention_matches_scalar() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const QUERIES: usize = 5;
        const KEYS: usize = 7;
        const HEADS: usize = 3;
        const HEAD_DIM: usize = 4;
        const WIDTH: usize = HEADS * HEAD_DIM;
        let query = Tensor::new(
            vec![QUERIES, WIDTH],
            (0..QUERIES * WIDTH)
                .map(|index| ((index * 13) % 41) as f32 / 17.0 - 1.0)
                .collect(),
        )
        .unwrap();
        let key = Tensor::new(
            vec![KEYS, WIDTH],
            (0..KEYS * WIDTH)
                .map(|index| ((index * 7) % 37) as f32 / 19.0 - 0.75)
                .collect(),
        )
        .unwrap();
        let value = Tensor::new(
            vec![KEYS, WIDTH],
            (0..KEYS * WIDTH)
                .map(|index| ((index * 11) % 43) as f32 / 23.0 - 0.875)
                .collect(),
        )
        .unwrap();
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let scalar_query = SCALAR_BACKEND.upload_tensor(&query).unwrap();
        let scalar_key = SCALAR_BACKEND.upload_tensor(&key).unwrap();
        let scalar_value = SCALAR_BACKEND.upload_tensor(&value).unwrap();
        let scalar_scores = SCALAR_BACKEND
            .attention_scores_device(&scalar_query, &scalar_key, HEADS, HEAD_DIM, scale)
            .unwrap();
        let scalar_probabilities = SCALAR_BACKEND.softmax_device(&scalar_scores).unwrap();
        let scalar_context = SCALAR_BACKEND
            .attention_values_device(&scalar_probabilities, &scalar_value, HEADS, HEAD_DIM)
            .unwrap();
        let expected_scores = SCALAR_BACKEND.download_tensor(&scalar_scores).unwrap();
        let expected_probabilities = SCALAR_BACKEND
            .download_tensor(&scalar_probabilities)
            .unwrap();
        let expected_context = SCALAR_BACKEND.download_tensor(&scalar_context).unwrap();

        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident attention parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident attention parity failed: {error:#}"),
        };
        let device_query = VULKAN_BACKEND.upload_tensor(&query).unwrap();
        let device_key = VULKAN_BACKEND.upload_tensor(&key).unwrap();
        let device_value = VULKAN_BACKEND.upload_tensor(&value).unwrap();
        let device_scores = VULKAN_BACKEND
            .attention_scores_device(&device_query, &device_key, HEADS, HEAD_DIM, scale)
            .unwrap();
        let device_probabilities = VULKAN_BACKEND.softmax_device(&device_scores).unwrap();
        let device_context = VULKAN_BACKEND
            .attention_values_device(&device_probabilities, &device_value, HEADS, HEAD_DIM)
            .unwrap();
        let scores = VULKAN_BACKEND.download_tensor(&device_scores).unwrap();
        let probabilities = VULKAN_BACKEND
            .download_tensor(&device_probabilities)
            .unwrap();
        let context = VULKAN_BACKEND.download_tensor(&device_context).unwrap();
        let after = crate::vulkan::persistence_stats().unwrap();

        let score_metrics = compare_tensors(&scores, &expected_scores).unwrap();
        let probability_metrics = compare_tensors(&probabilities, &expected_probabilities).unwrap();
        let context_metrics = compare_tensors(&context, &expected_context).unwrap();
        for metrics in [&score_metrics, &probability_metrics, &context_metrics] {
            metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 0.999_999,
                    maximum_absolute_error: 3e-6,
                    maximum_mean_absolute_error: 1e-6,
                })
                .unwrap();
        }
        assert_eq!(scores.shape(), &[HEADS, QUERIES, KEYS]);
        assert_eq!(probabilities.shape(), &[HEADS, QUERIES, KEYS]);
        assert_eq!(context.shape(), &[QUERIES, WIDTH]);
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            3,
            "Q, K, and V are uploaded once"
        );
        assert_eq!(
            after.resident_downloads - before.resident_downloads,
            3,
            "scores, probabilities, and context are downloaded for parity"
        );
        println!(
            "resident unfused attention: scores={:?} score_cosine={:.9} score_max={:.9} score_mean={:.9} probability_cosine={:.9} probability_max={:.9} probability_mean={:.9} context={:?} context_cosine={:.9} context_max={:.9} context_mean={:.9}",
            scores.shape(),
            score_metrics.cosine_similarity,
            score_metrics.maximum_absolute_error,
            score_metrics.mean_absolute_error,
            probability_metrics.cosine_similarity,
            probability_metrics.maximum_absolute_error,
            probability_metrics.mean_absolute_error,
            context.shape(),
            context_metrics.cosine_similarity,
            context_metrics.maximum_absolute_error,
            context_metrics.mean_absolute_error,
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_resident_wan_patch_layout_and_head_match_scalar() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        const CHANNELS: usize = 2;
        const TIME: usize = 2;
        const HEIGHT: usize = 4;
        const WIDTH: usize = 6;
        const PATCH: (usize, usize, usize) = (1, 2, 3);
        const TOKENS: usize = 8;
        const FEATURES: usize = 12;
        let latent = Tensor::new(
            vec![CHANNELS, TIME, HEIGHT, WIDTH],
            (0..CHANNELS * TIME * HEIGHT * WIDTH)
                .map(|index| index as f32 * 0.125 - 3.0)
                .collect(),
        )
        .unwrap();
        let token_output = Tensor::new(
            vec![TOKENS, FEATURES],
            (0..TOKENS * FEATURES)
                .map(|index| ((index * 7) % 31) as f32 * 0.0625 - 0.75)
                .collect(),
        )
        .unwrap();
        let head_input = Tensor::new(
            vec![3, 8],
            (0..24)
                .map(|index| ((index * 11) % 29) as f32 / 13.0 - 0.8)
                .collect(),
        )
        .unwrap();
        let timestep = Tensor::new(
            vec![1, 8],
            (0..8).map(|index| index as f32 * 0.03 - 0.1).collect(),
        )
        .unwrap();
        let modulation = Tensor::new(
            vec![16],
            (0..16)
                .map(|index| ((index * 5) % 17) as f32 * 0.02 - 0.15)
                .collect(),
        )
        .unwrap();

        let scalar_latent = SCALAR_BACKEND.upload_tensor(&latent).unwrap();
        let scalar_patch = SCALAR_BACKEND
            .patchify_device(&scalar_latent, PATCH)
            .unwrap();
        let expected_patch = SCALAR_BACKEND.download_tensor(&scalar_patch).unwrap();
        let scalar_tokens = SCALAR_BACKEND.upload_tensor(&token_output).unwrap();
        let scalar_unpatch = SCALAR_BACKEND
            .unpatchify_device(&scalar_tokens, CHANNELS, (TIME, HEIGHT, WIDTH), PATCH)
            .unwrap();
        let expected_unpatch = SCALAR_BACKEND.download_tensor(&scalar_unpatch).unwrap();
        let scalar_head_input = SCALAR_BACKEND.upload_tensor(&head_input).unwrap();
        let scalar_timestep = SCALAR_BACKEND.upload_tensor(&timestep).unwrap();
        let scalar_modulation = SCALAR_BACKEND.prepare_vector(&modulation).unwrap();
        let scalar_head = SCALAR_BACKEND
            .wan_head_modulate_device(&scalar_head_input, &scalar_timestep, &scalar_modulation)
            .unwrap();
        let expected_head = SCALAR_BACKEND.download_tensor(&scalar_head).unwrap();

        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident Wan layout/head parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident Wan layout/head parity failed: {error:#}"),
        };
        let device_latent = VULKAN_BACKEND.upload_tensor(&latent).unwrap();
        let device_patch = VULKAN_BACKEND
            .patchify_device(&device_latent, PATCH)
            .unwrap();
        let patch = VULKAN_BACKEND.download_tensor(&device_patch).unwrap();
        let device_tokens = VULKAN_BACKEND.upload_tensor(&token_output).unwrap();
        let device_unpatch = VULKAN_BACKEND
            .unpatchify_device(&device_tokens, CHANNELS, (TIME, HEIGHT, WIDTH), PATCH)
            .unwrap();
        let unpatch = VULKAN_BACKEND.download_tensor(&device_unpatch).unwrap();
        let device_head_input = VULKAN_BACKEND.upload_tensor(&head_input).unwrap();
        let device_timestep = VULKAN_BACKEND.upload_tensor(&timestep).unwrap();
        let device_modulation = VULKAN_BACKEND.prepare_vector(&modulation).unwrap();
        let device_head = VULKAN_BACKEND
            .wan_head_modulate_device(&device_head_input, &device_timestep, &device_modulation)
            .unwrap();
        let head = VULKAN_BACKEND.download_tensor(&device_head).unwrap();

        let patch_metrics = compare_tensors(&patch, &expected_patch).unwrap();
        let unpatch_metrics = compare_tensors(&unpatch, &expected_unpatch).unwrap();
        let head_metrics = compare_tensors(&head, &expected_head).unwrap();
        for metrics in [&patch_metrics, &unpatch_metrics] {
            metrics
                .require(ParityTolerance {
                    minimum_cosine_similarity: 1.0,
                    maximum_absolute_error: 0.0,
                    maximum_mean_absolute_error: 0.0,
                })
                .unwrap();
        }
        head_metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-6,
                maximum_mean_absolute_error: 1e-6,
            })
            .unwrap();
        assert_eq!(patch.shape(), &[TOKENS, FEATURES]);
        assert_eq!(unpatch.shape(), &[CHANNELS, TIME, HEIGHT, WIDTH]);
        assert_eq!(head.shape(), &[3, 8]);
        println!(
            "resident Wan layout/head: patch={:?} patch_cosine={:.9} patch_max={:.9} unpatch={:?} unpatch_cosine={:.9} unpatch_max={:.9} head={:?} head_cosine={:.9} head_max={:.9} head_mean={:.9}",
            patch.shape(),
            patch_metrics.cosine_similarity,
            patch_metrics.maximum_absolute_error,
            unpatch.shape(),
            unpatch_metrics.cosine_similarity,
            unpatch_metrics.maximum_absolute_error,
            head.shape(),
            head_metrics.cosine_similarity,
            head_metrics.maximum_absolute_error,
            head_metrics.mean_absolute_error,
        );

        drop(device_head);
        drop(device_modulation);
        drop(device_timestep);
        drop(device_head_input);
        drop(device_unpatch);
        drop(device_tokens);
        drop(device_patch);
        drop(device_latent);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_resident_causal_conv3d_matches_scalar() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();

        let input = Tensor::new(
            vec![1, 2, 3, 4, 5],
            (0..120)
                .map(|index| ((index * 11) % 37) as f32 * 0.125 - 2.0)
                .collect(),
        )
        .unwrap();
        let weight = Tensor::new(
            vec![3, 2, 3, 3, 3],
            (0..162)
                .map(|index| ((index * 5) % 13) as f32 * 0.0625 - 0.375)
                .collect(),
        )
        .unwrap();
        let bias = Tensor::new(vec![3], vec![0.125, -0.25, 0.375]).unwrap();
        let padding_before = [2, 1, 1];
        let padding_after = [0, 1, 1];

        let scalar_input = SCALAR_BACKEND.upload_tensor(&input).unwrap();
        let scalar_weight = SCALAR_BACKEND.prepare_conv3d(&weight, Some(&bias)).unwrap();
        let scalar_output = SCALAR_BACKEND
            .conv3d_prepared_device(&scalar_input, &scalar_weight, padding_before, padding_after)
            .unwrap();
        let expected = SCALAR_BACKEND.download_tensor(&scalar_output).unwrap();

        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping resident causal Conv3D parity: {error:#}");
                return;
            }
            Err(error) => panic!("required resident causal Conv3D parity failed: {error:#}"),
        };
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let device_weight = VULKAN_BACKEND.prepare_conv3d(&weight, Some(&bias)).unwrap();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();
        let device_output = VULKAN_BACKEND
            .conv3d_prepared_device(&device_input, &device_weight, padding_before, padding_after)
            .unwrap();
        let output = VULKAN_BACKEND.download_tensor(&device_output).unwrap();
        let after = crate::vulkan::persistence_stats().unwrap();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-5,
                maximum_mean_absolute_error: 2e-6,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, 3, 3, 4, 5]);
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            1
        );
        assert_eq!(
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            (weight.len() * 2 + bias.len() * 4) as u64
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "resident causal Conv3D: input={:?} weight={:?} output={:?} padding_before={padding_before:?} padding_after={padding_after:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            input.shape(),
            weight.shape(),
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
        );

        drop(device_output);
        drop(device_weight);
        drop(device_input);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads the Wan VAE weights; run explicitly for real-layer Vulkan parity"]
    fn real_wan_vae_head_norm_matches_scalar() {
        use crate::{
            parity::{ParityTolerance, compare_backends},
            safetensors::SafeTensorFile,
        };

        const VAE: &str =
            "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan_2.1_vae.safetensors";
        let weights = SafeTensorFile::open(VAE).unwrap();
        let stored_gamma = crate::sd_ops::load_tensor(&weights, "decoder.head.0.gamma").unwrap();
        assert_eq!(stored_gamma.shape(), &[96, 1, 1, 1]);
        let gamma = Tensor::new(vec![stored_gamma.len()], stored_gamma.data().to_vec()).unwrap();
        assert_eq!(gamma.shape(), &[96]);
        let input = Tensor::new(
            vec![1, 96, 1, 3, 5],
            (0..96 * 15)
                .map(|index| ((index * 17) as f32 * 0.013).sin() * 1.75)
                .collect(),
        )
        .unwrap();
        let parity = compare_backends(&SCALAR_BACKEND, &VULKAN_BACKEND, |backend| {
            backend.channel_rms_norm_3d(&input, &gamma, 1e-12)
        })
        .unwrap();
        eprintln!(
            "Wan decoder.head.0 channel RMSNorm: shape={:?} scalar_us={} vulkan_us={} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            parity.metrics.shape,
            parity.reference_runtime.as_micros(),
            parity.candidate_runtime.as_micros(),
            parity.metrics.cosine_similarity,
            parity.metrics.maximum_absolute_error,
            parity.metrics.mean_absolute_error,
        );
        parity
            .metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-6,
                maximum_mean_absolute_error: 2e-6,
            })
            .unwrap();
        crate::vulkan::print_statistics();
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads the Wan DiT weights; run explicitly for real-layer Vulkan parity"]
    fn real_wan_dit_time_linear_matches_scalar() {
        use crate::{
            dequant,
            gguf::GgufFile,
            parity::{ParityTolerance, compare_backends},
        };

        const DIT: &str = "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan2.1_t2v_1.3B_Q4_K.gguf";
        const PREFIX: &str = "model.diffusion_model.time_embedding.0";
        let gguf = GgufFile::open(std::path::Path::new(DIT)).unwrap();
        let tensors = gguf.tensor_map();
        let weight_name = format!("{PREFIX}.weight");
        let bias_name = format!("{PREFIX}.bias");
        let weight_info = tensors.get(weight_name.as_str()).unwrap();
        let bias_info = tensors.get(bias_name.as_str()).unwrap();
        assert_eq!(weight_info.dims, [256, 1536]);
        assert_eq!(bias_info.dims, [1536]);
        let weight = Tensor::new(
            vec![1536, 256],
            dequant::dequant(
                gguf.tensor_data(weight_info),
                weight_info.ggml_type,
                weight_info.n_elems(),
            ),
        )
        .unwrap();
        let bias = Tensor::new(
            vec![1536],
            dequant::dequant(
                gguf.tensor_data(bias_info),
                bias_info.ggml_type,
                bias_info.n_elems(),
            ),
        )
        .unwrap();
        let input =
            Tensor::new(vec![1, 256], crate::wan_dit::timestep_embedding(750.0, 256)).unwrap();
        let parity = compare_backends(&SCALAR_BACKEND, &VULKAN_BACKEND, |backend| {
            backend.linear(&input, &weight, Some(&bias))
        })
        .unwrap();
        eprintln!(
            "Wan time_embedding.0 linear: shape={:?} scalar_us={} vulkan_us={} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            parity.metrics.shape,
            parity.reference_runtime.as_micros(),
            parity.candidate_runtime.as_micros(),
            parity.metrics.cosine_similarity,
            parity.metrics.maximum_absolute_error,
            parity.metrics.mean_absolute_error,
        );
        parity
            .metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_99,
                maximum_absolute_error: 0.01,
                maximum_mean_absolute_error: 0.001,
            })
            .unwrap();
        crate::vulkan::print_statistics();
    }
}
