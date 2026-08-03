//! Wan 2.1 3D causal VAE decoder.
//!
//! The decoder is deliberately stateful: the latent is processed one frame at
//! a time and every causal 3x3x3 convolution retains its last two input feature
//! frames. Tensor layout is contiguous NCTHW, unlike SQD1/ggml dumps whose
//! dimension list is fastest-first `[W,H,T,C,N]`.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

#[cfg(feature = "vulkan")]
use crate::backend::{BackendKind, Conv2dWeightHandle, PreparedVectorHandle};
use crate::{
    backend::{Conv3dWeightHandle, DeviceTensor, SCALAR_BACKEND, TensorBackend},
    safetensors::SafeTensorFile,
    sd_ops,
    tensor::Tensor,
};

const RMS_EPSILON: f32 = 1e-12;
const CACHE_TIME: usize = 2;
const CACHE_SLOTS: usize = 33;
const USED_CACHE_SLOTS: usize = 32;

struct CausalConv3d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
}

struct PreparedCausalConv3d {
    weights: Conv3dWeightHandle,
}

impl CausalConv3d {
    fn load(
        weights: &SafeTensorFile,
        prefix: &str,
        stride: [usize; 3],
        padding: [usize; 3],
    ) -> Result<Self> {
        Ok(Self {
            weight: sd_ops::load_tensor(weights, &format!("{prefix}.weight"))?,
            bias: weights
                .info(&format!("{prefix}.bias"))
                .map(|_| sd_ops::load_tensor(weights, &format!("{prefix}.bias")))
                .transpose()?,
            stride,
            padding,
            dilation: [1, 1, 1],
        })
    }

    fn forward(&self, input: &Tensor, cache: Option<&Tensor>) -> Result<Tensor> {
        let mut padded_input = input.clone();
        let mut padding_before = [
            self.padding[0]
                .checked_mul(2)
                .context("causal temporal padding overflow")?,
            self.padding[1],
            self.padding[2],
        ];
        if let Some(cache) = cache {
            require_matching_nchw_axes(input, cache, "causal convolution cache")?;
            let cache_time = cache.shape()[2];
            if cache_time > padding_before[0] {
                bail!(
                    "causal convolution cache has {cache_time} frames but temporal padding is {}",
                    padding_before[0]
                );
            }
            padded_input = concat_time(&[cache, input])?;
            padding_before[0] -= cache_time;
        }
        padded_input.conv3d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            padding_before,
            [0, self.padding[1], self.padding[2]],
            self.dilation,
            1,
        )
    }

    fn prepare(&self, backend: &dyn TensorBackend) -> Result<PreparedCausalConv3d> {
        if self.stride != [1, 1, 1] || self.dilation != [1, 1, 1] {
            bail!("resident Wan causal Conv3D currently requires unit stride and dilation");
        }
        Ok(PreparedCausalConv3d {
            weights: backend.prepare_conv3d(&self.weight, self.bias.as_ref())?,
        })
    }

    fn forward_uncached_with_backend(
        &self,
        input: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedCausalConv3d,
    ) -> Result<DeviceTensor> {
        let padding_before = [
            self.padding[0]
                .checked_mul(2)
                .context("resident causal temporal padding overflow")?,
            self.padding[1],
            self.padding[2],
        ];
        backend.conv3d_prepared_device(
            input,
            &prepared.weights,
            padding_before,
            [0, self.padding[1], self.padding[2]],
        )
    }

    #[cfg(feature = "vulkan")]
    fn forward_with_backend(
        &self,
        input: &DeviceTensor,
        cache: Option<&DeviceTensor>,
        backend: &dyn TensorBackend,
        prepared: &PreparedCausalConv3d,
    ) -> Result<DeviceTensor> {
        let mut convolution_input = input.clone();
        let mut padding_before = [
            self.padding[0]
                .checked_mul(2)
                .context("resident causal temporal padding overflow")?,
            self.padding[1],
            self.padding[2],
        ];
        if let Some(cache) = cache {
            require_matching_device_nchw_axes(input, cache, "resident causal convolution cache")?;
            let cache_time = cache.shape()[2];
            if cache_time > padding_before[0] {
                bail!(
                    "resident causal convolution cache has {cache_time} frames but temporal padding is {}",
                    padding_before[0]
                );
            }
            convolution_input = backend.ncthw_concat_time_device(&[cache, input])?;
            padding_before[0] -= cache_time;
        }
        backend.conv3d_prepared_device(
            &convolution_input,
            &prepared.weights,
            padding_before,
            [0, self.padding[1], self.padding[2]],
        )
    }
}

struct Conv2dFrames {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: [usize; 2],
    padding: [usize; 2],
}

#[cfg(feature = "vulkan")]
struct PreparedConv2dFrames {
    weights: Conv2dWeightHandle,
}

impl Conv2dFrames {
    fn load(
        weights: &SafeTensorFile,
        prefix: &str,
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Result<Self> {
        Ok(Self {
            weight: sd_ops::load_tensor(weights, &format!("{prefix}.weight"))?,
            bias: weights
                .info(&format!("{prefix}.bias"))
                .map(|_| sd_ops::load_tensor(weights, &format!("{prefix}.bias")))
                .transpose()?,
            stride,
            padding,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let [batch, channels, time, height, width] = ncthw(input)?;
        let plane = height * width;
        let mut frame_data = vec![0.0; input.len()];
        frame_data.par_chunks_mut(plane).enumerate().for_each(
            |(destination_plane, destination)| {
                let channel = destination_plane % channels;
                let sample_time = destination_plane / channels;
                let sample = sample_time / time;
                let frame = sample_time % time;
                let source_plane = (sample * channels + channel) * time + frame;
                destination.copy_from_slice(
                    &input.data()[source_plane * plane..(source_plane + 1) * plane],
                );
            },
        );
        let frames = Tensor::new(vec![batch * time, channels, height, width], frame_data)?;
        let output = frames.conv2d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            [1, 1],
            1,
        )?;
        let output_channels = output.shape()[1];
        let output_height = output.shape()[2];
        let output_width = output.shape()[3];
        let output_plane = output_height * output_width;
        let mut data = vec![0.0; output.len()];
        data.par_chunks_mut(output_plane).enumerate().for_each(
            |(destination_plane, destination)| {
                let frame = destination_plane % time;
                let sample_channel = destination_plane / time;
                let sample = sample_channel / output_channels;
                let channel = sample_channel % output_channels;
                let source_plane = (sample * time + frame) * output_channels + channel;
                destination.copy_from_slice(
                    &output.data()[source_plane * output_plane..(source_plane + 1) * output_plane],
                );
            },
        );
        Tensor::new(
            vec![batch, output_channels, time, output_height, output_width],
            data,
        )
    }

    #[cfg(feature = "vulkan")]
    fn prepare(&self, backend: &dyn TensorBackend) -> Result<PreparedConv2dFrames> {
        Ok(PreparedConv2dFrames {
            weights: backend.prepare_conv2d(&self.weight, self.bias.as_ref())?,
        })
    }

    #[cfg(feature = "vulkan")]
    fn forward_frames_with_backend(
        &self,
        input: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedConv2dFrames,
    ) -> Result<DeviceTensor> {
        let frames = backend.ncthw_to_nchw_frames_device(input)?;
        backend.conv2d_prepared_device(&frames, &prepared.weights, self.stride, self.padding)
    }

    #[cfg(feature = "vulkan")]
    fn forward_with_backend(
        &self,
        input: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedConv2dFrames,
    ) -> Result<DeviceTensor> {
        let [batch, _, time, _, _]: [usize; 5] = input
            .shape()
            .try_into()
            .context("resident frame Conv2D input must be NCTHW")?;
        let frames = self.forward_frames_with_backend(input, backend, prepared)?;
        backend.nchw_frames_to_ncthw_device(&frames, batch, time)
    }
}

struct FeatureCache {
    slots: Vec<Option<Tensor>>,
    index: usize,
}

impl FeatureCache {
    fn new() -> Self {
        Self {
            slots: vec![None; CACHE_SLOTS],
            index: 0,
        }
    }

    fn begin_chunk(&mut self) {
        self.index = 0;
    }

    fn take_index(&mut self) -> Result<usize> {
        if self.index >= self.slots.len() {
            bail!("Wan VAE feature-cache index {} is out of range", self.index);
        }
        let index = self.index;
        self.index += 1;
        Ok(index)
    }
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DeviceFeatureCacheStats {
    current_bytes: usize,
    peak_bytes: usize,
    occupied_slots: usize,
    replaced_slots: usize,
    evicted_slots: usize,
    all_slots_resident: bool,
    all_slots_device_local: bool,
}

/// Device-owned counterpart of Wan's scalar feature cache. Cloning a slot
/// clones only the resident-buffer lease; tensor contents never return to the
/// host. Exactly the 32 cache-consuming decoder layers have slots.
#[cfg(feature = "vulkan")]
struct DeviceFeatureCache {
    slots: [Option<DeviceTensor>; USED_CACHE_SLOTS],
    active_index: usize,
    current_bytes: usize,
    peak_bytes: usize,
    replaced_slots: usize,
    evicted_slots: usize,
}

#[cfg(feature = "vulkan")]
impl DeviceFeatureCache {
    fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            active_index: 0,
            current_bytes: 0,
            peak_bytes: 0,
            replaced_slots: 0,
            evicted_slots: 0,
        }
    }

    fn begin_chunk(&mut self) {
        self.active_index = 0;
    }

    fn take_index(&mut self) -> Result<usize> {
        if self.active_index >= self.slots.len() {
            bail!(
                "Wan VAE device feature-cache index {} is out of range",
                self.active_index
            );
        }
        let index = self.active_index;
        self.active_index += 1;
        Ok(index)
    }

    fn temporal_prefix(&self, index: usize, input: &DeviceTensor) -> Result<Option<DeviceTensor>> {
        let slot = self.slots.get(index).with_context(|| {
            format!("Wan VAE device feature-cache slot {index} is out of range")
        })?;
        if let Some(prefix) = slot {
            require_matching_device_nchw_axes(input, prefix, "device feature-cache prefix")?;
            if prefix.shape()[2] > CACHE_TIME {
                bail!(
                    "device feature-cache prefix has {} frames, maximum is {CACHE_TIME}",
                    prefix.shape()[2]
                );
            }
        }
        Ok(slot.clone())
    }

    fn replace(&mut self, index: usize, tensor: DeviceTensor) -> Result<()> {
        if tensor.backend_kind() != BackendKind::Vulkan || !tensor.remains_resident() {
            bail!("Wan VAE device feature-cache accepts only resident Vulkan tensors");
        }
        let shape: [usize; 5] = tensor
            .shape()
            .try_into()
            .context("Wan VAE device feature-cache tensor must be NCTHW")?;
        if shape.contains(&0) || shape[2] > CACHE_TIME {
            bail!(
                "Wan VAE device feature-cache tensor has invalid shape {:?}",
                tensor.shape()
            );
        }
        let slot = self.slots.get_mut(index).with_context(|| {
            format!("Wan VAE device feature-cache slot {index} is out of range")
        })?;
        if let Some(previous) = slot.replace(tensor) {
            self.current_bytes = self.current_bytes.saturating_sub(previous.byte_len());
            self.replaced_slots += 1;
        }
        self.current_bytes = self
            .current_bytes
            .checked_add(slot.as_ref().expect("slot was just populated").byte_len())
            .context("Wan VAE device feature-cache byte count overflow")?;
        self.peak_bytes = self.peak_bytes.max(self.current_bytes);
        Ok(())
    }

    fn reset(&mut self) {
        let occupied = self.slots.iter().filter(|slot| slot.is_some()).count();
        for slot in &mut self.slots {
            *slot = None;
        }
        self.evicted_slots += occupied;
        self.active_index = 0;
        self.current_bytes = 0;
    }

    fn stats(&self) -> Result<DeviceFeatureCacheStats> {
        let occupied = self.slots.iter().flatten().collect::<Vec<_>>();
        let occupied_slots = occupied.len();
        let all_slots_resident =
            occupied_slots > 0 && occupied.iter().all(|tensor| tensor.remains_resident());
        let all_slots_device_local = occupied_slots > 0
            && occupied
                .iter()
                .map(|tensor| tensor.is_device_local())
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .all(|is_device_local| is_device_local);
        Ok(DeviceFeatureCacheStats {
            current_bytes: self.current_bytes,
            peak_bytes: self.peak_bytes,
            occupied_slots,
            replaced_slots: self.replaced_slots,
            evicted_slots: self.evicted_slots,
            all_slots_resident,
            all_slots_device_local,
        })
    }
}

#[cfg(feature = "vulkan")]
fn cached_causal_conv_with_backend(
    layer: &CausalConv3d,
    input: &DeviceTensor,
    cache: &mut DeviceFeatureCache,
    backend: &dyn TensorBackend,
    prepared: &PreparedCausalConv3d,
) -> Result<DeviceTensor> {
    let index = cache.take_index()?;
    let previous = cache.temporal_prefix(index, input)?;
    let input_time = input.shape()[2];
    let next_start = input_time.saturating_sub(CACHE_TIME);
    let mut next = backend.ncthw_slice_time_device(input, next_start, input_time - next_start)?;
    if next.shape()[2] < CACHE_TIME
        && let Some(previous) = previous.as_ref()
    {
        let previous_time = previous.shape()[2];
        let previous_tail = backend.ncthw_slice_time_device(previous, previous_time - 1, 1)?;
        next = backend.ncthw_concat_time_device(&[&previous_tail, &next])?;
    }
    let output = layer.forward_with_backend(input, previous.as_ref(), backend, prepared)?;
    cache.replace(index, next)?;
    Ok(output)
}

fn cached_causal_conv(
    layer: &CausalConv3d,
    input: &Tensor,
    cache: &mut FeatureCache,
) -> Result<Tensor> {
    let index = cache.take_index()?;
    let previous = cache.slots[index].clone();
    let mut next = last_time(input, CACHE_TIME)?;
    if next.shape()[2] < CACHE_TIME {
        if let Some(previous) = previous.as_ref() {
            next = concat_time(&[&last_time(previous, 1)?, &next])?;
        }
    }
    let output = layer.forward(input, previous.as_ref())?;
    cache.slots[index] = Some(next);
    Ok(output)
}

struct ResidualBlock {
    norm1: Tensor,
    conv1: CausalConv3d,
    norm2: Tensor,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

#[cfg(feature = "vulkan")]
struct PreparedResidualBlock {
    norm1: PreparedVectorHandle,
    conv1: PreparedCausalConv3d,
    norm2: PreparedVectorHandle,
    conv2: PreparedCausalConv3d,
    shortcut: Option<PreparedCausalConv3d>,
}

impl ResidualBlock {
    fn load(weights: &SafeTensorFile, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm1: load_flat(weights, &format!("{prefix}.residual.0.gamma"))?,
            conv1: CausalConv3d::load(
                weights,
                &format!("{prefix}.residual.2"),
                [1, 1, 1],
                [1, 1, 1],
            )?,
            norm2: load_flat(weights, &format!("{prefix}.residual.3.gamma"))?,
            conv2: CausalConv3d::load(
                weights,
                &format!("{prefix}.residual.6"),
                [1, 1, 1],
                [1, 1, 1],
            )?,
            shortcut: weights
                .info(&format!("{prefix}.shortcut.weight"))
                .map(|_| {
                    CausalConv3d::load(weights, &format!("{prefix}.shortcut"), [1, 1, 1], [0, 0, 0])
                })
                .transpose()?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        cache: &mut FeatureCache,
        backend: &dyn TensorBackend,
    ) -> Result<Tensor> {
        let residual = self
            .shortcut
            .as_ref()
            .map(|layer| layer.forward(input, None))
            .transpose()?
            .unwrap_or_else(|| input.clone());
        let hidden = input.channel_rms_norm_3d(&self.norm1, RMS_EPSILON)?.silu();
        let hidden = cached_causal_conv(&self.conv1, &hidden, cache)?;
        let hidden = hidden.channel_rms_norm_3d(&self.norm2, RMS_EPSILON)?.silu();
        backend.add(&cached_causal_conv(&self.conv2, &hidden, cache)?, &residual)
    }

    #[cfg(feature = "vulkan")]
    fn prepare(&self, backend: &dyn TensorBackend) -> Result<PreparedResidualBlock> {
        Ok(PreparedResidualBlock {
            norm1: backend.prepare_vector(&self.norm1)?,
            conv1: self.conv1.prepare(backend)?,
            norm2: backend.prepare_vector(&self.norm2)?,
            conv2: self.conv2.prepare(backend)?,
            shortcut: self
                .shortcut
                .as_ref()
                .map(|shortcut| shortcut.prepare(backend))
                .transpose()?,
        })
    }

    #[cfg(feature = "vulkan")]
    fn forward_with_backend(
        &self,
        input: &DeviceTensor,
        cache: &mut DeviceFeatureCache,
        backend: &dyn TensorBackend,
        prepared: &PreparedResidualBlock,
    ) -> Result<DeviceTensor> {
        let residual = if let Some(shortcut) = self.shortcut.as_ref() {
            let prepared_shortcut = prepared
                .shortcut
                .as_ref()
                .context("prepared residual block is missing shortcut weights")?;
            shortcut.forward_with_backend(input, None, backend, prepared_shortcut)?
        } else {
            if prepared.shortcut.is_some() {
                bail!("prepared residual block contains unexpected shortcut weights");
            }
            input.clone()
        };
        let hidden = backend.channel_rms_norm_3d_device(input, &prepared.norm1, RMS_EPSILON)?;
        let hidden = backend.silu_device(&hidden)?;
        let hidden =
            cached_causal_conv_with_backend(&self.conv1, &hidden, cache, backend, &prepared.conv1)?;
        let hidden = backend.channel_rms_norm_3d_device(&hidden, &prepared.norm2, RMS_EPSILON)?;
        let hidden = backend.silu_device(&hidden)?;
        let hidden =
            cached_causal_conv_with_backend(&self.conv2, &hidden, cache, backend, &prepared.conv2)?;
        backend.add_device(&hidden, &residual)
    }
}

struct SpatialAttention {
    norm: Tensor,
    qkv: Conv2dFrames,
    projection: Conv2dFrames,
}

#[cfg(feature = "vulkan")]
struct PreparedSpatialAttention {
    norm: PreparedVectorHandle,
    qkv: PreparedConv2dFrames,
    projection: PreparedConv2dFrames,
}

impl SpatialAttention {
    fn load(weights: &SafeTensorFile, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: load_flat(weights, &format!("{prefix}.norm.gamma"))?,
            qkv: Conv2dFrames::load(weights, &format!("{prefix}.to_qkv"), [1, 1], [0, 0])?,
            projection: Conv2dFrames::load(weights, &format!("{prefix}.proj"), [1, 1], [0, 0])?,
        })
    }

    fn forward(&self, input: &Tensor, backend: &dyn TensorBackend) -> Result<Tensor> {
        let [batch, channels, time, height, width] = ncthw(input)?;
        let positions = height
            .checked_mul(width)
            .context("Wan VAE attention position overflow")?;
        let normalized = input.channel_rms_norm_3d(&self.norm, RMS_EPSILON)?;
        let qkv = self.qkv.forward(&normalized)?;
        if qkv.shape()[1] != channels * 3 {
            bail!(
                "Wan VAE QKV produced {} channels, expected {}",
                qkv.shape()[1],
                channels * 3
            );
        }
        let sequence_len = batch * time * positions * channels;
        let mut query = vec![0.0; sequence_len];
        let mut key = vec![0.0; sequence_len];
        let mut value = vec![0.0; sequence_len];
        query
            .par_chunks_mut(channels)
            .zip(key.par_chunks_mut(channels))
            .zip(value.par_chunks_mut(channels))
            .enumerate()
            .for_each(|(row, ((query, key), value))| {
                let position = row % positions;
                let sample_time = row / positions;
                let sample = sample_time / time;
                let frame = sample_time % time;
                for channel in 0..channels {
                    let index = |qkv_channel: usize| {
                        ((sample * channels * 3 + qkv_channel) * time + frame) * positions
                            + position
                    };
                    query[channel] = qkv.data()[index(channel)];
                    key[channel] = qkv.data()[index(channels + channel)];
                    value[channel] = qkv.data()[index(channels * 2 + channel)];
                }
            });
        let shape = vec![batch * time, 1, positions, channels];
        let attended = Tensor::attention(
            &Tensor::new(shape.clone(), query)?,
            &Tensor::new(shape.clone(), key)?,
            &Tensor::new(shape, value)?,
        )?;
        let mut attended_ncthw = vec![0.0; input.len()];
        attended_ncthw
            .par_chunks_mut(positions)
            .enumerate()
            .for_each(|(destination_plane, destination)| {
                let frame = destination_plane % time;
                let sample_channel = destination_plane / time;
                let sample = sample_channel / channels;
                let channel = sample_channel % channels;
                for position in 0..positions {
                    destination[position] = attended.data()
                        [((sample * time + frame) * positions + position) * channels + channel];
                }
            });
        let attended = Tensor::new(vec![batch, channels, time, height, width], attended_ncthw)?;
        backend.add(&self.projection.forward(&attended)?, input)
    }

    #[cfg(feature = "vulkan")]
    fn prepare(&self, backend: &dyn TensorBackend) -> Result<PreparedSpatialAttention> {
        Ok(PreparedSpatialAttention {
            norm: backend.prepare_vector(&self.norm)?,
            qkv: self.qkv.prepare(backend)?,
            projection: self.projection.prepare(backend)?,
        })
    }

    #[cfg(feature = "vulkan")]
    fn forward_with_backend(
        &self,
        input: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedSpatialAttention,
    ) -> Result<DeviceTensor> {
        let [batch, channels, time, height, width]: [usize; 5] = input
            .shape()
            .try_into()
            .context("resident Wan VAE spatial attention input must be NCTHW")?;
        let positions = height
            .checked_mul(width)
            .context("resident Wan VAE attention position overflow")?;
        let normalized = backend.channel_rms_norm_3d_device(input, &prepared.norm, RMS_EPSILON)?;
        let qkv_frames =
            self.qkv
                .forward_frames_with_backend(&normalized, backend, &prepared.qkv)?;
        if qkv_frames.shape()[1] != channels * 3 {
            bail!(
                "resident Wan VAE QKV produced {} channels, expected {}",
                qkv_frames.shape()[1],
                channels * 3
            );
        }
        let (query, key, value) = backend.vae_qkv_to_sequences_device(&qkv_frames)?;
        let scale = 1.0 / (channels as f32).sqrt();
        let scores = backend.attention_scores_device(&query, &key, 1, channels, scale)?;
        let probabilities = backend.softmax_device(&scores)?;
        let attended = backend.attention_values_device(&probabilities, &value, 1, channels)?;
        if attended.shape() != [batch * time, positions, channels] {
            bail!("resident Wan VAE attention returned the wrong sequence shape");
        }
        let attended_frames =
            backend.vae_sequence_to_nchw_frames_device(&attended, height, width)?;
        let projected_frames = backend.conv2d_prepared_device(
            &attended_frames,
            &prepared.projection.weights,
            self.projection.stride,
            self.projection.padding,
        )?;
        let projected = backend.nchw_frames_to_ncthw_device(&projected_frames, batch, time)?;
        backend.add_device(&projected, input)
    }
}

struct Upsample {
    temporal: Option<CausalConv3d>,
    spatial: Conv2dFrames,
}

#[cfg(feature = "vulkan")]
struct PreparedTemporalUpsample {
    temporal: Option<PreparedCausalConv3d>,
}

#[cfg(feature = "vulkan")]
struct PreparedUpsample {
    temporal: PreparedTemporalUpsample,
    spatial: PreparedConv2dFrames,
}

impl Upsample {
    fn load(weights: &SafeTensorFile, prefix: &str, temporal: bool) -> Result<Self> {
        Ok(Self {
            temporal: temporal
                .then(|| {
                    CausalConv3d::load(
                        weights,
                        &format!("{prefix}.time_conv"),
                        [1, 1, 1],
                        [1, 0, 0],
                    )
                })
                .transpose()?,
            spatial: Conv2dFrames::load(weights, &format!("{prefix}.resample.1"), [1, 1], [1, 1])?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        cache: &mut FeatureCache,
        chunk_index: usize,
    ) -> Result<Tensor> {
        let hidden = self.forward_temporal(input, cache, chunk_index)?;
        let hidden = upsample_spatial_nearest(&hidden, 2)?;
        self.spatial.forward(&hidden)
    }

    fn forward_temporal(
        &self,
        input: &Tensor,
        cache: &mut FeatureCache,
        chunk_index: usize,
    ) -> Result<Tensor> {
        let mut hidden = input.clone();
        if let Some(time_conv) = self.temporal.as_ref() {
            let cache_index = cache.take_index()?;
            if chunk_index > 0 {
                let previous = cache.slots[cache_index].clone();
                let mut next = last_time(&hidden, CACHE_TIME)?;
                if chunk_index >= 2 && next.shape()[2] < CACHE_TIME {
                    if let Some(previous) = previous.as_ref() {
                        next = concat_time(&[&last_time(previous, 1)?, &next])?;
                    }
                }
                if chunk_index == 1 && next.shape()[2] < CACHE_TIME {
                    next = prepend_zero_time(&next, CACHE_TIME - next.shape()[2])?;
                }
                hidden = if chunk_index == 1 {
                    time_conv.forward(&hidden, None)?
                } else {
                    time_conv.forward(&hidden, previous.as_ref())?
                };
                cache.slots[cache_index] = Some(next);
                hidden = channels_to_time(&hidden)?;
            }
        }
        Ok(hidden)
    }

    #[cfg(feature = "vulkan")]
    fn prepare_temporal(&self, backend: &dyn TensorBackend) -> Result<PreparedTemporalUpsample> {
        Ok(PreparedTemporalUpsample {
            temporal: self
                .temporal
                .as_ref()
                .map(|temporal| temporal.prepare(backend))
                .transpose()?,
        })
    }

    #[cfg(feature = "vulkan")]
    fn forward_temporal_with_backend(
        &self,
        input: &DeviceTensor,
        cache: &mut DeviceFeatureCache,
        chunk_index: usize,
        backend: &dyn TensorBackend,
        prepared: &PreparedTemporalUpsample,
    ) -> Result<DeviceTensor> {
        let mut hidden = input.clone();
        if let Some(time_conv) = self.temporal.as_ref() {
            let prepared_time_conv = prepared
                .temporal
                .as_ref()
                .context("prepared temporal upsample is missing time_conv weights")?;
            let cache_index = cache.take_index()?;
            if chunk_index > 0 {
                let previous = cache.temporal_prefix(cache_index, &hidden)?;
                let hidden_time = hidden.shape()[2];
                let next_start = hidden_time.saturating_sub(CACHE_TIME);
                let mut next = backend.ncthw_slice_time_device(
                    &hidden,
                    next_start,
                    hidden_time - next_start,
                )?;
                if chunk_index >= 2
                    && next.shape()[2] < CACHE_TIME
                    && let Some(previous) = previous.as_ref()
                {
                    let previous_time = previous.shape()[2];
                    let previous_tail =
                        backend.ncthw_slice_time_device(previous, previous_time - 1, 1)?;
                    next = backend.ncthw_concat_time_device(&[&previous_tail, &next])?;
                }
                if chunk_index == 1 && next.shape()[2] < CACHE_TIME {
                    next = backend
                        .ncthw_prepend_zero_time_device(&next, CACHE_TIME - next.shape()[2])?;
                }
                hidden = if chunk_index == 1 {
                    time_conv.forward_with_backend(&hidden, None, backend, prepared_time_conv)?
                } else {
                    time_conv.forward_with_backend(
                        &hidden,
                        previous.as_ref(),
                        backend,
                        prepared_time_conv,
                    )?
                };
                cache.replace(cache_index, next)?;
                hidden = backend.ncthw_channels_to_time_device(&hidden)?;
            }
        } else if prepared.temporal.is_some() {
            bail!("prepared temporal upsample contains unexpected time_conv weights");
        }
        Ok(hidden)
    }

    #[cfg(feature = "vulkan")]
    fn prepare(&self, backend: &dyn TensorBackend) -> Result<PreparedUpsample> {
        Ok(PreparedUpsample {
            temporal: self.prepare_temporal(backend)?,
            spatial: self.spatial.prepare(backend)?,
        })
    }

    #[cfg(feature = "vulkan")]
    fn forward_with_backend(
        &self,
        input: &DeviceTensor,
        cache: &mut DeviceFeatureCache,
        chunk_index: usize,
        backend: &dyn TensorBackend,
        prepared: &PreparedUpsample,
    ) -> Result<DeviceTensor> {
        let hidden = self.forward_temporal_with_backend(
            input,
            cache,
            chunk_index,
            backend,
            &prepared.temporal,
        )?;
        let hidden = backend.ncthw_upsample_nearest_device(&hidden, 2)?;
        self.spatial
            .forward_with_backend(&hidden, backend, &prepared.spatial)
    }
}

pub struct WanVae {
    pre: CausalConv3d,
    decoder_in: CausalConv3d,
    middle_0: ResidualBlock,
    middle_attention: SpatialAttention,
    middle_2: ResidualBlock,
    up_residuals: Vec<ResidualBlock>,
    upsample_0: Upsample,
    upsample_1: Upsample,
    upsample_2: Upsample,
    head_norm: Tensor,
    head: CausalConv3d,
}

#[cfg(feature = "vulkan")]
struct PreparedDecoderMiddle {
    pre: PreparedCausalConv3d,
    decoder_in: PreparedCausalConv3d,
    middle_0: PreparedResidualBlock,
    middle_attention: PreparedSpatialAttention,
    middle_2: PreparedResidualBlock,
}

#[cfg(feature = "vulkan")]
struct PreparedWanVae {
    middle: PreparedDecoderMiddle,
    up_residuals: Vec<PreparedResidualBlock>,
    upsamples: [PreparedUpsample; 3],
    head_norm: PreparedVectorHandle,
    head: PreparedCausalConv3d,
}

impl WanVae {
    pub fn load(weights: &SafeTensorFile) -> Result<Self> {
        let residual = |index| ResidualBlock::load(weights, &format!("decoder.upsamples.{index}"));
        let mut up_residuals = Vec::with_capacity(12);
        for index in [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14] {
            up_residuals.push(residual(index)?);
        }
        Ok(Self {
            pre: CausalConv3d::load(weights, "conv2", [1, 1, 1], [0, 0, 0])?,
            decoder_in: CausalConv3d::load(weights, "decoder.conv1", [1, 1, 1], [1, 1, 1])?,
            middle_0: ResidualBlock::load(weights, "decoder.middle.0")?,
            middle_attention: SpatialAttention::load(weights, "decoder.middle.1")?,
            middle_2: ResidualBlock::load(weights, "decoder.middle.2")?,
            up_residuals,
            upsample_0: Upsample::load(weights, "decoder.upsamples.3", true)?,
            upsample_1: Upsample::load(weights, "decoder.upsamples.7", true)?,
            upsample_2: Upsample::load(weights, "decoder.upsamples.11", false)?,
            head_norm: load_flat(weights, "decoder.head.0.gamma")?,
            head: CausalConv3d::load(weights, "decoder.head.2", [1, 1, 1], [1, 1, 1])?,
        })
    }

    #[cfg(feature = "vulkan")]
    fn prepare_decoder_middle(&self, backend: &dyn TensorBackend) -> Result<PreparedDecoderMiddle> {
        Ok(PreparedDecoderMiddle {
            pre: self.pre.prepare(backend)?,
            decoder_in: self.decoder_in.prepare(backend)?,
            middle_0: self.middle_0.prepare(backend)?,
            middle_attention: self.middle_attention.prepare(backend)?,
            middle_2: self.middle_2.prepare(backend)?,
        })
    }

    #[cfg(feature = "vulkan")]
    fn decoder_middle_with_backend(
        &self,
        latents: &DeviceTensor,
        cache: &mut DeviceFeatureCache,
        backend: &dyn TensorBackend,
        prepared: &PreparedDecoderMiddle,
    ) -> Result<DeviceTensor> {
        let transformed =
            self.pre
                .forward_uncached_with_backend(latents, backend, &prepared.pre)?;
        let hidden = cached_causal_conv_with_backend(
            &self.decoder_in,
            &transformed,
            cache,
            backend,
            &prepared.decoder_in,
        )?;
        let hidden =
            self.middle_0
                .forward_with_backend(&hidden, cache, backend, &prepared.middle_0)?;
        let hidden = self.middle_attention.forward_with_backend(
            &hidden,
            backend,
            &prepared.middle_attention,
        )?;
        self.middle_2
            .forward_with_backend(&hidden, cache, backend, &prepared.middle_2)
    }

    #[cfg(feature = "vulkan")]
    fn prepare_with_backend(&self, backend: &dyn TensorBackend) -> Result<PreparedWanVae> {
        Ok(PreparedWanVae {
            middle: self.prepare_decoder_middle(backend)?,
            up_residuals: self
                .up_residuals
                .iter()
                .map(|residual| residual.prepare(backend))
                .collect::<Result<Vec<_>>>()?,
            upsamples: [
                self.upsample_0.prepare(backend)?,
                self.upsample_1.prepare(backend)?,
                self.upsample_2.prepare(backend)?,
            ],
            head_norm: backend.prepare_vector(&self.head_norm)?,
            head: self.head.prepare(backend)?,
        })
    }

    #[cfg(feature = "vulkan")]
    fn forward_chunk_with_backend(
        &self,
        mut hidden: DeviceTensor,
        cache: &mut DeviceFeatureCache,
        chunk_index: usize,
        backend: &dyn TensorBackend,
        prepared: &PreparedWanVae,
    ) -> Result<DeviceTensor> {
        hidden = cached_causal_conv_with_backend(
            &self.decoder_in,
            &hidden,
            cache,
            backend,
            &prepared.middle.decoder_in,
        )?;
        hidden = self.middle_0.forward_with_backend(
            &hidden,
            cache,
            backend,
            &prepared.middle.middle_0,
        )?;
        hidden = self.middle_attention.forward_with_backend(
            &hidden,
            backend,
            &prepared.middle.middle_attention,
        )?;
        hidden = self.middle_2.forward_with_backend(
            &hidden,
            cache,
            backend,
            &prepared.middle.middle_2,
        )?;

        let mut residual_index = 0;
        for stage in 0..4 {
            for _ in 0..3 {
                hidden = self.up_residuals[residual_index].forward_with_backend(
                    &hidden,
                    cache,
                    backend,
                    &prepared.up_residuals[residual_index],
                )?;
                residual_index += 1;
            }
            hidden = match stage {
                0 => self.upsample_0.forward_with_backend(
                    &hidden,
                    cache,
                    chunk_index,
                    backend,
                    &prepared.upsamples[0],
                )?,
                1 => self.upsample_1.forward_with_backend(
                    &hidden,
                    cache,
                    chunk_index,
                    backend,
                    &prepared.upsamples[1],
                )?,
                2 => self.upsample_2.forward_with_backend(
                    &hidden,
                    cache,
                    chunk_index,
                    backend,
                    &prepared.upsamples[2],
                )?,
                3 => hidden,
                _ => unreachable!(),
            };
        }
        hidden = backend.channel_rms_norm_3d_device(&hidden, &prepared.head_norm, RMS_EPSILON)?;
        hidden = backend.silu_device(&hidden)?;
        cached_causal_conv_with_backend(&self.head, &hidden, cache, backend, &prepared.head)
    }

    #[cfg(feature = "vulkan")]
    fn decode_device_with_backend(
        &self,
        latents: &DeviceTensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedWanVae,
    ) -> Result<(DeviceTensor, DeviceFeatureCacheStats)> {
        let [batch, channels, time, _, _]: [usize; 5] = latents
            .shape()
            .try_into()
            .context("resident Wan VAE latents must be NCTHW")?;
        if batch != 1 || channels != 16 || time == 0 {
            bail!(
                "resident Wan 2.1 VAE latents must have shape [1,16,T,H,W], got {:?}",
                latents.shape()
            );
        }
        let transformed =
            self.pre
                .forward_uncached_with_backend(latents, backend, &prepared.middle.pre)?;
        let mut cache = DeviceFeatureCache::new();
        let mut chunks = Vec::with_capacity(time);
        for chunk_index in 0..time {
            cache.begin_chunk();
            let chunk = backend.ncthw_slice_time_device(&transformed, chunk_index, 1)?;
            let chunk =
                self.forward_chunk_with_backend(chunk, &mut cache, chunk_index, backend, prepared)?;
            if cache.active_index != USED_CACHE_SLOTS {
                bail!(
                    "resident Wan VAE chunk consumed {} cache slots, expected {USED_CACHE_SLOTS}",
                    cache.active_index
                );
            }
            chunks.push(chunk);
        }
        let chunk_refs = chunks.iter().collect::<Vec<_>>();
        let output = backend.ncthw_concat_time_device(&chunk_refs)?;
        let output = backend.affine_device(&output, 0.5, 0.5)?;
        let output = backend.clamp_device(&output, 0.0, 1.0)?;
        let cache_stats = cache.stats()?;
        cache.reset();
        Ok((output, cache_stats))
    }

    /// Decode VAE-space latents to clamped `[0,1]` RGB pixels.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.decode_with_backend(latents, &SCALAR_BACKEND)
    }

    /// Decode through an explicit tensor backend. Model ordering, chunking,
    /// and cache ownership remain identical to the scalar reference graph.
    pub(crate) fn decode_with_backend(
        &self,
        latents: &Tensor,
        backend: &dyn TensorBackend,
    ) -> Result<Tensor> {
        self.decode_internal(latents, backend, None)
    }

    /// Correctness harness that retains every major decoder activation. This
    /// is intentionally opt-in because full-size traces consume substantial
    /// memory.
    #[allow(dead_code)]
    pub fn decode_with_trace(&self, latents: &Tensor) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let mut trace = Vec::new();
        let output = self.decode_internal(latents, &SCALAR_BACKEND, Some(&mut trace))?;
        Ok((output, trace))
    }

    fn decode_internal(
        &self,
        latents: &Tensor,
        backend: &dyn TensorBackend,
        mut trace: Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Tensor> {
        let [batch, channels, time, _, _] = ncthw(latents)?;
        if batch != 1 || channels != 16 {
            bail!(
                "Wan 2.1 VAE latents must have shape [1,16,T,H,W], got {:?}",
                latents.shape()
            );
        }
        let transformed = self.pre.forward(latents, None)?;
        record(&mut trace, "decode.prelude", &transformed);
        let mut cache = FeatureCache::new();
        let mut chunks = Vec::with_capacity(time);
        for chunk_index in 0..time {
            cache.begin_chunk();
            let chunk = slice_time(&transformed, chunk_index, chunk_index + 1)?;
            let chunk = self.forward_chunk(chunk, &mut cache, chunk_index, backend, &mut trace)?;
            if cache.index != USED_CACHE_SLOTS {
                bail!(
                    "Wan VAE chunk consumed {} cache slots, expected {USED_CACHE_SLOTS}",
                    cache.index
                );
            }
            chunks.push(chunk);
        }
        let chunk_refs: Vec<&Tensor> = chunks.iter().collect();
        let mut output = concat_time(&chunk_refs)?;
        record(&mut trace, "decode.raw", &output);
        output.data_mut().par_iter_mut().for_each(|value| {
            *value = ((*value + 1.0) * 0.5).clamp(0.0, 1.0);
        });
        Ok(output)
    }

    fn forward_chunk(
        &self,
        mut hidden: Tensor,
        cache: &mut FeatureCache,
        chunk_index: usize,
        backend: &dyn TensorBackend,
        trace: &mut Option<&mut Vec<(String, Tensor)>>,
    ) -> Result<Tensor> {
        hidden = cached_causal_conv(&self.decoder_in, &hidden, cache)?;
        record(trace, &format!("chunk.{chunk_index}.decoder_in"), &hidden);
        hidden = self.middle_0.forward(&hidden, cache, backend)?;
        record(trace, &format!("chunk.{chunk_index}.middle.0"), &hidden);
        hidden = self.middle_attention.forward(&hidden, backend)?;
        record(trace, &format!("chunk.{chunk_index}.middle.1"), &hidden);
        hidden = self.middle_2.forward(&hidden, cache, backend)?;
        record(trace, &format!("chunk.{chunk_index}.middle.2"), &hidden);

        let mut residual_index = 0;
        for stage in 0..4 {
            for layer in 0..3 {
                hidden = self.up_residuals[residual_index].forward(&hidden, cache, backend)?;
                residual_index += 1;
                record(
                    trace,
                    &format!("chunk.{chunk_index}.up.{stage}.residual.{layer}"),
                    &hidden,
                );
            }
            hidden = match stage {
                0 => self.upsample_0.forward(&hidden, cache, chunk_index)?,
                1 => self.upsample_1.forward(&hidden, cache, chunk_index)?,
                2 => self.upsample_2.forward(&hidden, cache, chunk_index)?,
                3 => hidden,
                _ => unreachable!(),
            };
            record(
                trace,
                &format!("chunk.{chunk_index}.up.{stage}.output"),
                &hidden,
            );
        }
        hidden = hidden
            .channel_rms_norm_3d(&self.head_norm, RMS_EPSILON)?
            .silu();
        cached_causal_conv(&self.head, &hidden, cache)
    }
}

#[allow(dead_code)]
pub fn decode(weights: &SafeTensorFile, latents: &Tensor) -> Result<Tensor> {
    WanVae::load(weights)?.decode(latents)
}

fn load_flat(weights: &SafeTensorFile, name: &str) -> Result<Tensor> {
    let tensor = sd_ops::load_tensor(weights, name)?;
    Tensor::new(vec![tensor.len()], tensor.data().to_vec())
}

fn ncthw(tensor: &Tensor) -> Result<[usize; 5]> {
    tensor
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected rank-5 NCTHW tensor, got {:?}", tensor.shape()))
}

fn require_matching_nchw_axes(first: &Tensor, second: &Tensor, label: &str) -> Result<()> {
    let first = ncthw(first)?;
    let second = ncthw(second)?;
    if [first[0], first[1], first[3], first[4]] != [second[0], second[1], second[3], second[4]] {
        bail!("{label} shape mismatch: {first:?} vs {second:?}");
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn require_matching_device_nchw_axes(
    first: &DeviceTensor,
    second: &DeviceTensor,
    label: &str,
) -> Result<()> {
    let first: [usize; 5] = first
        .shape()
        .try_into()
        .with_context(|| format!("{label} first tensor must be NCTHW"))?;
    let second: [usize; 5] = second
        .shape()
        .try_into()
        .with_context(|| format!("{label} second tensor must be NCTHW"))?;
    if [first[0], first[1], first[3], first[4]] != [second[0], second[1], second[3], second[4]] {
        bail!("{label} shape mismatch: {first:?} vs {second:?}");
    }
    if first[2] == 0 || second[2] == 0 {
        bail!("{label} cannot contain a zero-length temporal axis");
    }
    Ok(())
}

fn slice_time(input: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let [batch, channels, time, height, width] = ncthw(input)?;
    if start >= end || end > time {
        bail!(
            "invalid temporal slice {start}..{end} for shape {:?}",
            input.shape()
        );
    }
    let output_time = end - start;
    let plane = height * width;
    let mut data = vec![0.0; batch * channels * output_time * plane];
    data.par_chunks_mut(output_time * plane)
        .enumerate()
        .for_each(|(sample_channel, destination)| {
            let source = &input.data()
                [(sample_channel * time + start) * plane..(sample_channel * time + end) * plane];
            destination.copy_from_slice(source);
        });
    Tensor::new(vec![batch, channels, output_time, height, width], data)
}

fn last_time(input: &Tensor, count: usize) -> Result<Tensor> {
    let time = ncthw(input)?[2];
    slice_time(input, time.saturating_sub(count), time)
}

fn concat_time(inputs: &[&Tensor]) -> Result<Tensor> {
    let first = inputs
        .first()
        .context("temporal concat requires at least one tensor")?;
    let [batch, channels, _, height, width] = ncthw(first)?;
    let mut total_time = 0usize;
    for input in inputs {
        require_matching_nchw_axes(first, input, "temporal concat")?;
        total_time = total_time
            .checked_add(input.shape()[2])
            .context("temporal concat length overflow")?;
    }
    let plane = height * width;
    let mut data = Vec::with_capacity(batch * channels * total_time * plane);
    for sample in 0..batch {
        for channel in 0..channels {
            for input in inputs {
                let input_time = input.shape()[2];
                let start = (sample * channels + channel) * input_time * plane;
                data.extend_from_slice(&input.data()[start..start + input_time * plane]);
            }
        }
    }
    Tensor::new(vec![batch, channels, total_time, height, width], data)
}

fn prepend_zero_time(input: &Tensor, count: usize) -> Result<Tensor> {
    if count == 0 {
        return Ok(input.clone());
    }
    let [batch, channels, _, height, width] = ncthw(input)?;
    let zeros = Tensor::zeros(vec![batch, channels, count, height, width])?;
    concat_time(&[&zeros, input])
}

fn channels_to_time(input: &Tensor) -> Result<Tensor> {
    let [batch, doubled_channels, time, height, width] = ncthw(input)?;
    if doubled_channels % 2 != 0 {
        bail!("temporal channel shuffle needs an even channel count");
    }
    let channels = doubled_channels / 2;
    let plane = height * width;
    let mut data = vec![0.0; input.len()];
    data.par_chunks_mut(plane)
        .enumerate()
        .for_each(|(destination_plane, destination)| {
            let output_time_index = destination_plane % (time * 2);
            let sample_channel = destination_plane / (time * 2);
            let sample = sample_channel / channels;
            let channel = sample_channel % channels;
            let input_time = output_time_index / 2;
            let half = output_time_index % 2;
            let input_channel = half * channels + channel;
            let source_plane = (sample * doubled_channels + input_channel) * time + input_time;
            destination
                .copy_from_slice(&input.data()[source_plane * plane..(source_plane + 1) * plane]);
        });
    Tensor::new(vec![batch, channels, time * 2, height, width], data)
}

fn upsample_spatial_nearest(input: &Tensor, scale: usize) -> Result<Tensor> {
    if scale == 0 {
        bail!("spatial upsample scale must be non-zero");
    }
    let [batch, channels, time, height, width] = ncthw(input)?;
    let output_height = height
        .checked_mul(scale)
        .context("spatial upsample height overflow")?;
    let output_width = width
        .checked_mul(scale)
        .context("spatial upsample width overflow")?;
    let input_plane = height * width;
    let output_plane = output_height * output_width;
    let mut data = vec![0.0; batch * channels * time * output_plane];
    data.par_chunks_mut(output_plane)
        .enumerate()
        .for_each(|(source_plane, output)| {
            let input = &input.data()[source_plane * input_plane..(source_plane + 1) * input_plane];
            for y in 0..output_height {
                for x in 0..output_width {
                    output[y * output_width + x] = input[(y / scale) * width + x / scale];
                }
            }
        });
    Tensor::new(
        vec![batch, channels, time, output_height, output_width],
        data,
    )
}

fn record(trace: &mut Option<&mut Vec<(String, Tensor)>>, name: &str, tensor: &Tensor) {
    if let Some(trace) = trace.as_deref_mut() {
        trace.push((name.to_owned(), tensor.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct CountingBackend {
        additions: AtomicUsize,
    }

    impl TensorBackend for CountingBackend {
        fn kind(&self) -> BackendKind {
            BackendKind::ScalarCpu
        }

        fn name(&self) -> &'static str {
            "counting-scalar"
        }

        fn add(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
            self.additions.fetch_add(1, Ordering::Relaxed);
            SCALAR_BACKEND.add(left, right)
        }

        fn multiply(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
            SCALAR_BACKEND.multiply(left, right)
        }

        fn scale(&self, input: &Tensor, value: f32) -> Result<Tensor> {
            SCALAR_BACKEND.scale(input, value)
        }

        fn silu(&self, input: &Tensor) -> Result<Tensor> {
            SCALAR_BACKEND.silu(input)
        }

        fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
            SCALAR_BACKEND.gelu_tanh(input)
        }

        fn clamp(&self, input: &Tensor, minimum: f32, maximum: f32) -> Result<Tensor> {
            SCALAR_BACKEND.clamp(input, minimum, maximum)
        }

        fn channel_rms_norm_3d(
            &self,
            input: &Tensor,
            weight: &Tensor,
            epsilon: f32,
        ) -> Result<Tensor> {
            SCALAR_BACKEND.channel_rms_norm_3d(input, weight, epsilon)
        }

        fn linear(&self, input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
            SCALAR_BACKEND.linear(input, weight, bias)
        }
    }

    fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    fn conv3d(weight: Tensor, bias: Option<Tensor>, padding: [usize; 3]) -> CausalConv3d {
        CausalConv3d {
            weight,
            bias,
            stride: [1, 1, 1],
            padding,
            dilation: [1, 1, 1],
        }
    }

    fn conv2d(weight: Tensor, bias: Option<Tensor>) -> Conv2dFrames {
        Conv2dFrames {
            weight,
            bias,
            stride: [1, 1],
            padding: [0, 0],
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: {actual} != {expected}"
            );
        }
    }

    #[cfg(feature = "vulkan")]
    fn trace_spatial_attention_device(
        attention: &SpatialAttention,
        input: &Tensor,
        backend: &dyn TensorBackend,
        prepared: &PreparedSpatialAttention,
    ) -> Result<Vec<(&'static str, Tensor)>> {
        let [batch, channels, time, height, width] = ncthw(input)?;
        let device_input = backend.upload_tensor(input)?;
        let normalized =
            backend.channel_rms_norm_3d_device(&device_input, &prepared.norm, RMS_EPSILON)?;
        let normalized_frames = backend.ncthw_to_nchw_frames_device(&normalized)?;
        let qkv_frames = backend.conv2d_prepared_device(
            &normalized_frames,
            &prepared.qkv.weights,
            attention.qkv.stride,
            attention.qkv.padding,
        )?;
        let (query, key, value) = backend.vae_qkv_to_sequences_device(&qkv_frames)?;
        let scores = backend.attention_scores_device(
            &query,
            &key,
            1,
            channels,
            1.0 / (channels as f32).sqrt(),
        )?;
        let probabilities = backend.softmax_device(&scores)?;
        let attended = backend.attention_values_device(&probabilities, &value, 1, channels)?;
        let attended_frames =
            backend.vae_sequence_to_nchw_frames_device(&attended, height, width)?;
        let projected_frames = backend.conv2d_prepared_device(
            &attended_frames,
            &prepared.projection.weights,
            attention.projection.stride,
            attention.projection.padding,
        )?;
        let projected = backend.nchw_frames_to_ncthw_device(&projected_frames, batch, time)?;
        let output = backend.add_device(&projected, &device_input)?;
        let resident = [
            ("normalized", &normalized),
            ("normalized_frames", &normalized_frames),
            ("qkv_frames", &qkv_frames),
            ("query", &query),
            ("key", &key),
            ("value", &value),
            ("scores", &scores),
            ("probabilities", &probabilities),
            ("attended", &attended),
            ("attended_frames", &attended_frames),
            ("projected_frames", &projected_frames),
            ("projected", &projected),
            ("output", &output),
        ];
        resident
            .into_iter()
            .map(|(name, tensor)| Ok((name, backend.download_tensor(tensor)?)))
            .collect()
    }

    #[test]
    fn feature_cache_matches_one_shot_causal_convolution() {
        let layer = conv3d(tensor(&[1, 1, 3, 1, 1], &[1., 10., 100.]), None, [1, 0, 0]);
        let full = tensor(&[1, 1, 3, 1, 1], &[1., 2., 3.]);
        let one_shot = layer.forward(&full, None).unwrap();
        let mut cache = FeatureCache::new();
        let mut chunks = Vec::new();
        for frame in 0..3 {
            cache.begin_chunk();
            chunks.push(
                cached_causal_conv(
                    &layer,
                    &slice_time(&full, frame, frame + 1).unwrap(),
                    &mut cache,
                )
                .unwrap(),
            );
        }
        let refs: Vec<&Tensor> = chunks.iter().collect();
        let incremental = concat_time(&refs).unwrap();
        assert_eq!(incremental, one_shot);
        assert_eq!(incremental.data(), &[100., 210., 321.]);
        assert_eq!(cache.slots[0].as_ref().unwrap().data(), &[2., 3.]);
    }

    #[test]
    fn residual_block_uses_learned_shortcut_when_channels_change() {
        let zero_conv = |input_channels, output_channels| {
            conv3d(
                Tensor::zeros(vec![output_channels, input_channels, 1, 1, 1]).unwrap(),
                Some(Tensor::zeros(vec![output_channels]).unwrap()),
                [0, 0, 0],
            )
        };
        let block = ResidualBlock {
            norm1: tensor(&[1], &[1.]),
            conv1: zero_conv(1, 2),
            norm2: tensor(&[2], &[1., 1.]),
            conv2: zero_conv(2, 2),
            shortcut: Some(conv3d(
                tensor(&[2, 1, 1, 1, 1], &[2., -1.]),
                Some(tensor(&[2], &[0.5, 1.])),
                [0, 0, 0],
            )),
        };
        let mut cache = FeatureCache::new();
        let backend = CountingBackend::default();
        let output = block
            .forward(&tensor(&[1, 1, 1, 1, 1], &[3.]), &mut cache, &backend)
            .unwrap();
        assert_eq!(output.shape(), &[1, 2, 1, 1, 1]);
        assert_eq!(output.data(), &[6.5, -2.]);
        assert_eq!(cache.index, 2);
        assert_eq!(backend.additions.load(Ordering::Relaxed), 1);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_residual_block_with_shortcut_and_two_cache_slots_matches_scalar() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping Vulkan residual-block parity: {error:#}");
                return;
            }
            Err(error) => panic!("required Vulkan residual-block parity failed: {error:#}"),
        };
        let conv1_weight = Tensor::new(
            vec![3, 2, 3, 3, 3],
            (0..162)
                .map(|index| ((index * 5) % 11) as f32 * 0.03125 - 0.15625)
                .collect(),
        )
        .unwrap();
        let conv2_weight = Tensor::new(
            vec![3, 3, 3, 3, 3],
            (0..243)
                .map(|index| ((index * 7) % 13) as f32 * 0.03125 - 0.1875)
                .collect(),
        )
        .unwrap();
        let block = ResidualBlock {
            norm1: tensor(&[2], &[1.0, 0.75]),
            conv1: conv3d(
                conv1_weight,
                Some(tensor(&[3], &[0.0625, -0.125, 0.1875])),
                [1, 1, 1],
            ),
            norm2: tensor(&[3], &[1.0, 0.875, 1.125]),
            conv2: conv3d(
                conv2_weight,
                Some(tensor(&[3], &[-0.0625, 0.125, -0.1875])),
                [1, 1, 1],
            ),
            shortcut: Some(conv3d(
                tensor(&[3, 2, 1, 1, 1], &[1.0, 0.0, 0.5, -0.5, -0.25, 0.75]),
                Some(tensor(&[3], &[0.125, -0.25, 0.375])),
                [0, 0, 0],
            )),
        };
        let input = Tensor::new(
            vec![1, 2, 3, 2, 2],
            (0..24)
                .map(|index| ((index * 11) % 29) as f32 * 0.125 - 1.5)
                .collect(),
        )
        .unwrap();
        let backend = &crate::backend::VULKAN_BACKEND;
        let prepared = block.prepare(backend).unwrap();
        let device_input = backend.upload_tensor(&input).unwrap();
        let mut scalar_cache = FeatureCache::new();
        let mut device_cache = DeviceFeatureCache::new();
        let mut expected_chunks = Vec::new();
        let mut actual_chunks = Vec::new();
        let started = std::time::Instant::now();
        for chunk_index in 0..3 {
            scalar_cache.begin_chunk();
            let scalar_input = slice_time(&input, chunk_index, chunk_index + 1).unwrap();
            expected_chunks.push(
                block
                    .forward(&scalar_input, &mut scalar_cache, &SCALAR_BACKEND)
                    .unwrap(),
            );

            device_cache.begin_chunk();
            let device_chunk = backend
                .ncthw_slice_time_device(&device_input, chunk_index, 1)
                .unwrap();
            actual_chunks.push(
                block
                    .forward_with_backend(&device_chunk, &mut device_cache, backend, &prepared)
                    .unwrap(),
            );
            assert_eq!(scalar_cache.index, 2);
            assert_eq!(device_cache.active_index, 2);
        }
        let expected_refs = expected_chunks.iter().collect::<Vec<_>>();
        let expected = concat_time(&expected_refs).unwrap();
        let actual_refs = actual_chunks.iter().collect::<Vec<_>>();
        let actual = backend.ncthw_concat_time_device(&actual_refs).unwrap();
        let output = backend.download_tensor(&actual).unwrap();
        let runtime = started.elapsed();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-5,
                maximum_mean_absolute_error: 2e-6,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, 3, 3, 2, 2]);

        let cache_stats = device_cache.stats().unwrap();
        assert_eq!(cache_stats.current_bytes, (2 + 3) * 2 * 2 * 2 * 4);
        assert_eq!(cache_stats.peak_bytes, (2 + 3) * 2 * 2 * 2 * 4);
        assert_eq!(cache_stats.occupied_slots, 2);
        assert_eq!(cache_stats.replaced_slots, 4);
        assert!(cache_stats.all_slots_resident);
        assert!(!cache_stats.all_slots_device_local);
        let after = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            5
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "Vulkan residual block: input={:?} output={:?} shortcut=true chunks=[1,1,1] cache_slots=2 cosine={:.9} max_abs={:.9} mean_abs={:.9} runtime_ms={:.3} current_vulkan_bytes={} cache_current_bytes={} cache_peak_bytes={} cache_replaced={} cache_device_local={} host_uploads={} weight_uploads={} downloads={}",
            input.shape(),
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            runtime.as_secs_f64() * 1_000.0,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            cache_stats.current_bytes,
            cache_stats.peak_bytes,
            cache_stats.replaced_slots,
            cache_stats.all_slots_device_local,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
        );

        device_cache.reset();
        drop(actual);
        drop(actual_chunks);
        drop(device_input);
        drop(prepared);
        drop(device_cache);
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
    #[ignore = "loads real Wan VAE weights; run explicitly for residual-block parity"]
    fn real_wan_vae_residual_block_matches_scalar() {
        use crate::{
            parity::{ParityTolerance, compare_tensors},
            safetensors::SafeTensorFile,
        };

        const VAE: &str =
            "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan_2.1_vae.safetensors";
        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = crate::vulkan::persistence_stats().unwrap();
        let weights = SafeTensorFile::open(VAE).unwrap();
        let block = ResidualBlock::load(&weights, "decoder.upsamples.12").unwrap();
        let input_channels = block.norm1.len();
        let output_channels = block.norm2.len();
        let input = Tensor::new(
            vec![1, input_channels, 1, 2, 3],
            (0..input_channels * 6)
                .map(|index| ((index * 17) as f32 * 0.013).sin() * 1.25)
                .collect(),
        )
        .unwrap();
        let mut scalar_cache = FeatureCache::new();
        scalar_cache.begin_chunk();
        let scalar_started = std::time::Instant::now();
        let expected = block
            .forward(&input, &mut scalar_cache, &SCALAR_BACKEND)
            .unwrap();
        let scalar_runtime = scalar_started.elapsed();

        let backend = &crate::backend::VULKAN_BACKEND;
        let prepared = block.prepare(backend).unwrap();
        let device_input = backend.upload_tensor(&input).unwrap();
        let mut device_cache = DeviceFeatureCache::new();
        device_cache.begin_chunk();
        let vulkan_started = std::time::Instant::now();
        let actual = block
            .forward_with_backend(&device_input, &mut device_cache, backend, &prepared)
            .unwrap();
        let output = backend.download_tensor(&actual).unwrap();
        let vulkan_runtime = vulkan_started.elapsed();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_99,
                maximum_absolute_error: 0.01,
                maximum_mean_absolute_error: 0.001,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, output_channels, 1, 2, 3]);
        assert_eq!(scalar_cache.index, 2);
        assert_eq!(device_cache.active_index, 2);
        let cache_stats = device_cache.stats().unwrap();
        assert_eq!(cache_stats.occupied_slots, 2);
        assert_eq!(
            cache_stats.current_bytes,
            (input_channels + output_channels) * 6 * 4
        );
        let after = crate::vulkan::persistence_stats().unwrap();
        let expected_weight_uploads = if block.shortcut.is_some() { 5 } else { 4 };
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            expected_weight_uploads
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "real Wan VAE residual block decoder.upsamples.12: input={:?} output={:?} shortcut={} scalar_ms={:.3} vulkan_ms={:.3} cosine={:.9} max_abs={:.9} mean_abs={:.9} current_vulkan_bytes={} cache_current_bytes={} cache_peak_bytes={} host_uploads={} weight_uploads={} downloads={}",
            input.shape(),
            output.shape(),
            block.shortcut.is_some(),
            scalar_runtime.as_secs_f64() * 1_000.0,
            vulkan_runtime.as_secs_f64() * 1_000.0,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            cache_stats.current_bytes,
            cache_stats.peak_bytes,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
        );

        device_cache.reset();
        drop(actual);
        drop(device_input);
        drop(prepared);
        drop(device_cache);
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

    #[test]
    fn temporal_upsample_covers_first_second_and_later_chunk_branches() {
        let mut time_weight = vec![0.0; 4 * 2 * 3];
        for half in 0..2 {
            for channel in 0..2 {
                let output_channel = half * 2 + channel;
                time_weight[(output_channel * 2 + channel) * 3 + 2] = 1.0;
            }
        }
        let upsample = Upsample {
            temporal: Some(conv3d(
                tensor(&[4, 2, 3, 1, 1], &time_weight),
                Some(Tensor::zeros(vec![4]).unwrap()),
                [1, 0, 0],
            )),
            spatial: conv2d(
                tensor(&[2, 2, 1, 1], &[1., 0., 0., 1.]),
                Some(Tensor::zeros(vec![2]).unwrap()),
            ),
        };
        let mut cache = FeatureCache::new();

        cache.begin_chunk();
        let first = upsample
            .forward(&tensor(&[1, 2, 1, 1, 1], &[1., 10.]), &mut cache, 0)
            .unwrap();
        assert_eq!(first.shape(), &[1, 2, 1, 2, 2]);
        assert!(cache.slots[0].is_none());

        cache.begin_chunk();
        let second = upsample
            .forward(&tensor(&[1, 2, 1, 1, 1], &[2., 20.]), &mut cache, 1)
            .unwrap();
        assert_eq!(second.shape(), &[1, 2, 2, 2, 2]);
        assert_eq!(cache.slots[0].as_ref().unwrap().shape()[2], 2);
        assert_eq!(cache.slots[0].as_ref().unwrap().data(), &[0., 2., 0., 20.]);

        cache.begin_chunk();
        let later = upsample
            .forward(&tensor(&[1, 2, 1, 1, 1], &[3., 30.]), &mut cache, 2)
            .unwrap();
        assert_eq!(later.shape(), &[1, 2, 2, 2, 2]);
        assert_eq!(cache.slots[0].as_ref().unwrap().data(), &[2., 3., 20., 30.]);
        assert!(later.data()[..8].iter().all(|&value| value == 3.0));
        assert!(later.data()[8..].iter().all(|&value| value == 30.0));
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_temporal_upsample_branches_match_scalar_and_cache_state() {
        use crate::parity::compare_tensors;

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping Vulkan temporal-upsample branch parity: {error:#}");
                return;
            }
            Err(error) => panic!("required Vulkan temporal-upsample parity failed: {error:#}"),
        };
        let mut time_weight = vec![0.0; 4 * 2 * 3];
        for channel_half in 0..2 {
            for channel in 0..2 {
                let output_channel = channel_half * 2 + channel;
                time_weight[(output_channel * 2 + channel) * 3 + 2] = 1.0;
            }
        }
        let upsample = Upsample {
            temporal: Some(conv3d(
                tensor(&[4, 2, 3, 1, 1], &time_weight),
                Some(Tensor::zeros(vec![4]).unwrap()),
                [1, 0, 0],
            )),
            spatial: conv2d(
                tensor(&[2, 2, 1, 1], &[1.0, 0.0, 0.0, 1.0]),
                Some(Tensor::zeros(vec![2]).unwrap()),
            ),
        };
        let input = tensor(
            &[1, 2, 3, 1, 2],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            ],
        );
        let backend = &crate::backend::VULKAN_BACKEND;
        let prepared = upsample.prepare_temporal(backend).unwrap();
        let device_input = backend.upload_tensor(&input).unwrap();
        let mut scalar_cache = FeatureCache::new();
        let mut device_cache = DeviceFeatureCache::new();
        let mut resident_outputs = Vec::new();
        let started = std::time::Instant::now();

        for chunk_index in 0..3 {
            let scalar_input = slice_time(&input, chunk_index, chunk_index + 1).unwrap();
            scalar_cache.begin_chunk();
            let expected = upsample
                .forward_temporal(&scalar_input, &mut scalar_cache, chunk_index)
                .unwrap();

            device_cache.begin_chunk();
            let device_chunk = backend
                .ncthw_slice_time_device(&device_input, chunk_index, 1)
                .unwrap();
            let actual = upsample
                .forward_temporal_with_backend(
                    &device_chunk,
                    &mut device_cache,
                    chunk_index,
                    backend,
                    &prepared,
                )
                .unwrap();
            let actual_host = backend.download_tensor(&actual).unwrap();
            let metrics = compare_tensors(&actual_host, &expected).unwrap();
            assert_eq!(actual_host, expected);
            assert_eq!(device_cache.active_index, 1);
            assert_eq!(scalar_cache.index, 1);
            if chunk_index == 0 {
                assert_eq!(actual_host.shape(), &[1, 2, 1, 1, 2]);
                assert!(scalar_cache.slots[0].is_none());
                assert!(device_cache.slots[0].is_none());
            } else {
                assert_eq!(actual_host.shape(), &[1, 2, 2, 1, 2]);
                let expected_cache = scalar_cache.slots[0].as_ref().unwrap();
                let actual_cache = backend
                    .download_tensor(device_cache.slots[0].as_ref().unwrap())
                    .unwrap();
                assert_eq!(&actual_cache, expected_cache);
            }
            println!(
                "temporal upsample chunk_idx={chunk_index}: input={:?} output={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                scalar_input.shape(),
                actual_host.shape(),
                metrics.cosine_similarity,
                metrics.maximum_absolute_error,
                metrics.mean_absolute_error,
            );
            resident_outputs.push(actual);
        }

        let runtime = started.elapsed();
        let cache_stats = device_cache.stats().unwrap();
        assert_eq!(cache_stats.current_bytes, 1 * 2 * 2 * 1 * 2 * 4);
        assert_eq!(cache_stats.peak_bytes, 1 * 2 * 2 * 1 * 2 * 4);
        assert_eq!(cache_stats.occupied_slots, 1);
        assert_eq!(cache_stats.replaced_slots, 1);
        assert!(cache_stats.all_slots_resident);
        assert!(!cache_stats.all_slots_device_local);
        let after = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            1
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 5);
        println!(
            "temporal upsample branches: input={:?} chunk_outputs=[1,2,2] runtime_ms={:.3} current_vulkan_bytes={} cache_current_bytes={} cache_peak_bytes={} cache_replaced={} cache_device_local={} host_uploads={} weight_uploads={} downloads={}",
            input.shape(),
            runtime.as_secs_f64() * 1_000.0,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            cache_stats.current_bytes,
            cache_stats.peak_bytes,
            cache_stats.replaced_slots,
            cache_stats.all_slots_device_local,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
        );

        device_cache.reset();
        drop(resident_outputs);
        drop(device_input);
        drop(prepared);
        drop(device_cache);
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

    #[test]
    fn spatial_attention_is_independent_per_frame_and_attends_over_pixels() {
        let mut qkv_weight = vec![0.0; 6 * 2];
        qkv_weight[4 * 2] = 1.0;
        qkv_weight[(5 * 2) + 1] = 1.0;
        let attention = SpatialAttention {
            norm: tensor(&[2], &[1., 1.]),
            qkv: conv2d(
                tensor(&[6, 2, 1, 1], &qkv_weight),
                Some(Tensor::zeros(vec![6]).unwrap()),
            ),
            projection: conv2d(
                tensor(&[2, 2, 1, 1], &[1., 0., 0., 1.]),
                Some(Tensor::zeros(vec![2]).unwrap()),
            ),
        };
        let input = tensor(&[1, 2, 1, 1, 2], &[1., 3., 2., 4.]);
        let output = attention.forward(&input, &SCALAR_BACKEND).unwrap();
        assert_close(
            output.data(),
            &[1.7404919, 3.7404919, 3.198141, 5.198141],
            2e-6,
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_spatial_attention_matches_scalar_and_isolates_frames() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping Vulkan spatial-attention parity: {error:#}");
                return;
            }
            Err(error) => panic!("required Vulkan spatial-attention parity failed: {error:#}"),
        };
        let attention = SpatialAttention {
            norm: tensor(&[2], &[1.0, 0.875]),
            qkv: conv2d(
                tensor(
                    &[6, 2, 1, 1],
                    &[
                        1.0, 0.0, 0.0, 1.0, // query
                        0.5, -0.25, 0.25, 0.75, // key
                        1.0, 0.5, -0.5, 1.0, // value
                    ],
                ),
                Some(tensor(&[6], &[0.0625, -0.125, 0.0, 0.125, -0.0625, 0.1875])),
            ),
            projection: conv2d(
                tensor(&[2, 2, 1, 1], &[0.75, -0.25, 0.5, 1.0]),
                Some(tensor(&[2], &[0.125, -0.0625])),
            ),
        };
        let input = Tensor::new(
            vec![2, 2, 2, 2, 3],
            (0..48)
                .map(|index| ((index * 13) % 31) as f32 * 0.125 - 1.75)
                .collect(),
        )
        .unwrap();
        let expected = attention.forward(&input, &SCALAR_BACKEND).unwrap();
        let scalar_prepared = attention.prepare(&SCALAR_BACKEND).unwrap();
        let scalar_trace =
            trace_spatial_attention_device(&attention, &input, &SCALAR_BACKEND, &scalar_prepared)
                .unwrap();
        let scalar_device_result = &scalar_trace.last().unwrap().1;
        let scalar_graph_metrics = compare_tensors(&scalar_device_result, &expected).unwrap();
        println!(
            "scalar prepared VAE spatial attention: cosine={:.9} max_abs={:.9} mean_abs={:.9}",
            scalar_graph_metrics.cosine_similarity,
            scalar_graph_metrics.maximum_absolute_error,
            scalar_graph_metrics.mean_absolute_error,
        );
        let backend = &crate::backend::VULKAN_BACKEND;
        let prepared = attention.prepare(backend).unwrap();
        let started = std::time::Instant::now();
        let device_trace =
            trace_spatial_attention_device(&attention, &input, backend, &prepared).unwrap();
        let runtime = started.elapsed();
        for ((scalar_name, scalar), (device_name, device)) in scalar_trace.iter().zip(&device_trace)
        {
            assert_eq!(scalar_name, device_name);
            let layer_metrics = compare_tensors(device, scalar).unwrap();
            println!(
                "VAE spatial layer {device_name}: shape={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9}",
                device.shape(),
                layer_metrics.cosine_similarity,
                layer_metrics.maximum_absolute_error,
                layer_metrics.mean_absolute_error,
            );
        }
        let output = &device_trace.last().unwrap().1;
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-5,
                maximum_mean_absolute_error: 2e-6,
            })
            .unwrap();
        assert_eq!(output.shape(), &[2, 2, 2, 2, 3]);

        let mut changed_input = input.clone();
        let [batch, channels, time, height, width] = [2, 2, 2, 2, 3];
        let plane = height * width;
        for sample in 0..batch {
            for channel in 0..channels {
                let start = ((sample * channels + channel) * time + 1) * plane;
                for value in &mut changed_input.data_mut()[start..start + plane] {
                    *value = *value * -3.0 + 7.0;
                }
            }
        }
        let changed_device_input = backend.upload_tensor(&changed_input).unwrap();
        let changed_device_output = attention
            .forward_with_backend(&changed_device_input, backend, &prepared)
            .unwrap();
        let changed_output = backend.download_tensor(&changed_device_output).unwrap();
        let mut changed_frame_difference = 0.0f32;
        for sample in 0..batch {
            for channel in 0..channels {
                let first_frame = (sample * channels + channel) * time * plane;
                assert_eq!(
                    &output.data()[first_frame..first_frame + plane],
                    &changed_output.data()[first_frame..first_frame + plane]
                );
                let second_frame = first_frame + plane;
                for position in 0..plane {
                    changed_frame_difference += (output.data()[second_frame + position]
                        - changed_output.data()[second_frame + position])
                        .abs();
                }
            }
        }
        assert!(changed_frame_difference > 1.0);
        let after = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            3
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            2
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 14);
        println!(
            "Vulkan VAE spatial attention: input={:?} qkv={:?} output={:?} frames={} positions={} cosine={:.9} max_abs={:.9} mean_abs={:.9} runtime_ms={:.3} changed_frame_delta={:.6} current_vulkan_bytes={} device_local_bytes={} host_uploads={} weight_uploads={} downloads={}",
            input.shape(),
            attention.qkv.weight.shape(),
            output.shape(),
            batch * time,
            plane,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            runtime.as_secs_f64() * 1_000.0,
            changed_frame_difference,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            after.resident_device_local_bytes - before.resident_device_local_bytes,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
        );

        drop(changed_device_output);
        drop(changed_device_input);
        drop(prepared);
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
    #[ignore = "loads real Wan VAE weights; run explicitly for spatial-attention parity"]
    fn real_wan_vae_spatial_attention_matches_scalar() {
        use crate::parity::{ParityTolerance, compare_tensors};

        const VAE: &str =
            "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan_2.1_vae.safetensors";
        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = crate::vulkan::persistence_stats().unwrap();
        let weights = SafeTensorFile::open(VAE).unwrap();
        let attention = SpatialAttention::load(&weights, "decoder.middle.1").unwrap();
        let channels = attention.norm.len();
        let input = Tensor::new(
            vec![1, channels, 1, 2, 3],
            (0..channels * 6)
                .map(|index| ((index * 19) as f32 * 0.011).sin() * 1.5)
                .collect(),
        )
        .unwrap();
        let scalar_started = std::time::Instant::now();
        let expected = attention.forward(&input, &SCALAR_BACKEND).unwrap();
        let scalar_runtime = scalar_started.elapsed();

        let backend = &crate::backend::VULKAN_BACKEND;
        let prepared = attention.prepare(backend).unwrap();
        let device_input = backend.upload_tensor(&input).unwrap();
        let vulkan_started = std::time::Instant::now();
        let actual = attention
            .forward_with_backend(&device_input, backend, &prepared)
            .unwrap();
        let output = backend.download_tensor(&actual).unwrap();
        let vulkan_runtime = vulkan_started.elapsed();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_99,
                maximum_absolute_error: 0.01,
                maximum_mean_absolute_error: 0.001,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, channels, 1, 2, 3]);
        let after = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            3
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "real Wan VAE spatial attention decoder.middle.1: input={:?} output={:?} scalar_ms={:.3} vulkan_ms={:.3} cosine={:.9} max_abs={:.9} mean_abs={:.9} current_vulkan_bytes={} device_local_bytes={} host_uploads={} weight_uploads={} downloads={}",
            input.shape(),
            output.shape(),
            scalar_runtime.as_secs_f64() * 1_000.0,
            vulkan_runtime.as_secs_f64() * 1_000.0,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            after.resident_device_local_bytes - before.resident_device_local_bytes,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
        );

        drop(actual);
        drop(device_input);
        drop(prepared);
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
    #[ignore = "loads real Wan VAE weights; run explicitly for decoder-middle parity"]
    fn real_wan_vae_decoder_middle_matches_scalar() {
        use crate::parity::{ParityTolerance, compare_tensors};

        const VAE: &str =
            "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan_2.1_vae.safetensors";
        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = crate::vulkan::persistence_stats().unwrap();
        let weights = SafeTensorFile::open(VAE).unwrap();
        let vae = WanVae::load(&weights).unwrap();
        let input = Tensor::new(
            vec![1, 16, 1, 2, 3],
            (0..96)
                .map(|index| ((index * 23) as f32 * 0.017).sin() * 1.25)
                .collect(),
        )
        .unwrap();
        let scalar_started = std::time::Instant::now();
        let transformed = vae.pre.forward(&input, None).unwrap();
        let mut scalar_cache = FeatureCache::new();
        scalar_cache.begin_chunk();
        let hidden = cached_causal_conv(&vae.decoder_in, &transformed, &mut scalar_cache).unwrap();
        let hidden = vae
            .middle_0
            .forward(&hidden, &mut scalar_cache, &SCALAR_BACKEND)
            .unwrap();
        let hidden = vae
            .middle_attention
            .forward(&hidden, &SCALAR_BACKEND)
            .unwrap();
        let expected = vae
            .middle_2
            .forward(&hidden, &mut scalar_cache, &SCALAR_BACKEND)
            .unwrap();
        let scalar_runtime = scalar_started.elapsed();
        assert_eq!(scalar_cache.index, 5);

        let backend = &crate::backend::VULKAN_BACKEND;
        let prepared = vae.prepare_decoder_middle(backend).unwrap();
        let device_input = backend.upload_tensor(&input).unwrap();
        let mut device_cache = DeviceFeatureCache::new();
        device_cache.begin_chunk();
        let vulkan_started = std::time::Instant::now();
        let actual = vae
            .decoder_middle_with_backend(&device_input, &mut device_cache, backend, &prepared)
            .unwrap();
        let output = backend.download_tensor(&actual).unwrap();
        let vulkan_runtime = vulkan_started.elapsed();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_99,
                maximum_absolute_error: 0.01,
                maximum_mean_absolute_error: 0.001,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, 384, 1, 2, 3]);
        assert_eq!(device_cache.active_index, 5);
        let cache_stats = device_cache.stats().unwrap();
        assert_eq!(cache_stats.occupied_slots, 5);
        assert_eq!(cache_stats.replaced_slots, 0);
        let expected_weight_uploads = 13
            + usize::from(vae.middle_0.shortcut.is_some())
            + usize::from(vae.middle_2.shortcut.is_some());
        let after = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            expected_weight_uploads as u64
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "real Wan VAE decoder middle: input={:?} output={:?} cache_slots={} scalar_ms={:.3} vulkan_ms={:.3} cosine={:.9} max_abs={:.9} mean_abs={:.9} current_vulkan_bytes={} device_local_bytes={} cache_current_bytes={} cache_peak_bytes={} host_uploads={} weight_uploads={} downloads={}",
            input.shape(),
            output.shape(),
            cache_stats.occupied_slots,
            scalar_runtime.as_secs_f64() * 1_000.0,
            vulkan_runtime.as_secs_f64() * 1_000.0,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            after.resident_device_local_bytes - before.resident_device_local_bytes,
            cache_stats.current_bytes,
            cache_stats.peak_bytes,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
        );

        device_cache.reset();
        drop(actual);
        drop(device_input);
        drop(prepared);
        drop(device_cache);
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
    fn device_feature_cache_owns_exactly_32_slots_and_resets_cleanly() {
        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping device feature-cache ownership test: {error:#}");
                return;
            }
            Err(error) => panic!("required device feature-cache test failed: {error:#}"),
        };
        let backend = &crate::backend::VULKAN_BACKEND;
        let source = tensor(&[1, 1, 2, 1, 1], &[1.0, 2.0]);
        let device_source = backend.upload_tensor(&source).unwrap();
        let mismatched = backend
            .upload_tensor(&Tensor::zeros(vec![1, 2, 1, 1, 1]).unwrap())
            .unwrap();
        let scalar_source = SCALAR_BACKEND.upload_tensor(&source).unwrap();
        let mut cache = DeviceFeatureCache::new();
        assert_eq!(cache.slots.len(), 32);
        cache.begin_chunk();
        for expected_index in 0..32 {
            let index = cache.take_index().unwrap();
            assert_eq!(index, expected_index);
            let slot_tensor = backend
                .ncthw_slice_time_device(&device_source, expected_index % 2, 1)
                .unwrap();
            cache.replace(index, slot_tensor).unwrap();
        }
        assert!(cache.take_index().is_err());
        let full_stats = cache.stats().unwrap();
        assert_eq!(
            full_stats,
            DeviceFeatureCacheStats {
                current_bytes: 32 * std::mem::size_of::<f32>(),
                peak_bytes: 32 * std::mem::size_of::<f32>(),
                occupied_slots: 32,
                replaced_slots: 0,
                evicted_slots: 0,
                all_slots_resident: true,
                all_slots_device_local: false,
            }
        );
        assert!(cache.temporal_prefix(0, &mismatched).is_err());
        assert!(cache.replace(0, scalar_source).is_err());

        cache.replace(0, device_source.clone()).unwrap();
        let replaced_stats = cache.stats().unwrap();
        assert_eq!(
            replaced_stats.current_bytes,
            33 * std::mem::size_of::<f32>()
        );
        assert_eq!(replaced_stats.peak_bytes, 33 * std::mem::size_of::<f32>());
        assert_eq!(replaced_stats.occupied_slots, 32);
        assert_eq!(replaced_stats.replaced_slots, 1);
        assert!(replaced_stats.all_slots_resident);
        assert!(!replaced_stats.all_slots_device_local);

        cache.reset();
        let reset_stats = cache.stats().unwrap();
        assert_eq!(reset_stats.current_bytes, 0);
        assert_eq!(reset_stats.peak_bytes, 33 * std::mem::size_of::<f32>());
        assert_eq!(reset_stats.occupied_slots, 0);
        assert_eq!(reset_stats.replaced_slots, 1);
        assert_eq!(reset_stats.evicted_slots, 32);
        assert!(!reset_stats.all_slots_resident);
        assert!(!reset_stats.all_slots_device_local);
        assert_eq!(cache.active_index, 0);
        assert!(cache.slots.iter().all(Option::is_none));

        println!(
            "DeviceFeatureCache: slots={} current_bytes={} peak_bytes={} occupied={} replaced={} evicted={} resident={} device_local={}",
            cache.slots.len(),
            reset_stats.current_bytes,
            reset_stats.peak_bytes,
            reset_stats.occupied_slots,
            reset_stats.replaced_slots,
            reset_stats.evicted_slots,
            reset_stats.all_slots_resident,
            reset_stats.all_slots_device_local,
        );

        drop(cache);
        drop(mismatched);
        drop(device_source);
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
    fn cached_vulkan_causal_conv3d_three_chunks_matches_scalar_one_shot() {
        use crate::parity::{ParityTolerance, compare_tensors};

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let before = match crate::vulkan::persistence_stats() {
            Ok(stats) => stats,
            Err(error) if std::env::var_os("QUARTZ_REQUIRE_VULKAN").is_none() => {
                eprintln!("skipping cached Vulkan causal Conv3D parity: {error:#}");
                return;
            }
            Err(error) => panic!("required cached Vulkan causal Conv3D parity failed: {error:#}"),
        };
        let backend = &crate::backend::VULKAN_BACKEND;
        let input = Tensor::new(
            vec![1, 2, 5, 2, 3],
            (0..60)
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
        let bias = tensor(&[3], &[0.125, -0.25, 0.375]);
        let layer = conv3d(weight, Some(bias), [1, 1, 1]);
        assert_eq!(layer.stride, [1, 1, 1]);
        assert_eq!(layer.dilation, [1, 1, 1]);
        let expected = layer.forward(&input, None).unwrap();

        let prepared = layer.prepare(backend).unwrap();
        let device_input = backend.upload_tensor(&input).unwrap();
        let mut cache = DeviceFeatureCache::new();
        let chunks = [(0, 1), (1, 2), (3, 2)];
        let started = std::time::Instant::now();
        let mut outputs = Vec::new();
        for (start, count) in chunks {
            cache.begin_chunk();
            assert_eq!(cache.active_index, 0);
            let chunk = backend
                .ncthw_slice_time_device(&device_input, start, count)
                .unwrap();
            let output =
                cached_causal_conv_with_backend(&layer, &chunk, &mut cache, backend, &prepared)
                    .unwrap();
            assert_eq!(cache.active_index, 1);
            assert_eq!(output.shape(), &[1, 3, count, 2, 3]);
            outputs.push(output);
        }
        let output_refs = outputs.iter().collect::<Vec<_>>();
        let incremental = backend.ncthw_concat_time_device(&output_refs).unwrap();
        let output = backend.download_tensor(&incremental).unwrap();
        let runtime = started.elapsed();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999_999,
                maximum_absolute_error: 2e-5,
                maximum_mean_absolute_error: 2e-6,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, 3, 5, 2, 3]);

        let cache_stats = cache.stats().unwrap();
        assert_eq!(cache_stats.current_bytes, 1 * 2 * 2 * 2 * 3 * 4);
        assert_eq!(cache_stats.peak_bytes, 1 * 2 * 2 * 2 * 3 * 4);
        assert_eq!(cache_stats.occupied_slots, 1);
        assert_eq!(cache_stats.replaced_slots, 2);
        assert_eq!(cache_stats.evicted_slots, 0);
        assert!(cache_stats.all_slots_resident);
        assert!(!cache_stats.all_slots_device_local);
        let after = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after.resident_weight_uploads - before.resident_weight_uploads,
            1
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "cached causal Conv3D: input={:?} weight={:?} output={:?} padding_before={:?} padding_after={:?} stride={:?} chunks={chunks:?} cosine={:.9} max_abs={:.9} mean_abs={:.9} runtime_ms={:.3} current_vulkan_bytes={} cache_current_bytes={} cache_peak_bytes={} cache_occupied={} cache_replaced={} cache_device_local={} host_uploads={} weight_uploads={} downloads={}",
            input.shape(),
            layer.weight.shape(),
            output.shape(),
            [2, 1, 1],
            [0, 1, 1],
            layer.stride,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            runtime.as_secs_f64() * 1_000.0,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            cache_stats.current_bytes,
            cache_stats.peak_bytes,
            cache_stats.occupied_slots,
            cache_stats.replaced_slots,
            cache_stats.all_slots_device_local,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
        );

        cache.reset();
        let reset_stats = cache.stats().unwrap();
        assert_eq!(reset_stats.current_bytes, 0);
        assert_eq!(reset_stats.occupied_slots, 0);
        assert_eq!(reset_stats.evicted_slots, 1);
        assert!(cache.slots.iter().all(Option::is_none));

        drop(incremental);
        drop(outputs);
        drop(device_input);
        drop(prepared);
        drop(cache);
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

    #[test]
    fn temporal_channel_shuffle_maps_channel_halves_to_adjacent_frames() {
        let input = tensor(
            &[1, 4, 2, 1, 1],
            &[1., 2., 10., 20., 100., 200., 1000., 2000.],
        );
        let output = channels_to_time(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 4, 1, 1]);
        assert_eq!(output.data(), &[1., 100., 2., 200., 10., 1000., 20., 2000.]);
    }
}

#[cfg(test)]
mod parity {
    use super::*;
    use std::path::Path;

    const VAE: &str =
        "/home/tiny/projects/saient/models/wan2.1-t2v-1.3b-mobile-pack/wan_2.1_vae.safetensors";
    const REFERENCE: &str = "/home/tiny/projects/tinyq4/reference/vae";

    fn read_dump(path: &Path) -> Option<(Vec<i64>, Vec<f32>)> {
        let bytes = std::fs::read(path).ok()?;
        if bytes.get(..4)? != b"SQD1" {
            return None;
        }
        let dimensions = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?) as usize;
        let shape = (0..dimensions)
            .map(|index| {
                i64::from_le_bytes(bytes[8 + index * 8..16 + index * 8].try_into().unwrap())
            })
            .collect();
        let data = bytes[8 + dimensions * 8..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        Some((shape, data))
    }

    #[cfg(feature = "vulkan")]
    fn current_rss_kib() -> Option<u64> {
        std::fs::read_to_string("/proc/self/status")
            .ok()?
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmRSS:")?
                    .split_whitespace()
                    .next()?
                    .parse()
                    .ok()
            })
    }

    fn run_case(
        label: &str,
        input_name: &str,
        output_name: &str,
        input_shape: [i64; 5],
        tensor_shape: [usize; 5],
        output_shape: [i64; 5],
        expected_tensor_shape: [usize; 5],
    ) {
        let vae_path = Path::new(VAE);
        let input_path = Path::new(REFERENCE).join(input_name);
        let output_path = Path::new(REFERENCE).join(output_name);
        for path in [vae_path, input_path.as_path(), output_path.as_path()] {
            assert!(path.exists(), "required parity input is missing: {path:?}");
        }
        let (captured_input_shape, input) = read_dump(&input_path).unwrap();
        let (captured_output_shape, expected) = read_dump(&output_path).unwrap();
        assert_eq!(captured_input_shape, input_shape);
        assert_eq!(captured_output_shape, output_shape);
        let weights = SafeTensorFile::open(vae_path).unwrap();
        let decoder = WanVae::load(&weights).unwrap();
        let input = Tensor::new(tensor_shape.to_vec(), input).unwrap();
        let start = std::time::Instant::now();
        let actual = decoder.decode(&input).unwrap();
        eprintln!("{label} Wan VAE decode took {:?}", start.elapsed());
        assert_eq!(actual.shape(), expected_tensor_shape);
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut expected_norm = 0.0f64;
        let mut maximum_error = 0.0f32;
        let mut mean_error = 0.0f64;
        for (&actual, &expected) in actual.data().iter().zip(&expected) {
            let error = (actual - expected).abs();
            maximum_error = maximum_error.max(error);
            mean_error += error as f64;
            dot += actual as f64 * expected as f64;
            actual_norm += actual as f64 * actual as f64;
            expected_norm += expected as f64 * expected as f64;
        }
        mean_error /= expected.len() as f64;
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        eprintln!(
            "{label} VAE cosine={cosine:.9} max_abs={maximum_error:.7} mean_abs={mean_error:.7}"
        );
        assert!(
            cosine > 0.999,
            "{label} VAE cosine {cosine:.9} is below parity threshold"
        );
        assert!(
            maximum_error < 0.03,
            "{label} VAE maximum error {maximum_error:.7} exceeds tolerance"
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads the Wan VAE and validates the first real resident causal Conv3D"]
    fn resident_vulkan_prelude_matches_scalar() {
        use std::time::Instant;

        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let vae_path = Path::new(VAE);
        let input_path = Path::new(REFERENCE).join("vae_in_small.bin");
        assert!(vae_path.exists());
        assert!(input_path.exists());
        let (captured_shape, input_values) = read_dump(&input_path).unwrap();
        assert_eq!(captured_shape, [8, 8, 2, 16, 1]);
        let input = Tensor::new(vec![1, 16, 2, 8, 8], input_values).unwrap();
        let weights = SafeTensorFile::open(vae_path).unwrap();
        let decoder = WanVae::load(&weights).unwrap();

        let scalar_started = Instant::now();
        let expected = decoder.pre.forward(&input, None).unwrap();
        let scalar_runtime = scalar_started.elapsed();
        let before = crate::vulkan::persistence_stats().unwrap();
        let prepare_started = Instant::now();
        let prepared = decoder.pre.prepare(&VULKAN_BACKEND).unwrap();
        let prepare_runtime = prepare_started.elapsed();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let execute_started = Instant::now();
        let device_output = decoder
            .pre
            .forward_uncached_with_backend(&device_input, &VULKAN_BACKEND, &prepared)
            .unwrap();
        let execute_runtime = execute_started.elapsed();
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
        assert_eq!(output.shape(), expected.shape());
        assert_eq!(
            after_prepare.resident_weight_uploads - before.resident_weight_uploads,
            1
        );
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "Wan VAE real prelude Conv3D: input={:?} output={:?} cosine={:.9} max_abs={:.9} mean_abs={:.9} scalar_ms={:.3} prepare_ms={:.3} execute_ms={:.3} device_local_bytes={} peak_resident_bytes={}",
            input.shape(),
            output.shape(),
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            scalar_runtime.as_secs_f64() * 1_000.0,
            prepare_runtime.as_secs_f64() * 1_000.0,
            execute_runtime.as_secs_f64() * 1_000.0,
            after_prepare.resident_device_local_bytes - before.resident_device_local_bytes,
            after.peak_resident_allocated_bytes,
        );
        crate::vulkan::print_statistics();

        drop(device_output);
        drop(device_input);
        drop(prepared);
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
    #[ignore = "loads the Wan VAE and validates the full small resident decoder"]
    fn resident_vulkan_small_decode_matches_reference() {
        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let vae_path = Path::new(VAE);
        let input_path = Path::new(REFERENCE).join("vae_in_small.bin");
        let output_path = Path::new(REFERENCE).join("vae_out_small.bin");
        for path in [vae_path, input_path.as_path(), output_path.as_path()] {
            assert!(path.exists(), "required parity input is missing: {path:?}");
        }
        let (captured_input_shape, input_values) = read_dump(&input_path).unwrap();
        let (captured_output_shape, expected_values) = read_dump(&output_path).unwrap();
        assert_eq!(captured_input_shape, [8, 8, 2, 16, 1]);
        assert_eq!(captured_output_shape, [64, 64, 5, 3, 1]);
        let input = Tensor::new(vec![1, 16, 2, 8, 8], input_values).unwrap();
        let expected = Tensor::new(vec![1, 3, 5, 64, 64], expected_values).unwrap();
        let weights = SafeTensorFile::open(vae_path).unwrap();
        let decoder = WanVae::load(&weights).unwrap();

        let before = crate::vulkan::persistence_stats().unwrap();
        let prepare_started = std::time::Instant::now();
        let prepared = decoder.prepare_with_backend(&VULKAN_BACKEND).unwrap();
        let prepare_runtime = prepare_started.elapsed();
        let after_prepare = crate::vulkan::persistence_stats().unwrap();
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let execute_started = std::time::Instant::now();
        let (device_output, cache_stats) = decoder
            .decode_device_with_backend(&device_input, &VULKAN_BACKEND, &prepared)
            .unwrap();
        let execute_runtime = execute_started.elapsed();
        let output = VULKAN_BACKEND.download_tensor(&device_output).unwrap();
        let after = crate::vulkan::persistence_stats().unwrap();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999,
                maximum_absolute_error: 0.03,
                maximum_mean_absolute_error: 0.005,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, 3, 5, 64, 64]);
        assert_eq!(cache_stats.occupied_slots, USED_CACHE_SLOTS);
        assert!(cache_stats.all_slots_resident);
        assert!(!cache_stats.all_slots_device_local);
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "resident Vulkan small VAE decode: input={:?} output={:?} prepare_ms={:.3} execute_ms={:.3} cosine={:.9} max_abs={:.9} mean_abs={:.9} current_vulkan_bytes={} peak_vulkan_bytes={} device_local_bytes={} peak_device_local_bytes={} cache_current_bytes={} cache_peak_bytes={} cache_occupied={} cache_replaced={} cache_device_local={} host_uploads={} weight_uploads={} downloads={} rss_kib={:?}",
            input.shape(),
            output.shape(),
            prepare_runtime.as_secs_f64() * 1_000.0,
            execute_runtime.as_secs_f64() * 1_000.0,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            after.peak_resident_allocated_bytes,
            after.resident_device_local_bytes - before.resident_device_local_bytes,
            after.peak_resident_device_local_bytes,
            cache_stats.current_bytes,
            cache_stats.peak_bytes,
            cache_stats.occupied_slots,
            cache_stats.replaced_slots,
            cache_stats.all_slots_device_local,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
            current_rss_kib(),
        );

        drop(device_output);
        drop(device_input);
        drop(prepared);
        let after_drop = crate::vulkan::persistence_stats().unwrap();
        assert_eq!(
            after_drop.resident_allocated_bytes,
            before.resident_allocated_bytes
        );
        assert_eq!(
            after_drop.resident_device_local_bytes,
            before.resident_device_local_bytes
        );
        assert!(after_prepare.resident_weight_uploads > before.resident_weight_uploads);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    #[ignore = "loads the Wan VAE and validates the full-size resident decoder"]
    fn resident_vulkan_full_decode_matches_reference() {
        use crate::{
            backend::{TensorBackend, VULKAN_BACKEND},
            parity::{ParityTolerance, compare_tensors},
        };

        let _persistence_guard = crate::vulkan::PERSISTENCE_TEST_LOCK.lock().unwrap();
        let vae_path = Path::new(VAE);
        let input_path = Path::new(REFERENCE).join("vae_in_full.bin");
        let output_path = Path::new(REFERENCE).join("vae_out_full.bin");
        for path in [vae_path, input_path.as_path(), output_path.as_path()] {
            assert!(path.exists(), "required parity input is missing: {path:?}");
        }
        let (captured_input_shape, input_values) = read_dump(&input_path).unwrap();
        let (captured_output_shape, expected_values) = read_dump(&output_path).unwrap();
        assert_eq!(captured_input_shape, [52, 30, 2, 16, 1]);
        assert_eq!(captured_output_shape, [416, 240, 5, 3, 1]);
        let input = Tensor::new(vec![1, 16, 2, 30, 52], input_values).unwrap();
        let expected = Tensor::new(vec![1, 3, 5, 240, 416], expected_values).unwrap();
        let weights = SafeTensorFile::open(vae_path).unwrap();
        let decoder = WanVae::load(&weights).unwrap();

        let before = crate::vulkan::persistence_stats().unwrap();
        let prepare_started = std::time::Instant::now();
        let prepared = decoder.prepare_with_backend(&VULKAN_BACKEND).unwrap();
        let prepare_runtime = prepare_started.elapsed();
        let device_input = VULKAN_BACKEND.upload_tensor(&input).unwrap();
        let execute_started = std::time::Instant::now();
        let (device_output, cache_stats) = decoder
            .decode_device_with_backend(&device_input, &VULKAN_BACKEND, &prepared)
            .unwrap();
        let execute_runtime = execute_started.elapsed();
        let output = VULKAN_BACKEND.download_tensor(&device_output).unwrap();
        let after = crate::vulkan::persistence_stats().unwrap();
        let metrics = compare_tensors(&output, &expected).unwrap();
        metrics
            .require(ParityTolerance {
                minimum_cosine_similarity: 0.999,
                maximum_absolute_error: 0.03,
                maximum_mean_absolute_error: 0.005,
            })
            .unwrap();
        assert_eq!(output.shape(), &[1, 3, 5, 240, 416]);
        assert_eq!(cache_stats.occupied_slots, USED_CACHE_SLOTS);
        assert!(cache_stats.all_slots_resident);
        assert!(!cache_stats.all_slots_device_local);
        assert_eq!(
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            1
        );
        assert_eq!(after.resident_downloads - before.resident_downloads, 1);
        println!(
            "resident Vulkan full VAE decode: input={:?} output={:?} prepare_ms={:.3} execute_ms={:.3} cosine={:.9} max_abs={:.9} mean_abs={:.9} current_vulkan_bytes={} peak_vulkan_bytes={} device_local_bytes={} peak_device_local_bytes={} cache_current_bytes={} cache_peak_bytes={} cache_occupied={} cache_replaced={} cache_device_local={} host_uploads={} weight_uploads={} downloads={} rss_kib={:?}",
            input.shape(),
            output.shape(),
            prepare_runtime.as_secs_f64() * 1_000.0,
            execute_runtime.as_secs_f64() * 1_000.0,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.mean_absolute_error,
            after.resident_allocated_bytes - before.resident_allocated_bytes,
            after.peak_resident_allocated_bytes,
            after.resident_device_local_bytes - before.resident_device_local_bytes,
            after.peak_resident_device_local_bytes,
            cache_stats.current_bytes,
            cache_stats.peak_bytes,
            cache_stats.occupied_slots,
            cache_stats.replaced_slots,
            cache_stats.all_slots_device_local,
            after.resident_tensor_uploads - before.resident_tensor_uploads,
            after.resident_weight_uploads - before.resident_weight_uploads,
            after.resident_downloads - before.resident_downloads,
            current_rss_kib(),
        );

        drop(device_output);
        drop(device_input);
        drop(prepared);
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

    #[test]
    #[ignore = "loads the 254MB VAE and performs scalar 3D convolutions; run explicitly"]
    fn small_decode_matches_reference() {
        run_case(
            "small",
            "vae_in_small.bin",
            "vae_out_small.bin",
            [8, 8, 2, 16, 1],
            [1, 16, 2, 8, 8],
            [64, 64, 5, 3, 1],
            [1, 3, 5, 64, 64],
        );
    }

    #[test]
    #[ignore = "loads the 254MB VAE and performs full 240x416 scalar 3D convolutions"]
    fn full_decode_matches_reference() {
        run_case(
            "full",
            "vae_in_full.bin",
            "vae_out_full.bin",
            [52, 30, 2, 16, 1],
            [1, 16, 2, 30, 52],
            [416, 240, 5, 3, 1],
            [1, 3, 5, 240, 416],
        );
    }
}
