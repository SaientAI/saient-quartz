//! Wan 2.1 3D causal VAE decoder.
//!
//! The decoder is deliberately stateful: the latent is processed one frame at
//! a time and every causal 3x3x3 convolution retains its last two input feature
//! frames. Tensor layout is contiguous NCTHW, unlike SQD1/ggml dumps whose
//! dimension list is fastest-first `[W,H,T,C,N]`.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

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
}

struct Conv2dFrames {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: [usize; 2],
    padding: [usize; 2],
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
}

struct SpatialAttention {
    norm: Tensor,
    qkv: Conv2dFrames,
    projection: Conv2dFrames,
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
}

struct Upsample {
    temporal: Option<CausalConv3d>,
    spatial: Conv2dFrames,
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
        hidden = upsample_spatial_nearest(&hidden, 2)?;
        self.spatial.forward(&hidden)
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
