//! Correctness-first tensor kernels for the Quartz diffusion path.
//!
//! Layout is explicit and contiguous. Image operators use NCHW and attention
//! uses BHQD. These kernels form the CPU reference implementation; Android
//! NEON/Vulkan kernels must match these tests before they can replace a path.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        validate_shape(&shape, data.len())?;
        Ok(Self { shape, data })
    }

    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let len = element_count(&shape)?;
        Ok(Self {
            shape,
            data: vec![0.0; len],
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.shape != other.shape {
            bail!("add shape mismatch: {:?} vs {:?}", self.shape, other.shape);
        }
        let data = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(a, b)| a + b)
            .collect();
        Self::new(self.shape.clone(), data)
    }

    pub fn silu(mut self) -> Self {
        self.data
            .par_iter_mut()
            .for_each(|x| *x /= 1.0 + (-*x).exp());
        self
    }

    pub fn quick_gelu(mut self) -> Self {
        self.data
            .par_iter_mut()
            .for_each(|x| *x *= 1.0 / (1.0 + (-1.702 * *x).exp()));
        self
    }

    pub fn gelu(mut self) -> Self {
        self.data.par_iter_mut().for_each(|value| {
            *value *= 0.5 * (1.0 + erf(*value * std::f32::consts::FRAC_1_SQRT_2));
        });
        self
    }

    /// Linear projection over the final dimension.
    /// `weight` is `[out_features, in_features]` and `bias` is `[out_features]`.
    pub fn linear(&self, weight: &Self, bias: Option<&Self>) -> Result<Self> {
        if weight.shape.len() != 2 {
            bail!("linear weight must be rank 2, got {:?}", weight.shape);
        }
        let out_features = weight.shape[0];
        let in_features = weight.shape[1];
        if self.shape.last().copied() != Some(in_features) {
            bail!(
                "linear input width {:?} does not match weight width {in_features}",
                self.shape.last()
            );
        }
        if let Some(bias) = bias {
            if bias.shape != [out_features] {
                bail!(
                    "linear bias shape {:?} must be [{out_features}]",
                    bias.shape
                );
            }
        }
        let rows = self.len() / in_features;
        let mut data = vec![0.0f32; rows * out_features];
        data.par_chunks_mut(out_features)
            .enumerate()
            .for_each(|(row, output)| {
                let input = &self.data[row * in_features..(row + 1) * in_features];
                for (out, value) in output.iter_mut().enumerate() {
                    let weights = &weight.data[out * in_features..(out + 1) * in_features];
                    let mut sum = bias.map_or(0.0, |b| b.data[out]);
                    for index in 0..in_features {
                        sum += input[index] * weights[index];
                    }
                    *value = sum;
                }
            });
        let mut shape = self.shape.clone();
        *shape.last_mut().expect("validated non-empty shape") = out_features;
        Self::new(shape, data)
    }

    /// Layer normalization over the final dimension.
    pub fn layer_norm(&self, weight: &Self, bias: &Self, epsilon: f32) -> Result<Self> {
        let width = *self
            .shape
            .last()
            .context("layer norm input has no dimensions")?;
        if weight.shape != [width] || bias.shape != [width] {
            bail!("layer norm parameters must both have shape [{width}]");
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("layer norm epsilon must be positive and finite");
        }
        let mut data = self.data.clone();
        data.par_chunks_mut(width).for_each(|row| {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row
                .iter()
                .map(|x| {
                    let d = *x - mean;
                    d * d
                })
                .sum::<f32>()
                / width as f32;
            let inverse_std = 1.0 / (variance + epsilon).sqrt();
            for i in 0..width {
                row[i] = (row[i] - mean) * inverse_std * weight.data[i] + bias.data[i];
            }
        });
        Self::new(self.shape.clone(), data)
    }

    /// Group normalization for a contiguous NCHW tensor.
    pub fn group_norm(
        &self,
        groups: usize,
        weight: &Self,
        bias: &Self,
        epsilon: f32,
    ) -> Result<Self> {
        let [batch, channels, height, width] = nchw(&self.shape)?;
        if groups == 0 || channels % groups != 0 {
            bail!("group count {groups} must divide channel count {channels}");
        }
        if weight.shape != [channels] || bias.shape != [channels] {
            bail!("group norm parameters must both have shape [{channels}]");
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            bail!("group norm epsilon must be positive and finite");
        }
        let channels_per_group = channels / groups;
        let plane = height * width;
        let group_len = channels_per_group * plane;
        let mut data = self.data.clone();
        data.par_chunks_mut(group_len)
            .enumerate()
            .for_each(|(group_index, values)| {
                let group = group_index % groups;
                let mean = values.iter().sum::<f32>() / group_len as f32;
                let variance = values
                    .iter()
                    .map(|x| {
                        let d = *x - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / group_len as f32;
                let inverse_std = 1.0 / (variance + epsilon).sqrt();
                for local_channel in 0..channels_per_group {
                    let channel = group * channels_per_group + local_channel;
                    let channel_start = local_channel * plane;
                    for value in &mut values[channel_start..channel_start + plane] {
                        *value = (*value - mean) * inverse_std * weight.data[channel]
                            + bias.data[channel];
                    }
                }
            });
        Self::new(vec![batch, channels, height, width], data)
    }

    /// NCHW convolution with OIHW weights and symmetric zero padding.
    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self> {
        let [batch, in_channels, in_height, in_width] = nchw(&self.shape)?;
        if weight.shape.len() != 4 {
            bail!("conv2d weight must be rank 4, got {:?}", weight.shape);
        }
        let out_channels = weight.shape[0];
        let weight_in_channels = weight.shape[1];
        let kernel_height = weight.shape[2];
        let kernel_width = weight.shape[3];
        if groups == 0 || in_channels % groups != 0 || out_channels % groups != 0 {
            bail!(
                "conv2d group count {groups} is incompatible with {in_channels} input and {out_channels} output channels"
            );
        }
        if weight_in_channels != in_channels / groups {
            bail!(
                "conv2d weight has {weight_in_channels} input channels per group, expected {}",
                in_channels / groups
            );
        }
        if stride.contains(&0) || dilation.contains(&0) || kernel_height == 0 || kernel_width == 0 {
            bail!("conv2d stride, dilation, and kernel dimensions must be non-zero");
        }
        if let Some(bias) = bias {
            if bias.shape != [out_channels] {
                bail!(
                    "conv2d bias shape {:?} must be [{out_channels}]",
                    bias.shape
                );
            }
        }
        let effective_kernel_height = dilation[0]
            .checked_mul(kernel_height - 1)
            .and_then(|value| value.checked_add(1))
            .context("conv2d effective kernel height overflow")?;
        let effective_kernel_width = dilation[1]
            .checked_mul(kernel_width - 1)
            .and_then(|value| value.checked_add(1))
            .context("conv2d effective kernel width overflow")?;
        let padded_height = padding[0]
            .checked_mul(2)
            .and_then(|value| in_height.checked_add(value))
            .context("conv2d padded height overflow")?;
        let padded_width = padding[1]
            .checked_mul(2)
            .and_then(|value| in_width.checked_add(value))
            .context("conv2d padded width overflow")?;
        if padded_height < effective_kernel_height || padded_width < effective_kernel_width {
            bail!("conv2d effective kernel is larger than the padded input");
        }
        let out_height = (padded_height - effective_kernel_height) / stride[0] + 1;
        let out_width = (padded_width - effective_kernel_width) / stride[1] + 1;
        let output_plane = out_height
            .checked_mul(out_width)
            .context("conv2d output plane overflow")?;
        let input_plane = in_height
            .checked_mul(in_width)
            .context("conv2d input plane overflow")?;
        let input_channels_per_group = in_channels / groups;
        let output_channels_per_group = out_channels / groups;
        let weight_plane = kernel_height
            .checked_mul(kernel_width)
            .context("conv2d weight plane overflow")?;
        let output_len = batch
            .checked_mul(out_channels)
            .and_then(|value| value.checked_mul(output_plane))
            .context("conv2d output size overflow")?;
        let mut data = vec![0.0f32; output_len];

        data.par_chunks_mut(output_plane)
            .enumerate()
            .for_each(|(plane_index, output)| {
                let sample = plane_index / out_channels;
                let out_channel = plane_index % out_channels;
                let group = out_channel / output_channels_per_group;
                let input_channel_start = group * input_channels_per_group;
                for out_y in 0..out_height {
                    for out_x in 0..out_width {
                        let mut sum = bias.map_or(0.0, |b| b.data[out_channel]);
                        for local_in_channel in 0..input_channels_per_group {
                            let in_channel = input_channel_start + local_in_channel;
                            let input_base = (sample * in_channels + in_channel) * input_plane;
                            let weight_base = (out_channel * input_channels_per_group
                                + local_in_channel)
                                * weight_plane;
                            for kernel_y in 0..kernel_height {
                                let padded_y = out_y * stride[0] + kernel_y * dilation[0];
                                if padded_y < padding[0] {
                                    continue;
                                }
                                let in_y = padded_y - padding[0];
                                if in_y >= in_height {
                                    continue;
                                }
                                for kernel_x in 0..kernel_width {
                                    let padded_x = out_x * stride[1] + kernel_x * dilation[1];
                                    if padded_x < padding[1] {
                                        continue;
                                    }
                                    let in_x = padded_x - padding[1];
                                    if in_x >= in_width {
                                        continue;
                                    }
                                    sum += self.data[input_base + in_y * in_width + in_x]
                                        * weight.data
                                            [weight_base + kernel_y * kernel_width + kernel_x];
                                }
                            }
                        }
                        output[out_y * out_width + out_x] = sum;
                    }
                }
            });
        Self::new(vec![batch, out_channels, out_height, out_width], data)
    }

    pub fn upsample_nearest2d(&self, scale: [usize; 2]) -> Result<Self> {
        let [batch, channels, height, width] = nchw(&self.shape)?;
        if scale.contains(&0) {
            bail!("upsample scale must be non-zero");
        }
        let out_height = height
            .checked_mul(scale[0])
            .context("upsample height overflow")?;
        let out_width = width
            .checked_mul(scale[1])
            .context("upsample width overflow")?;
        let output_plane = out_height
            .checked_mul(out_width)
            .context("upsample output plane overflow")?;
        let input_plane = height
            .checked_mul(width)
            .context("upsample input plane overflow")?;
        let output_len = batch
            .checked_mul(channels)
            .and_then(|value| value.checked_mul(output_plane))
            .context("upsample output size overflow")?;
        let mut data = vec![0.0f32; output_len];
        data.par_chunks_mut(output_plane)
            .enumerate()
            .for_each(|(plane_index, output)| {
                let input = &self.data[plane_index * input_plane..(plane_index + 1) * input_plane];
                for y in 0..out_height {
                    let in_y = y / scale[0];
                    for x in 0..out_width {
                        output[y * out_width + x] = input[in_y * width + x / scale[1]];
                    }
                }
            });
        Self::new(vec![batch, channels, out_height, out_width], data)
    }

    pub fn concat_channels(tensors: &[&Self]) -> Result<Self> {
        let first = tensors
            .first()
            .context("concat requires at least one tensor")?;
        let [batch, _, height, width] = nchw(&first.shape)?;
        let mut total_channels = 0usize;
        for tensor in tensors {
            let [next_batch, channels, next_height, next_width] = nchw(&tensor.shape)?;
            if [next_batch, next_height, next_width] != [batch, height, width] {
                bail!(
                    "concat N/H/W mismatch: {:?} vs {:?}",
                    first.shape,
                    tensor.shape
                );
            }
            total_channels = total_channels
                .checked_add(channels)
                .context("concat channel count overflow")?;
        }
        let plane = height
            .checked_mul(width)
            .context("concat plane size overflow")?;
        let output_len = batch
            .checked_mul(total_channels)
            .and_then(|value| value.checked_mul(plane))
            .context("concat output size overflow")?;
        let mut data = Vec::with_capacity(output_len);
        for sample in 0..batch {
            for tensor in tensors {
                let channels = tensor.shape[1];
                let start = sample * channels * plane;
                data.extend_from_slice(&tensor.data[start..start + channels * plane]);
            }
        }
        Self::new(vec![batch, total_channels, height, width], data)
    }

    /// Scaled dot-product attention for Q/K/V tensors in BHQD/BHKD layout.
    pub fn attention(query: &Self, key: &Self, value: &Self) -> Result<Self> {
        #[cfg(feature = "vulkan")]
        if crate::vulkan::sd_acceleration_requested() {
            return crate::vulkan::attention(query, key, value, false);
        }
        Self::attention_impl(query, key, value, false)
    }

    /// CLIP text attention, where query N cannot observe keys after N.
    pub fn attention_causal(query: &Self, key: &Self, value: &Self) -> Result<Self> {
        #[cfg(feature = "vulkan")]
        if crate::vulkan::sd_acceleration_requested() {
            return crate::vulkan::attention(query, key, value, true);
        }
        Self::attention_impl(query, key, value, true)
    }

    fn attention_impl(query: &Self, key: &Self, value: &Self, causal: bool) -> Result<Self> {
        if query.shape.len() != 4 || key.shape.len() != 4 || value.shape.len() != 4 {
            bail!("attention inputs must all be rank 4");
        }
        let (batch, heads, queries, width) = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            query.shape[3],
        );
        let keys = key.shape[2];
        if key.shape != [batch, heads, keys, width] || value.shape != [batch, heads, keys, width] {
            bail!(
                "attention K/V shapes are incompatible with Q: {:?}, {:?}, {:?}",
                query.shape,
                key.shape,
                value.shape
            );
        }
        if causal && queries != keys {
            bail!("causal attention requires equal query and key lengths");
        }
        let rows = batch
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(queries))
            .context("attention row count overflow")?;
        let output_len = rows
            .checked_mul(width)
            .context("attention output size overflow")?;
        let mut data = vec![0.0f32; output_len];
        let scale = 1.0 / (width as f32).sqrt();
        data.par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, output)| {
                let query_index = row % queries;
                let batch_head = row / queries;
                let q_start = (batch_head * queries + query_index) * width;
                let q = &query.data[q_start..q_start + width];
                let visible_keys = if causal { query_index + 1 } else { keys };
                let mut scores = vec![0.0f32; visible_keys];
                for key_index in 0..visible_keys {
                    let k_start = (batch_head * keys + key_index) * width;
                    let mut dot = 0.0;
                    for d in 0..width {
                        dot += q[d] * key.data[k_start + d];
                    }
                    scores[key_index] = dot * scale;
                }
                softmax(&mut scores);
                for key_index in 0..visible_keys {
                    let v_start = (batch_head * keys + key_index) * width;
                    for d in 0..width {
                        output[d] += scores[key_index] * value.data[v_start + d];
                    }
                }
            });
        Self::new(vec![batch, heads, queries, width], data)
    }
}

fn element_count(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        bail!("tensor shape cannot be empty");
    }
    shape.iter().try_fold(1usize, |total, &dim| {
        if dim == 0 {
            bail!("tensor dimensions must be non-zero");
        }
        total
            .checked_mul(dim)
            .context("tensor element count overflow")
    })
}

fn validate_shape(shape: &[usize], actual_len: usize) -> Result<()> {
    let expected = element_count(shape)?;
    if expected != actual_len {
        bail!(
            "tensor shape {:?} requires {expected} values, got {actual_len}",
            shape
        );
    }
    Ok(())
}

fn nchw(shape: &[usize]) -> Result<[usize; 4]> {
    shape
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected rank-4 NCHW tensor, got {shape:?}"))
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

/// Error-function approximation with maximum error around 1.2e-7. This keeps
/// GEGLU independent of a framework or math helper library.
fn erf(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs() as f64;
    let t = 1.0 / (1.0 + 0.5 * x);
    let tau = t
        * (-x * x - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    sign * (1.0 - tau as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), data.to_vec()).unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn conv2d_handles_padding_stride_and_bias() {
        let input = tensor(&[1, 1, 3, 3], &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let weight = tensor(&[1, 1, 2, 2], &[1., 0., 0., -1.]);
        let bias = tensor(&[1], &[0.5]);
        let output = input
            .conv2d(&weight, Some(&bias), [1, 1], [0, 0], [1, 1], 1)
            .unwrap();
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_close(output.data(), &[-3.5, -3.5, -3.5, -3.5], 1e-6);
    }

    #[test]
    fn grouped_conv_does_not_cross_channels() {
        let input = tensor(&[1, 2, 1, 2], &[1., 2., 10., 20.]);
        let weight = tensor(&[2, 1, 1, 1], &[2., 3.]);
        let output = input
            .conv2d(&weight, None, [1, 1], [0, 0], [1, 1], 2)
            .unwrap();
        assert_close(output.data(), &[2., 4., 30., 60.], 1e-6);
    }

    #[test]
    fn group_norm_normalizes_each_group() {
        let input = tensor(&[1, 2, 1, 2], &[1., 3., 10., 14.]);
        let weight = tensor(&[2], &[1., 1.]);
        let bias = tensor(&[2], &[0., 0.]);
        let output = input.group_norm(2, &weight, &bias, 1e-5).unwrap();
        assert_close(
            output.data(),
            &[-0.999995, 0.999995, -0.999999, 0.999999],
            2e-5,
        );
    }

    #[test]
    fn layer_norm_uses_last_dimension() {
        let input = tensor(&[2, 2], &[1., 3., 2., 6.]);
        let weight = tensor(&[2], &[2., 0.5]);
        let bias = tensor(&[2], &[1., -1.]);
        let output = input.layer_norm(&weight, &bias, 1e-5).unwrap();
        assert_close(
            output.data(),
            &[-0.99999, -0.5000025, -0.9999975, -0.5000006],
            2e-5,
        );
    }

    #[test]
    fn linear_projects_the_last_dimension() {
        let input = tensor(&[1, 2, 2], &[1., 2., 3., 4.]);
        let weight = tensor(&[2, 2], &[1., 0., 0.5, 2.]);
        let bias = tensor(&[2], &[1., -1.]);
        let output = input.linear(&weight, Some(&bias)).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_close(output.data(), &[2., 3.5, 4., 8.5], 1e-6);
    }

    #[test]
    fn gelu_matches_known_values() {
        let output = tensor(&[3], &[-1.0, 0.0, 1.0]).gelu();
        assert_close(output.data(), &[-0.15865526, 0.0, 0.8413447], 2e-7);
    }

    #[test]
    fn nearest_upsample_repeats_pixels() {
        let input = tensor(&[1, 1, 2, 2], &[1., 2., 3., 4.]);
        let output = input.upsample_nearest2d([2, 2]).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        assert_close(
            output.data(),
            &[
                1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.,
            ],
            0.0,
        );
    }

    #[test]
    fn concat_channels_preserves_sample_order() {
        let first = tensor(&[2, 1, 1, 1], &[1., 2.]);
        let second = tensor(&[2, 2, 1, 1], &[3., 4., 5., 6.]);
        let output = Tensor::concat_channels(&[&first, &second]).unwrap();
        assert_eq!(output.shape(), &[2, 3, 1, 1]);
        assert_eq!(output.data(), &[1., 3., 4., 2., 5., 6.]);
    }

    #[test]
    fn attention_matches_a_small_reference() {
        let query = tensor(&[1, 1, 1, 2], &[1., 0.]);
        let key = tensor(&[1, 1, 2, 2], &[1., 0., 0., 1.]);
        let value = tensor(&[1, 1, 2, 2], &[2., 4., 10., 20.]);
        let output = Tensor::attention(&query, &key, &value).unwrap();
        let first_weight = (1.0f32 / 2.0f32.sqrt()).exp();
        let expected_weight = first_weight / (first_weight + 1.0);
        assert_close(
            output.data(),
            &[
                expected_weight * 2.0 + (1.0 - expected_weight) * 10.0,
                expected_weight * 4.0 + (1.0 - expected_weight) * 20.0,
            ],
            1e-5,
        );
    }

    #[test]
    fn causal_attention_cannot_see_future_values() {
        let query = tensor(&[1, 1, 2, 1], &[1.0, 1.0]);
        let key = tensor(&[1, 1, 2, 1], &[1.0, 100.0]);
        let value = tensor(&[1, 1, 2, 1], &[3.0, 9.0]);
        let output = Tensor::attention_causal(&query, &key, &value).unwrap();
        assert_eq!(output.data()[0], 3.0);
        assert!(output.data()[1] > 8.99);
    }

    #[test]
    fn rejects_invalid_shapes_instead_of_panicking() {
        assert!(Tensor::new(vec![2, 2], vec![0.0; 3]).is_err());
        assert!(Tensor::zeros(vec![1, 0, 2]).is_err());
    }
}
