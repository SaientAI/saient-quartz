//! Quartz-owned SD1.5 conditional UNet graph.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{safetensors::SafeTensorFile, sd_ops, sd_scheduler, tensor::Tensor};

const GROUPS: usize = 32;
const RESNET_EPSILON: f32 = 1e-5;
const TRANSFORMER_GROUP_EPSILON: f32 = 1e-6;
const LAYER_NORM_EPSILON: f32 = 1e-5;
const ATTENTION_HEADS: usize = 8;

pub fn predict_noise(
    weights: &SafeTensorFile,
    sample: &Tensor,
    timestep: f32,
    context: &Tensor,
) -> Result<Tensor> {
    let [batch, channels, height, width]: [usize; 4] = sample
        .shape()
        .try_into()
        .context("SD1.5 UNet sample must be NCHW")?;
    if channels != 4 || height < 8 || width < 8 || height % 8 != 0 || width % 8 != 0 {
        bail!(
            "SD1.5 UNet sample must be [B, 4, H, W] with H/W divisible by 8, got {:?}",
            sample.shape()
        );
    }
    if context.shape() != [batch, 77, 768] {
        bail!(
            "SD1.5 UNet context must be [{batch}, 77, 768], got {:?}",
            context.shape()
        );
    }

    let time = time_embedding(weights, timestep, batch)?;
    let mut hidden = sd_ops::conv2d(sample, weights, "conv_in", [1, 1])?;
    let mut skips = vec![hidden.clone()];

    for block in 0..4 {
        // Stage boundary: evicts the previous down block's cached weights before this
        // one's are touched, so a staged-loading run never has to hold more than one
        // block's tensors resident at once. A no-op unless staged loading was opted in.
        begin_weight_stage()?;
        for layer in 0..2 {
            hidden = resnet(
                weights,
                &hidden,
                &time,
                &format!("down_blocks.{block}.resnets.{layer}"),
            )?;
            if block < 3 {
                hidden = transformer(
                    weights,
                    &hidden,
                    context,
                    &format!("down_blocks.{block}.attentions.{layer}"),
                )?;
            }
            skips.push(hidden.clone());
        }
        if block < 3 {
            hidden = sd_ops::conv2d_full(
                &hidden,
                weights,
                &format!("down_blocks.{block}.downsamplers.0.conv"),
                [2, 2],
                [1, 1],
            )?;
            skips.push(hidden.clone());
        }
    }

    begin_weight_stage()?;
    hidden = resnet(weights, &hidden, &time, "mid_block.resnets.0")?;
    hidden = transformer(weights, &hidden, context, "mid_block.attentions.0")?;
    hidden = resnet(weights, &hidden, &time, "mid_block.resnets.1")?;

    for block in 0..4 {
        begin_weight_stage()?;
        for layer in 0..3 {
            let skip = skips.pop().with_context(|| {
                format!("UNet skip stack underflow at up block {block}.{layer}")
            })?;
            let joined = Tensor::concat_channels(&[&hidden, &skip])?;
            hidden = resnet(
                weights,
                &joined,
                &time,
                &format!("up_blocks.{block}.resnets.{layer}"),
            )?;
            if block > 0 {
                hidden = transformer(
                    weights,
                    &hidden,
                    context,
                    &format!("up_blocks.{block}.attentions.{layer}"),
                )?;
            }
        }
        if block < 3 {
            hidden = hidden.upsample_nearest2d([2, 2])?;
            hidden = sd_ops::conv2d(
                &hidden,
                weights,
                &format!("up_blocks.{block}.upsamplers.0.conv"),
                [1, 1],
            )?;
        }
    }
    if !skips.is_empty() {
        bail!("UNet left {} unused skip tensors", skips.len());
    }

    begin_weight_stage()?;
    hidden = sd_ops::group_norm(&hidden, weights, "conv_norm_out", GROUPS, RESNET_EPSILON)?;
    hidden = hidden.silu();
    sd_ops::conv2d(&hidden, weights, "conv_out", [1, 1])
}

/// Evict cached Vulkan weights between UNet stages. A no-op when Vulkan is
/// unavailable/disabled or when staged loading hasn't been opted into, so
/// calling this unconditionally at every block boundary costs nothing in the
/// default (CPU or whole-file-cached Vulkan) path.
fn begin_weight_stage() -> Result<()> {
    #[cfg(feature = "vulkan")]
    {
        return crate::vulkan::begin_weight_stage();
    }
    #[cfg(not(feature = "vulkan"))]
    {
        Ok(())
    }
}

/// Every named-tensor prefix that becomes its own staged-loading eviction
/// boundary in `predict_noise`, in call order. Kept in one place so the
/// budget sizing below can never drift from the actual `begin_weight_stage`
/// call sites above.
fn stage_prefixes() -> Vec<Vec<String>> {
    let mut stages = vec![vec!["conv_in".to_string(), "time_embedding".to_string()]];
    for block in 0..4 {
        stages.push(vec![format!("down_blocks.{block}.")]);
    }
    stages.push(vec!["mid_block.".to_string()]);
    for block in 0..4 {
        stages.push(vec![format!("up_blocks.{block}.")]);
    }
    stages.push(vec!["conv_norm_out".to_string(), "conv_out".to_string()]);
    stages
}

/// The Vulkan weight-arena budget staged loading needs for this UNet: the
/// largest single stage's tensor bytes (plus slack) and how many tensors that
/// stage holds. Sized directly from the model's real tensor lengths so it
/// self-adjusts for whatever UNet is loaded (SD1.5 today, a larger UNet
/// later) rather than hardcoding a number that only happens to fit one model.
pub fn staged_loading_budget(weights: &SafeTensorFile) -> Result<(usize, usize)> {
    let mut max_bytes = 0usize;
    let mut max_count = 0usize;
    for stage in stage_prefixes() {
        let mut bytes = 0usize;
        let mut count = 0usize;
        for name in weights.tensor_names() {
            if stage.iter().any(|prefix| name.starts_with(prefix.as_str())) {
                let info = weights
                    .info(name)
                    .with_context(|| format!("missing tensor info for {name}"))?;
                bytes += info.byte_len();
                count += 1;
            }
        }
        max_bytes = max_bytes.max(bytes);
        max_count = max_count.max(count);
    }
    if max_bytes == 0 {
        bail!("could not size a staged-loading budget: no UNet tensors matched a known stage");
    }
    // 25% slack: alignment padding within the arena is handled separately, this
    // just protects against undercounting if a stage's tensor set grows later.
    let budget_bytes = max_bytes + max_bytes / 4;
    let tensor_count_hint = max_count + max_count / 4 + 1;
    Ok((budget_bytes, tensor_count_hint))
}

pub(crate) fn time_embedding(
    weights: &SafeTensorFile,
    timestep: f32,
    batch: usize,
) -> Result<Tensor> {
    let one = sd_scheduler::timestep_embedding(timestep, 320)?;
    let mut data = Vec::with_capacity(batch * one.len());
    for _ in 0..batch {
        data.extend_from_slice(&one);
    }
    let embedding = Tensor::new(vec![batch, 320], data)?;
    let embedding = sd_ops::linear(&embedding, weights, "time_embedding.linear_1")?.silu();
    sd_ops::linear(&embedding, weights, "time_embedding.linear_2")
}

pub(crate) fn resnet(
    weights: &SafeTensorFile,
    input: &Tensor,
    time: &Tensor,
    prefix: &str,
) -> Result<Tensor> {
    let time = sd_ops::linear(
        &time.clone().silu(),
        weights,
        &format!("{prefix}.time_emb_proj"),
    )?;
    #[cfg(feature = "vulkan")]
    if crate::vulkan::sd_acceleration_requested() {
        return crate::vulkan::resnet(input, Some(&time), weights, prefix, GROUPS, RESNET_EPSILON);
    }
    let residual = if weights
        .info(&format!("{prefix}.conv_shortcut.weight"))
        .is_some()
    {
        sd_ops::conv2d(input, weights, &format!("{prefix}.conv_shortcut"), [0, 0])?
    } else {
        input.clone()
    };
    let mut hidden = sd_ops::group_norm(
        input,
        weights,
        &format!("{prefix}.norm1"),
        GROUPS,
        RESNET_EPSILON,
    )?
    .silu();
    hidden = sd_ops::conv2d(&hidden, weights, &format!("{prefix}.conv1"), [1, 1])?;

    hidden = add_channel_bias(&hidden, &time)?;
    hidden = sd_ops::group_norm(
        &hidden,
        weights,
        &format!("{prefix}.norm2"),
        GROUPS,
        RESNET_EPSILON,
    )?
    .silu();
    hidden = sd_ops::conv2d(&hidden, weights, &format!("{prefix}.conv2"), [1, 1])?;
    residual.add(&hidden)
}

pub(crate) fn add_channel_bias(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let [batch, channels, height, width]: [usize; 4] = input
        .shape()
        .try_into()
        .context("channel bias input must be NCHW")?;
    if bias.shape() != [batch, channels] {
        bail!(
            "channel bias shape {:?} must be [{batch}, {channels}]",
            bias.shape()
        );
    }
    let plane = height * width;
    let mut output = input.clone();
    output
        .data_mut()
        .par_chunks_mut(plane)
        .enumerate()
        .for_each(|(plane_index, values)| {
            let value = bias.data()[plane_index];
            for element in values {
                *element += value;
            }
        });
    Ok(output)
}

fn transformer(
    weights: &SafeTensorFile,
    input: &Tensor,
    context: &Tensor,
    prefix: &str,
) -> Result<Tensor> {
    let residual = input.clone();
    let mut hidden = sd_ops::group_norm(
        input,
        weights,
        &format!("{prefix}.norm"),
        GROUPS,
        TRANSFORMER_GROUP_EPSILON,
    )?;
    hidden = sd_ops::conv2d(&hidden, weights, &format!("{prefix}.proj_in"), [0, 0])?;
    let [_, _, height, width]: [usize; 4] = hidden
        .shape()
        .try_into()
        .expect("convolution preserves NCHW rank");
    let mut sequence = nchw_to_sequence(&hidden)?;
    let block = format!("{prefix}.transformer_blocks.0");

    let norm = sd_ops::layer_norm(
        &sequence,
        weights,
        &format!("{block}.norm1"),
        LAYER_NORM_EPSILON,
    )?;
    let attended = attention(weights, &norm, &norm, &format!("{block}.attn1"))?;
    sequence = sequence.add(&attended)?;

    let norm = sd_ops::layer_norm(
        &sequence,
        weights,
        &format!("{block}.norm2"),
        LAYER_NORM_EPSILON,
    )?;
    let attended = attention(weights, &norm, context, &format!("{block}.attn2"))?;
    sequence = sequence.add(&attended)?;

    let norm = sd_ops::layer_norm(
        &sequence,
        weights,
        &format!("{block}.norm3"),
        LAYER_NORM_EPSILON,
    )?;
    let feed_forward = feed_forward(weights, &norm, &format!("{block}.ff"))?;
    sequence = sequence.add(&feed_forward)?;

    hidden = sequence_to_nchw(&sequence, height, width)?;
    hidden = sd_ops::conv2d(&hidden, weights, &format!("{prefix}.proj_out"), [0, 0])?;
    residual.add(&hidden)
}

pub(crate) fn feed_forward(
    weights: &SafeTensorFile,
    input: &Tensor,
    prefix: &str,
) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    if crate::vulkan::sd_acceleration_requested() {
        return crate::vulkan::feed_forward(input, weights, prefix);
    }
    let projected = sd_ops::linear(input, weights, &format!("{prefix}.net.0.proj"))?;
    let gated = geglu(&projected)?;
    sd_ops::linear(&gated, weights, &format!("{prefix}.net.2"))
}

fn attention(
    weights: &SafeTensorFile,
    query_input: &Tensor,
    key_value_input: &Tensor,
    prefix: &str,
) -> Result<Tensor> {
    if query_input.shape().len() != 3 || key_value_input.shape().len() != 3 {
        bail!("UNet attention inputs must have [B, sequence, channels] layout");
    }
    if query_input.shape()[0] != key_value_input.shape()[0] {
        bail!("UNet attention batch mismatch");
    }
    #[cfg(feature = "vulkan")]
    if crate::vulkan::sd_acceleration_requested() {
        return crate::vulkan::projected_attention(
            query_input,
            key_value_input,
            weights,
            prefix,
            ATTENTION_HEADS,
        );
    }
    let query = sd_ops::linear(query_input, weights, &format!("{prefix}.to_q"))?;
    let key = sd_ops::linear(key_value_input, weights, &format!("{prefix}.to_k"))?;
    let value = sd_ops::linear(key_value_input, weights, &format!("{prefix}.to_v"))?;
    let query = split_heads(&query, ATTENTION_HEADS)?;
    let key = split_heads(&key, ATTENTION_HEADS)?;
    let value = split_heads(&value, ATTENTION_HEADS)?;
    let attended = Tensor::attention(&query, &key, &value)?;
    let attended = merge_heads(&attended)?;
    sd_ops::linear(&attended, weights, &format!("{prefix}.to_out.0"))
}

pub(crate) fn geglu(projected: &Tensor) -> Result<Tensor> {
    let total = *projected
        .shape()
        .last()
        .context("GEGLU input has no final dimension")?;
    if total % 2 != 0 {
        bail!("GEGLU projection width {total} is not even");
    }
    let width = total / 2;
    let rows = projected.len() / total;
    let mut values = Vec::with_capacity(rows * width);
    let mut gates = Vec::with_capacity(rows * width);
    for row in projected.data().chunks_exact(total) {
        values.extend_from_slice(&row[..width]);
        gates.extend_from_slice(&row[width..]);
    }
    let gates = Tensor::new(vec![rows, width], gates)?.gelu();
    for (value, gate) in values.iter_mut().zip(gates.data()) {
        *value *= *gate;
    }
    let mut shape = projected.shape().to_vec();
    *shape.last_mut().expect("validated final dimension") = width;
    Tensor::new(shape, values)
}

pub(crate) fn split_heads(input: &Tensor, heads: usize) -> Result<Tensor> {
    if input.shape().len() != 3 {
        bail!(
            "attention head split requires rank 3, got {:?}",
            input.shape()
        );
    }
    let batch = input.shape()[0];
    let sequence = input.shape()[1];
    let channels = input.shape()[2];
    if heads == 0 || channels % heads != 0 {
        bail!("{heads} attention heads do not divide {channels} channels");
    }
    let width = channels / heads;
    let mut output = vec![0.0; input.len()];
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, values)| {
            let position = row % sequence;
            let sample_head = row / sequence;
            let head = sample_head % heads;
            let sample = sample_head / heads;
            let input_start = (sample * sequence + position) * channels + head * width;
            values.copy_from_slice(&input.data()[input_start..input_start + width]);
        });
    Tensor::new(vec![batch, heads, sequence, width], output)
}

pub(crate) fn merge_heads(input: &Tensor) -> Result<Tensor> {
    if input.shape().len() != 4 {
        bail!(
            "attention head merge requires rank 4, got {:?}",
            input.shape()
        );
    }
    let batch = input.shape()[0];
    let heads = input.shape()[1];
    let sequence = input.shape()[2];
    let width = input.shape()[3];
    let channels = heads * width;
    let mut output = vec![0.0; input.len()];
    output
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(row, values)| {
            let sample = row / sequence;
            let position = row % sequence;
            for head in 0..heads {
                let input_start = ((sample * heads + head) * sequence + position) * width;
                values[head * width..(head + 1) * width]
                    .copy_from_slice(&input.data()[input_start..input_start + width]);
            }
        });
    Tensor::new(vec![batch, sequence, channels], output)
}

pub(crate) fn nchw_to_sequence(input: &Tensor) -> Result<Tensor> {
    let [batch, channels, height, width]: [usize; 4] = input
        .shape()
        .try_into()
        .context("NCHW-to-sequence requires rank 4")?;
    let positions = height * width;
    let mut output = vec![0.0; input.len()];
    output
        .par_chunks_mut(channels)
        .enumerate()
        .for_each(|(row, values)| {
            let sample = row / positions;
            let position = row % positions;
            for (channel, value) in values.iter_mut().enumerate() {
                *value = input.data()[(sample * channels + channel) * positions + position];
            }
        });
    Tensor::new(vec![batch, positions, channels], output)
}

pub(crate) fn sequence_to_nchw(input: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    if input.shape().len() != 3 || input.shape()[1] != height * width {
        bail!(
            "sequence shape {:?} is incompatible with {height}x{width}",
            input.shape()
        );
    }
    let batch = input.shape()[0];
    let positions = input.shape()[1];
    let channels = input.shape()[2];
    let mut output = vec![0.0; input.len()];
    output
        .par_chunks_mut(positions)
        .enumerate()
        .for_each(|(plane_index, values)| {
            let sample = plane_index / channels;
            let channel = plane_index % channels;
            for (position, value) in values.iter_mut().enumerate() {
                *value = input.data()[(sample * positions + position) * channels + channel];
            }
        });
    Tensor::new(vec![batch, channels, height, width], output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_layout_round_trips() {
        let input = Tensor::new(vec![2, 3, 8], (0..48).map(|x| x as f32).collect()).unwrap();
        assert_eq!(
            merge_heads(&split_heads(&input, 4).unwrap()).unwrap(),
            input
        );
    }

    #[test]
    fn geglu_splits_values_before_gates() {
        let projected = Tensor::new(vec![1, 4], vec![2.0, 3.0, 0.0, 1.0]).unwrap();
        let output = geglu(&projected).unwrap();
        assert_eq!(output.shape(), &[1, 2]);
        assert!(output.data()[0].abs() < 1e-6);
        assert!((output.data()[1] - 2.524034).abs() < 1e-5);
    }

    #[test]
    fn rejects_latent_shapes_that_break_the_skip_pyramid() {
        let sample = Tensor::zeros(vec![1, 4, 7, 8]).unwrap();
        let context = Tensor::zeros(vec![1, 77, 768]).unwrap();
        let error = predict_noise_fixture_shape_check(&sample, &context).unwrap_err();
        assert!(error.to_string().contains("divisible by 8"));
    }

    #[test]
    #[ignore = "requires QUARTZ_SD15_MODEL_DIR pointing to the official SD1.5 FP16 pack"]
    fn matches_official_unet_golden_output() {
        let root = std::env::var("QUARTZ_SD15_MODEL_DIR").unwrap();
        let pack = crate::sd15::Sd15Pack::open(root).unwrap();
        let context = pack
            .encode_prompt("A photo of a lion in the wild, ultra realistic")
            .unwrap();
        let sample = Tensor::new(
            vec![1, 4, 8, 8],
            (0..256)
                .map(|index| ((index % 17) as f32 - 8.0) / 8.0)
                .collect(),
        )
        .unwrap();
        let output = pack.predict_noise(&sample, 951.0, &context).unwrap();
        let expected = [
            -1.0276078,
            -0.81983143,
            -0.6800161,
            -0.5596536,
            -0.39237583,
            -0.24639142,
            -0.13097233,
            0.014924392,
            -0.06278713,
            0.013539311,
            0.09616388,
            0.27323303,
            0.39601672,
            0.5781605,
            0.8111968,
            1.0397592,
        ];
        for (index, expected) in expected.into_iter().enumerate() {
            let actual = output.data()[index];
            assert!(
                (actual - expected).abs() < 5e-5,
                "UNet output {index}: {actual} != {expected}"
            );
        }
    }

    fn predict_noise_fixture_shape_check(sample: &Tensor, context: &Tensor) -> Result<()> {
        let [batch, channels, height, width]: [usize; 4] = sample
            .shape()
            .try_into()
            .context("SD1.5 UNet sample must be NCHW")?;
        if channels != 4 || height < 8 || width < 8 || height % 8 != 0 || width % 8 != 0 {
            bail!("sample dimensions must be divisible by 8");
        }
        if context.shape() != [batch, 77, 768] {
            bail!("invalid context");
        }
        Ok(())
    }
}
