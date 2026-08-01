//! SDXL's conditional UNet graph.
//!
//! Structurally different from SD1.5's (sd_unet.rs), not just larger:
//! - 3 down/up levels instead of 4 (`block_out_channels: [320, 640, 1280]`).
//! - The first down/last up level has no attention at all (`DownBlock2D`/`UpBlock2D`).
//! - Transformer depth per level is non-uniform: `[1, 2, 10]` down, reversed up.
//! - Attention head count varies per level (`[5, 10, 20]`) at a constant 64-wide head.
//! - Transformer blocks use a *linear* proj_in/proj_out (`use_linear_projection:
//!   true`), not SD1.5's conv1x1.
//! - An extra `add_embedding` branch folds the pooled text embedding and six
//!   sinusoidal "time_ids" (original_size, crop_coords_top_left, target_size)
//!   into the timestep embedding before it reaches any ResNet block.
//!
//! ResNet blocks, the time embedding base MLP, feed-forward/GEGLU, and the
//! head split/merge + NCHW<->sequence reshapes are architecturally identical
//! to SD1.5's and are reused directly from `sd_unet` (shared via `pub(crate)`).

use anyhow::{Context, Result, bail};

use crate::{safetensors::SafeTensorFile, sd_ops, sd_scheduler, sd_unet, tensor::Tensor};

const GROUPS: usize = 32;
const RESNET_EPSILON: f32 = 1e-5;
const TRANSFORMER_GROUP_EPSILON: f32 = 1e-6;
const LAYER_NORM_EPSILON: f32 = 1e-5;
const ADDITION_TIME_EMBED_DIM: usize = 256;

/// `attention_head_dim` per down/mid level, matching `unet/config.json` (num
/// heads, not width — head width is a constant 64 across all levels).
const LEVEL_HEADS: [usize; 3] = [5, 10, 20];
/// `transformer_layers_per_block` per down/mid level.
const LEVEL_DEPTH: [usize; 3] = [1, 2, 10];

pub fn predict_noise(
    weights: &SafeTensorFile,
    sample: &Tensor,
    timestep: f32,
    context: &Tensor,
    pooled_text_embed: &[f32],
    time_ids: [f32; 6],
) -> Result<Tensor> {
    let [batch, channels, height, width]: [usize; 4] = sample
        .shape()
        .try_into()
        .context("SDXL UNet sample must be NCHW")?;
    if channels != 4 || height < 8 || width < 8 || height % 8 != 0 || width % 8 != 0 {
        bail!(
            "SDXL UNet sample must be [B, 4, H, W] with H/W divisible by 8, got {:?}",
            sample.shape()
        );
    }
    if context.shape() != [batch, 77, 2048] {
        bail!(
            "SDXL UNet context must be [{batch}, 77, 2048], got {:?}",
            context.shape()
        );
    }
    if pooled_text_embed.len() != 1280 {
        bail!(
            "SDXL pooled text embedding must have 1280 elements, got {}",
            pooled_text_embed.len()
        );
    }

    let base_time = sd_unet::time_embedding(weights, timestep, batch)?;
    let augmentation = add_embedding(weights, pooled_text_embed, time_ids)?;
    let time = base_time.add(&augmentation)?;

    let mut hidden = sd_ops::conv2d(sample, weights, "conv_in", [1, 1])?;
    let mut skips = vec![hidden.clone()];

    for level in 0..3 {
        for layer in 0..2 {
            // Per-layer, not per-level: level 2's 10-deep transformer blocks alone
            // exceed this Adreno GPU's 2GB single-allocation cap (measured on a
            // real S24) if the whole level shares one stage.
            begin_weight_stage()?;
            hidden = sd_unet::resnet(
                weights,
                &hidden,
                &time,
                &format!("down_blocks.{level}.resnets.{layer}"),
            )?;
            if level > 0 {
                hidden = transformer(
                    weights,
                    &hidden,
                    context,
                    &format!("down_blocks.{level}.attentions.{layer}"),
                    LEVEL_DEPTH[level],
                    LEVEL_HEADS[level],
                )?;
            }
            skips.push(hidden.clone());
        }
        if level < 2 {
            hidden = sd_ops::conv2d_full(
                &hidden,
                weights,
                &format!("down_blocks.{level}.downsamplers.0.conv"),
                [2, 2],
                [1, 1],
            )?;
            skips.push(hidden.clone());
        }
    }

    begin_weight_stage()?;
    hidden = sd_unet::resnet(weights, &hidden, &time, "mid_block.resnets.0")?;
    hidden = transformer(
        weights,
        &hidden,
        context,
        "mid_block.attentions.0",
        LEVEL_DEPTH[2],
        LEVEL_HEADS[2],
    )?;
    begin_weight_stage()?;
    hidden = sd_unet::resnet(weights, &hidden, &time, "mid_block.resnets.1")?;

    for up_index in 0..3 {
        let level = 2 - up_index; // reversed: channels/depth/heads mirror the down path
        let has_attention = up_index < 2; // UpBlock2D (no attention) is the last level
        for layer in 0..3 {
            begin_weight_stage()?;
            let skip = skips.pop().with_context(|| {
                format!("UNet skip stack underflow at up level {up_index}.{layer}")
            })?;
            let joined = Tensor::concat_channels(&[&hidden, &skip])?;
            hidden = sd_unet::resnet(
                weights,
                &joined,
                &time,
                &format!("up_blocks.{up_index}.resnets.{layer}"),
            )?;
            if has_attention {
                hidden = transformer(
                    weights,
                    &hidden,
                    context,
                    &format!("up_blocks.{up_index}.attentions.{layer}"),
                    LEVEL_DEPTH[level],
                    LEVEL_HEADS[level],
                )?;
            }
        }
        if up_index < 2 {
            hidden = hidden.upsample_nearest2d([2, 2])?;
            hidden = sd_ops::conv2d(
                &hidden,
                weights,
                &format!("up_blocks.{up_index}.upsamplers.0.conv"),
                [1, 1],
            )?;
        }
    }
    if !skips.is_empty() {
        bail!("SDXL UNet left {} unused skip tensors", skips.len());
    }

    begin_weight_stage()?;
    hidden = sd_ops::group_norm(&hidden, weights, "conv_norm_out", GROUPS, RESNET_EPSILON)?;
    hidden = hidden.silu();
    sd_ops::conv2d(&hidden, weights, "conv_out", [1, 1])
}

/// The "text_time" micro-conditioning branch: pooled text embedding (1280) concatenated
/// with six sinusoidally-embedded time_ids (256 each = 1536) projected through a small
/// MLP, then added to the base timestep embedding before any ResNet block sees it.
fn add_embedding(
    weights: &SafeTensorFile,
    pooled_text_embed: &[f32],
    time_ids: [f32; 6],
) -> Result<Tensor> {
    let mut input = Vec::with_capacity(2816);
    input.extend_from_slice(pooled_text_embed);
    for value in time_ids {
        input.extend(sd_scheduler::timestep_embedding(
            value,
            ADDITION_TIME_EMBED_DIM,
        )?);
    }
    let embedding = Tensor::new(vec![1, 2816], input)?;
    let embedding = sd_ops::linear(&embedding, weights, "add_embedding.linear_1")?.silu();
    sd_ops::linear(&embedding, weights, "add_embedding.linear_2")
}

fn transformer(
    weights: &SafeTensorFile,
    input: &Tensor,
    context: &Tensor,
    prefix: &str,
    depth: usize,
    heads: usize,
) -> Result<Tensor> {
    let residual = input.clone();
    let normalized = sd_ops::group_norm(
        input,
        weights,
        &format!("{prefix}.norm"),
        GROUPS,
        TRANSFORMER_GROUP_EPSILON,
    )?;
    let [_, _, height, width]: [usize; 4] = normalized
        .shape()
        .try_into()
        .expect("group norm preserves NCHW rank");
    let mut sequence = sd_unet::nchw_to_sequence(&normalized)?;
    sequence = sd_ops::linear(&sequence, weights, &format!("{prefix}.proj_in"))?;

    for block_index in 0..depth {
        let block = format!("{prefix}.transformer_blocks.{block_index}");
        let norm = sd_ops::layer_norm(
            &sequence,
            weights,
            &format!("{block}.norm1"),
            LAYER_NORM_EPSILON,
        )?;
        let attended = attention(weights, &norm, &norm, &format!("{block}.attn1"), heads)?;
        sequence = sequence.add(&attended)?;

        let norm = sd_ops::layer_norm(
            &sequence,
            weights,
            &format!("{block}.norm2"),
            LAYER_NORM_EPSILON,
        )?;
        let attended = attention(weights, &norm, context, &format!("{block}.attn2"), heads)?;
        sequence = sequence.add(&attended)?;

        let norm = sd_ops::layer_norm(
            &sequence,
            weights,
            &format!("{block}.norm3"),
            LAYER_NORM_EPSILON,
        )?;
        let feed_forward = sd_unet::feed_forward(weights, &norm, &format!("{block}.ff"))?;
        sequence = sequence.add(&feed_forward)?;
    }

    sequence = sd_ops::linear(&sequence, weights, &format!("{prefix}.proj_out"))?;
    let hidden = sd_unet::sequence_to_nchw(&sequence, height, width)?;
    residual.add(&hidden)
}

fn attention(
    weights: &SafeTensorFile,
    query_input: &Tensor,
    key_value_input: &Tensor,
    prefix: &str,
    heads: usize,
) -> Result<Tensor> {
    if query_input.shape().len() != 3 || key_value_input.shape().len() != 3 {
        bail!("SDXL UNet attention inputs must have [B, sequence, channels] layout");
    }
    if query_input.shape()[0] != key_value_input.shape()[0] {
        bail!("SDXL UNet attention batch mismatch");
    }
    #[cfg(feature = "vulkan")]
    if crate::vulkan::sd_acceleration_requested() {
        return crate::vulkan::projected_attention(
            query_input,
            key_value_input,
            weights,
            prefix,
            heads,
        );
    }
    let query = sd_ops::linear(query_input, weights, &format!("{prefix}.to_q"))?;
    let key = sd_ops::linear(key_value_input, weights, &format!("{prefix}.to_k"))?;
    let value = sd_ops::linear(key_value_input, weights, &format!("{prefix}.to_v"))?;
    let query = sd_unet::split_heads(&query, heads)?;
    let key = sd_unet::split_heads(&key, heads)?;
    let value = sd_unet::split_heads(&value, heads)?;
    let attended = Tensor::attention(&query, &key, &value)?;
    let attended = sd_unet::merge_heads(&attended)?;
    sd_ops::linear(&attended, weights, &format!("{prefix}.to_out.0"))
}

/// Evict cached Vulkan weights between UNet stages; see sd_unet's identical helper.
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

/// Every named-tensor prefix that becomes its own staged-loading eviction boundary
/// above, in call order — mirrors `sd_unet::stage_prefixes` for SDXL's 3-level shape.
fn stage_prefixes() -> Vec<Vec<String>> {
    let mut stages = vec![vec![
        "conv_in".to_string(),
        "time_embedding".to_string(),
        "add_embedding".to_string(),
    ]];
    for level in 0..3 {
        for layer in 0..2 {
            stages.push(vec![
                format!("down_blocks.{level}.resnets.{layer}"),
                format!("down_blocks.{level}.attentions.{layer}"),
            ]);
        }
        if level < 2 {
            stages.push(vec![format!("down_blocks.{level}.downsamplers.0.conv")]);
        }
    }
    stages.push(vec![
        "mid_block.resnets.0".to_string(),
        "mid_block.attentions.0".to_string(),
    ]);
    stages.push(vec!["mid_block.resnets.1".to_string()]);
    for up_index in 0..3 {
        for layer in 0..3 {
            stages.push(vec![
                format!("up_blocks.{up_index}.resnets.{layer}"),
                format!("up_blocks.{up_index}.attentions.{layer}"),
            ]);
        }
        if up_index < 2 {
            stages.push(vec![format!("up_blocks.{up_index}.upsamplers.0.conv")]);
        }
    }
    stages.push(vec!["conv_norm_out".to_string(), "conv_out".to_string()]);
    stages
}

/// Vulkan weight-arena budget for staged loading, sized from this UNet's real
/// tensor bytes exactly like `sd_unet::staged_loading_budget`.
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
        bail!("could not size a staged-loading budget: no SDXL UNet tensors matched a known stage");
    }
    let budget_bytes = max_bytes + max_bytes / 4;
    let tensor_count_hint = max_count + max_count / 4 + 1;
    Ok((budget_bytes, tensor_count_hint))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_latent_shapes_that_break_the_skip_pyramid() {
        let sample = Tensor::zeros(vec![1, 4, 7, 8]).unwrap();
        let context = Tensor::zeros(vec![1, 77, 2048]).unwrap();
        let error = predict_noise(
            &fixture_shape_check_only(),
            &sample,
            0.0,
            &context,
            &[0.0; 1280],
            [0.0; 6],
        )
        .unwrap_err();
        assert!(error.to_string().contains("divisible by 8"));
    }

    #[test]
    fn rejects_wrong_pooled_embed_length() {
        let sample = Tensor::zeros(vec![1, 4, 8, 8]).unwrap();
        let context = Tensor::zeros(vec![1, 77, 2048]).unwrap();
        let error = predict_noise(
            &fixture_shape_check_only(),
            &sample,
            0.0,
            &context,
            &[0.0; 100],
            [0.0; 6],
        )
        .unwrap_err();
        assert!(error.to_string().contains("1280 elements"));
    }

    /// A `SafeTensorFile` is only reachable via `open`, which needs a real (non-empty)
    /// file. These tests fail on input validation before any tensor lookup happens, so
    /// a single throwaway tensor is enough — its content is never read.
    fn fixture_shape_check_only() -> SafeTensorFile {
        let header = r#"{"unused":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("quartz-sdxl-fixture-{nonce}.safetensors"));
        std::fs::write(&path, bytes).unwrap();
        SafeTensorFile::open(&path).unwrap()
    }
}
