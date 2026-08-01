//! Quartz-owned Stable Diffusion VAE decoder.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{safetensors::SafeTensorFile, sd_ops, tensor::Tensor};

const GROUPS: usize = 32;
const EPSILON: f32 = 1e-6;
// SDXL's 128x128 latent cannot be decoded as one VAE attention map: the
// 512-wide, 16K-token operation is both too large for the bounded Vulkan
// kernel and too long for mobile GPU watchdogs. Keep each local attention map
// at or below 40x40 while retaining context around every stitched core.
const TILE_CORE: usize = 32;
const TILE_CONTEXT: usize = 4;
pub const LATENT_SCALE: f32 = 0.18215;

/// Decode denoised SD1.5 latents to an NCHW RGB tensor. The public diffusion
/// path supplies `[1, 4, 64, 64]`; smaller spatial shapes remain valid for
/// kernel/reference tests because the decoder is fully convolutional.
pub fn decode(weights: &SafeTensorFile, latents: &Tensor) -> Result<Tensor> {
    require_latents(latents)?;
    let mut scaled = latents.clone();
    for value in scaled.data_mut() {
        *value /= LATENT_SCALE;
    }
    decode_unscaled(weights, &scaled)
}

pub fn decode_unscaled(weights: &SafeTensorFile, latents: &Tensor) -> Result<Tensor> {
    require_latents(latents)?;
    if latents.shape()[2] > TILE_CORE + TILE_CONTEXT * 2
        || latents.shape()[3] > TILE_CORE + TILE_CONTEXT * 2
    {
        return decode_unscaled_tiled(weights, latents);
    }
    decode_unscaled_full(weights, latents)
}

fn decode_unscaled_full(weights: &SafeTensorFile, latents: &Tensor) -> Result<Tensor> {
    let mut sample = sd_ops::conv2d(latents, weights, "post_quant_conv", [0, 0])?;
    sample = sd_ops::conv2d(&sample, weights, "decoder.conv_in", [1, 1])?;

    sample = resnet(weights, &sample, "decoder.mid_block.resnets.0")?;
    sample = attention(weights, &sample, "decoder.mid_block.attentions.0")?;
    sample = resnet(weights, &sample, "decoder.mid_block.resnets.1")?;

    for block in 0..4 {
        for layer in 0..3 {
            sample = resnet(
                weights,
                &sample,
                &format!("decoder.up_blocks.{block}.resnets.{layer}"),
            )?;
        }
        if block < 3 {
            sample = sample.upsample_nearest2d([2, 2])?;
            sample = sd_ops::conv2d(
                &sample,
                weights,
                &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                [1, 1],
            )?;
        }
    }

    sample = sd_ops::group_norm(&sample, weights, "decoder.conv_norm_out", GROUPS, EPSILON)?;
    sample = sample.silu();
    sd_ops::conv2d(&sample, weights, "decoder.conv_out", [1, 1])
}

fn decode_unscaled_tiled(weights: &SafeTensorFile, latents: &Tensor) -> Result<Tensor> {
    let latent_height = latents.shape()[2];
    let latent_width = latents.shape()[3];
    let vertical = tile_windows(latent_height);
    let horizontal = tile_windows(latent_width);
    let output_height = latent_height * 8;
    let output_width = latent_width * 8;
    let mut output = vec![0.0; 3 * output_height * output_width];
    let mut output_weights = vec![0.0; output_height * output_width];
    let tile_count = vertical.len() * horizontal.len();
    let mut tile_number = 0;

    for &(core_top, core_bottom, tile_top, tile_bottom) in &vertical {
        for &(core_left, core_right, tile_left, tile_right) in &horizontal {
            tile_number += 1;
            eprintln!(
                "VAE tile {tile_number}/{tile_count}: core=({core_top}..{core_bottom}, {core_left}..{core_right}) input=({tile_top}..{tile_bottom}, {tile_left}..{tile_right})"
            );
            let tile = extract_tile(latents, tile_top, tile_bottom, tile_left, tile_right)?;
            let decoded = decode_unscaled_full(weights, &tile)?;
            let tile_output_height = (tile_bottom - tile_top) * 8;
            let tile_output_width = (tile_right - tile_left) * 8;
            if decoded.shape() != [1, 3, tile_output_height, tile_output_width] {
                bail!(
                    "VAE tile decoded to {:?}, expected [1, 3, {tile_output_height}, {tile_output_width}]",
                    decoded.shape()
                );
            }
            let output_tile_top = tile_top * 8;
            let output_tile_left = tile_left * 8;
            let output_core_top = core_top * 8;
            let output_core_bottom = core_bottom * 8;
            let output_core_left = core_left * 8;
            let output_core_right = core_right * 8;
            let output_tile_bottom = tile_bottom * 8;
            let output_tile_right = tile_right * 8;
            for tile_row in 0..tile_output_height {
                let output_row = output_tile_top + tile_row;
                let vertical_weight = tile_axis_weight(
                    output_row,
                    output_tile_top,
                    output_core_top,
                    output_core_bottom,
                    output_tile_bottom,
                );
                for tile_column in 0..tile_output_width {
                    let output_column = output_tile_left + tile_column;
                    let horizontal_weight = tile_axis_weight(
                        output_column,
                        output_tile_left,
                        output_core_left,
                        output_core_right,
                        output_tile_right,
                    );
                    let weight = vertical_weight * horizontal_weight;
                    let output_pixel = output_row * output_width + output_column;
                    output_weights[output_pixel] += weight;
                    let tile_pixel = tile_row * tile_output_width + tile_column;
                    for channel in 0..3 {
                        output[channel * output_height * output_width + output_pixel] += decoded
                            .data()[channel * tile_output_height * tile_output_width + tile_pixel]
                            * weight;
                    }
                }
            }
        }
    }
    for (pixel, &weight) in output_weights.iter().enumerate() {
        if weight <= 0.0 || !weight.is_finite() {
            bail!("VAE tile blending left output pixel {pixel} uncovered");
        }
        for channel in 0..3 {
            output[channel * output_height * output_width + pixel] /= weight;
        }
    }
    Tensor::new(vec![1, 3, output_height, output_width], output)
}

fn tile_axis_weight(
    position: usize,
    tile_start: usize,
    core_start: usize,
    core_end: usize,
    tile_end: usize,
) -> f32 {
    if position < core_start && tile_start < core_start {
        return (position - tile_start) as f32 / (core_start - tile_start) as f32;
    }
    if position >= core_end && core_end < tile_end {
        return (tile_end - position) as f32 / (tile_end - core_end) as f32;
    }
    1.0
}

fn tile_windows(length: usize) -> Vec<(usize, usize, usize, usize)> {
    (0..length)
        .step_by(TILE_CORE)
        .map(|core_start| {
            let core_end = (core_start + TILE_CORE).min(length);
            let tile_start = core_start.saturating_sub(TILE_CONTEXT);
            let tile_end = (core_end + TILE_CONTEXT).min(length);
            (core_start, core_end, tile_start, tile_end)
        })
        .collect()
}

fn extract_tile(
    input: &Tensor,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
) -> Result<Tensor> {
    let channels = input.shape()[1];
    let input_height = input.shape()[2];
    let input_width = input.shape()[3];
    let tile_height = bottom - top;
    let tile_width = right - left;
    let mut data = Vec::with_capacity(channels * tile_height * tile_width);
    for channel in 0..channels {
        for row in top..bottom {
            let start = (channel * input_height + row) * input_width + left;
            data.extend_from_slice(&input.data()[start..start + tile_width]);
        }
    }
    Tensor::new(vec![1, channels, tile_height, tile_width], data)
}

fn require_latents(latents: &Tensor) -> Result<()> {
    if latents.shape().len() != 4 || latents.shape()[0] != 1 || latents.shape()[1] != 4 {
        bail!(
            "Stable Diffusion VAE latents must have shape [1, 4, H, W], got {:?}",
            latents.shape()
        );
    }
    Ok(())
}

fn resnet(weights: &SafeTensorFile, input: &Tensor, prefix: &str) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    if crate::vulkan::sd_acceleration_requested() {
        return crate::vulkan::resnet(input, None, weights, prefix, GROUPS, EPSILON);
    }
    let residual = if weights
        .info(&format!("{prefix}.conv_shortcut.weight"))
        .is_some()
    {
        sd_ops::conv2d(input, weights, &format!("{prefix}.conv_shortcut"), [0, 0])?
    } else {
        input.clone()
    };
    let mut hidden =
        sd_ops::group_norm(input, weights, &format!("{prefix}.norm1"), GROUPS, EPSILON)?;
    hidden = hidden.silu();
    hidden = sd_ops::conv2d(&hidden, weights, &format!("{prefix}.conv1"), [1, 1])?;
    hidden = sd_ops::group_norm(
        &hidden,
        weights,
        &format!("{prefix}.norm2"),
        GROUPS,
        EPSILON,
    )?;
    hidden = hidden.silu();
    hidden = sd_ops::conv2d(&hidden, weights, &format!("{prefix}.conv2"), [1, 1])?;
    residual.add(&hidden)
}

fn attention(weights: &SafeTensorFile, input: &Tensor, prefix: &str) -> Result<Tensor> {
    let shape: [usize; 4] = input
        .shape()
        .try_into()
        .context("VAE attention input must be NCHW")?;
    let [batch, channels, height, width] = shape;
    let positions = height
        .checked_mul(width)
        .context("VAE attention position count overflow")?;
    let normalized = sd_ops::group_norm(
        input,
        weights,
        &format!("{prefix}.group_norm"),
        GROUPS,
        EPSILON,
    )?;
    let sequence = nchw_to_sequence(&normalized)?;
    let query = sd_ops::linear(&sequence, weights, &format!("{prefix}.to_q"))?;
    let key = sd_ops::linear(&sequence, weights, &format!("{prefix}.to_k"))?;
    let value = sd_ops::linear(&sequence, weights, &format!("{prefix}.to_v"))?;
    let query = Tensor::new(vec![batch, 1, positions, channels], query.data().to_vec())?;
    let key = Tensor::new(vec![batch, 1, positions, channels], key.data().to_vec())?;
    let value = Tensor::new(vec![batch, 1, positions, channels], value.data().to_vec())?;
    let attended = Tensor::attention(&query, &key, &value)?;
    let attended = Tensor::new(vec![batch, positions, channels], attended.data().to_vec())?;
    let projected = sd_ops::linear(&attended, weights, &format!("{prefix}.to_out.0"))?;
    let projected = sequence_to_nchw(&projected, height, width)?;
    input.add(&projected)
}

fn nchw_to_sequence(input: &Tensor) -> Result<Tensor> {
    let [batch, channels, height, width]: [usize; 4] = input
        .shape()
        .try_into()
        .context("NCHW-to-sequence requires rank 4")?;
    let positions = height * width;
    let mut output = vec![0.0; batch * positions * channels];
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

fn sequence_to_nchw(input: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    if input.shape().len() != 3 || input.shape()[1] != height * width {
        bail!(
            "sequence-to-NCHW shape {:?} is incompatible with {height}x{width}",
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
    fn sequence_layout_round_trips_nchw() {
        let input = Tensor::new(vec![1, 2, 2, 2], (0..8).map(|x| x as f32).collect()).unwrap();
        let sequence = nchw_to_sequence(&input).unwrap();
        assert_eq!(sequence.shape(), &[1, 4, 2]);
        assert_eq!(sequence.data(), &[0., 4., 1., 5., 2., 6., 3., 7.]);
        assert_eq!(sequence_to_nchw(&sequence, 2, 2).unwrap(), input);
    }

    #[test]
    fn rejects_invalid_latents() {
        let input = Tensor::zeros(vec![1, 3, 8, 8]).unwrap();
        assert!(
            require_latents(&input)
                .unwrap_err()
                .to_string()
                .contains("[1, 4")
        );
    }

    #[test]
    fn tiled_decode_windows_cover_large_latent_with_bounded_context() {
        let windows = tile_windows(128);
        assert_eq!(
            windows,
            vec![
                (0, 32, 0, 36),
                (32, 64, 28, 68),
                (64, 96, 60, 100),
                (96, 128, 92, 128),
            ]
        );
        assert!(windows.iter().all(|window| window.3 - window.2 <= 40));
        for pair in windows.windows(2) {
            assert_eq!(pair[0].1, pair[1].0);
        }
    }

    #[test]
    fn extracts_nchw_tile_without_reordering_channels() {
        let input = Tensor::new(vec![1, 4, 3, 4], (0..48).map(|x| x as f32).collect()).unwrap();
        let tile = extract_tile(&input, 1, 3, 1, 4).unwrap();
        assert_eq!(tile.shape(), &[1, 4, 2, 3]);
        assert_eq!(
            tile.data(),
            &[
                5., 6., 7., 9., 10., 11., 17., 18., 19., 21., 22., 23., 29., 30., 31., 33., 34.,
                35., 41., 42., 43., 45., 46., 47.,
            ]
        );
    }

    #[test]
    fn tile_blend_weight_fades_only_inside_context() {
        assert_eq!(tile_axis_weight(20, 20, 24, 48, 52), 0.0);
        assert_eq!(tile_axis_weight(22, 20, 24, 48, 52), 0.5);
        assert_eq!(tile_axis_weight(24, 20, 24, 48, 52), 1.0);
        assert_eq!(tile_axis_weight(47, 20, 24, 48, 52), 1.0);
        assert_eq!(tile_axis_weight(50, 20, 24, 48, 52), 0.5);
        assert_eq!(tile_axis_weight(51, 20, 24, 48, 52), 0.25);
        assert_eq!(tile_axis_weight(0, 0, 0, 24, 28), 1.0);
    }

    #[test]
    #[ignore = "requires QUARTZ_SD15_MODEL_DIR pointing to the official SD1.5 FP16 pack"]
    fn matches_official_vae_golden_output() {
        let root = std::env::var("QUARTZ_SD15_MODEL_DIR").unwrap();
        let pack = crate::sd15::Sd15Pack::open(root).unwrap();
        let latent = Tensor::new(vec![1, 4, 1, 1], vec![0.0, 0.1, -0.2, 0.3]).unwrap();
        let output = pack.decode_unscaled_latents(&latent).unwrap();
        let expected = [
            0.12734652, 0.9618441, 0.09397145, 0.1157567, 0.35260347, 0.22421816, 0.24894658,
            0.14262436, 0.13596976, 0.34138137, 0.23534852, 0.16338205, 0.5154935, 0.3543424,
            0.24537653, 0.24747209,
        ];
        for (index, expected) in expected.into_iter().enumerate() {
            let actual = output.data()[index];
            assert!(
                (actual - expected).abs() < 5e-5,
                "VAE output {index}: {actual} != {expected}"
            );
        }
    }
}
