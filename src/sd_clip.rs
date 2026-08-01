//! Quartz-owned CLIP text encoder for the Stable Diffusion 1.x conditioning path.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{
    safetensors::{DType, SafeTensorFile, TensorView},
    tensor::Tensor,
};

const SEQUENCE: usize = 77;
const HIDDEN: usize = 768;
const HEADS: usize = 12;
const HEAD_WIDTH: usize = HIDDEN / HEADS;
const LAYERS: usize = 12;
const EPSILON: f32 = 1e-5;

pub fn encode(weights: &SafeTensorFile, tokens: &[u32; SEQUENCE]) -> Result<Tensor> {
    let token_embedding = weights.view("text_model.embeddings.token_embedding.weight")?;
    let position_embedding = weights.view("text_model.embeddings.position_embedding.weight")?;
    require_view(
        token_embedding,
        "text_model.embeddings.token_embedding.weight",
        &[49_408, HIDDEN],
    )?;
    require_view(
        position_embedding,
        "text_model.embeddings.position_embedding.weight",
        &[SEQUENCE, HIDDEN],
    )?;

    let mut embedding_data = vec![0.0f32; SEQUENCE * HIDDEN];
    embedding_data
        .par_chunks_mut(HIDDEN)
        .enumerate()
        .try_for_each(|(position, row)| -> Result<()> {
            let token = usize::try_from(tokens[position]).context("token ID does not fit usize")?;
            if token >= 49_408 {
                bail!("CLIP token ID {token} is outside the SD1.5 vocabulary");
            }
            for hidden in 0..HIDDEN {
                row[hidden] = token_embedding.value(token * HIDDEN + hidden)
                    + position_embedding.value(position * HIDDEN + hidden);
            }
            Ok(())
        })?;
    let mut hidden_states = Tensor::new(vec![1, SEQUENCE, HIDDEN], embedding_data)?;

    for layer in 0..LAYERS {
        let prefix = format!("text_model.encoder.layers.{layer}");
        let residual = hidden_states.clone();
        let normalized = layer_norm(&hidden_states, weights, &format!("{prefix}.layer_norm1"))?;
        let query = split_heads(&linear(
            &normalized,
            weights,
            &format!("{prefix}.self_attn.q_proj"),
        )?)?;
        let key = split_heads(&linear(
            &normalized,
            weights,
            &format!("{prefix}.self_attn.k_proj"),
        )?)?;
        let value = split_heads(&linear(
            &normalized,
            weights,
            &format!("{prefix}.self_attn.v_proj"),
        )?)?;
        let attention = Tensor::attention_causal(&query, &key, &value)?;
        let attention = merge_heads(&attention)?;
        let attention = linear(&attention, weights, &format!("{prefix}.self_attn.out_proj"))?;
        hidden_states = residual.add(&attention)?;

        let residual = hidden_states.clone();
        let normalized = layer_norm(&hidden_states, weights, &format!("{prefix}.layer_norm2"))?;
        let intermediate = linear(&normalized, weights, &format!("{prefix}.mlp.fc1"))?;
        let intermediate = intermediate.quick_gelu();
        let output = linear(&intermediate, weights, &format!("{prefix}.mlp.fc2"))?;
        hidden_states = residual.add(&output)?;
    }

    layer_norm(&hidden_states, weights, "text_model.final_layer_norm")
}

fn layer_norm(input: &Tensor, weights: &SafeTensorFile, prefix: &str) -> Result<Tensor> {
    let weight = load_tensor(weights, &format!("{prefix}.weight"))?;
    let bias = load_tensor(weights, &format!("{prefix}.bias"))?;
    input.layer_norm(&weight, &bias, EPSILON)
}

fn load_tensor(weights: &SafeTensorFile, name: &str) -> Result<Tensor> {
    let view = weights.view(name)?;
    let data = (0..view.len()).map(|index| view.value(index)).collect();
    Tensor::new(view.shape.to_vec(), data)
}

/// Project directly from the mapped FP16 model so large matrices never acquire
/// a second resident f32 copy.
fn linear(input: &Tensor, weights: &SafeTensorFile, prefix: &str) -> Result<Tensor> {
    let weight_name = format!("{prefix}.weight");
    let bias_name = format!("{prefix}.bias");
    let weight = weights.view(&weight_name)?;
    let bias = weights.view(&bias_name)?;
    #[cfg(feature = "vulkan")]
    if crate::vulkan::sd_acceleration_requested() {
        return crate::vulkan::linear(input, weights.mapped(&weight_name)?, Some(bias));
    }
    if weight.dtype != DType::F16 || weight.shape.len() != 2 {
        bail!("{weight_name} must be a rank-2 F16 tensor");
    }
    let out_features = weight.shape[0];
    let in_features = weight.shape[1];
    if input.shape().last().copied() != Some(in_features) {
        bail!(
            "{prefix} input width {:?} does not match mapped weight width {in_features}",
            input.shape().last()
        );
    }
    require_view(bias, &bias_name, &[out_features])?;

    let rows = input.len() / in_features;
    let mut output = vec![0.0f32; rows * out_features];
    output
        .par_chunks_mut(out_features)
        .enumerate()
        .for_each(|(row_index, output_row)| {
            let input_row = &input.data()[row_index * in_features..(row_index + 1) * in_features];
            for (out, value) in output_row.iter_mut().enumerate() {
                let weight_offset = out * in_features;
                let mut sum = bias.value(out);
                for (input_value, weight_index) in input_row
                    .iter()
                    .zip(weight_offset..weight_offset + in_features)
                {
                    sum += *input_value * weight.value(weight_index);
                }
                *value = sum;
            }
        });
    let mut output_shape = input.shape().to_vec();
    *output_shape
        .last_mut()
        .expect("input has a final dimension") = out_features;
    Tensor::new(output_shape, output)
}

fn require_view(view: TensorView<'_>, name: &str, shape: &[usize]) -> Result<()> {
    if view.dtype != DType::F16 {
        bail!("{name} must be F16, got {:?}", view.dtype);
    }
    if view.shape != shape {
        bail!("{name} shape is {:?}; expected {shape:?}", view.shape);
    }
    Ok(())
}

fn split_heads(input: &Tensor) -> Result<Tensor> {
    if input.shape() != [1, SEQUENCE, HIDDEN] {
        bail!(
            "CLIP head split requires [1, 77, 768], got {:?}",
            input.shape()
        );
    }
    let mut output = vec![0.0f32; input.len()];
    for head in 0..HEADS {
        for position in 0..SEQUENCE {
            for dimension in 0..HEAD_WIDTH {
                output[(head * SEQUENCE + position) * HEAD_WIDTH + dimension] =
                    input.data()[position * HIDDEN + head * HEAD_WIDTH + dimension];
            }
        }
    }
    Tensor::new(vec![1, HEADS, SEQUENCE, HEAD_WIDTH], output)
}

fn merge_heads(input: &Tensor) -> Result<Tensor> {
    if input.shape() != [1, HEADS, SEQUENCE, HEAD_WIDTH] {
        bail!(
            "CLIP head merge requires [1, 12, 77, 64], got {:?}",
            input.shape()
        );
    }
    let mut output = vec![0.0f32; input.len()];
    for head in 0..HEADS {
        for position in 0..SEQUENCE {
            for dimension in 0..HEAD_WIDTH {
                output[position * HIDDEN + head * HEAD_WIDTH + dimension] =
                    input.data()[(head * SEQUENCE + position) * HEAD_WIDTH + dimension];
            }
        }
    }
    Tensor::new(vec![1, SEQUENCE, HIDDEN], output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_split_and_merge_round_trip() {
        let input = Tensor::new(
            vec![1, SEQUENCE, HIDDEN],
            (0..SEQUENCE * HIDDEN).map(|value| value as f32).collect(),
        )
        .unwrap();
        assert_eq!(merge_heads(&split_heads(&input).unwrap()).unwrap(), input);
    }

    #[test]
    #[ignore = "requires QUARTZ_SD15_MODEL_DIR pointing to the official SD1.5 FP16 pack"]
    fn matches_official_clip_golden_output() {
        let root = std::env::var("QUARTZ_SD15_MODEL_DIR").unwrap();
        let pack = crate::sd15::Sd15Pack::open(root).unwrap();
        let output = pack
            .encode_prompt("A photo of a lion in the wild, ultra realistic")
            .unwrap();
        let expected = [
            // Generated once with Transformers CLIPTextModel after promoting the
            // published FP16 weights to f32, matching Quartz accumulation.
            -0.38847336,
            0.022983685,
            -0.05208571,
            -0.18408374,
            -0.027318122,
            -0.3356097,
            -0.017577872,
            -0.18697253,
            0.18771185,
            -0.09064246,
            -0.22800781,
            -0.14970362,
            -0.07405163,
            -0.35468072,
            0.113429874,
            -0.10164956,
        ];
        for (index, expected) in expected.into_iter().enumerate() {
            let actual = output.data()[index];
            assert!(
                (actual - expected).abs() < 2e-3,
                "CLIP output {index}: {actual} != {expected}"
            );
        }
    }
}
