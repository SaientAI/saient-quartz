//! SDXL's dual CLIP text encoder.
//!
//! SDXL conditions the UNet on two encoders concatenated along the hidden
//! dimension: CLIP-L (`text_encoder`, 768-wide, quick_gelu) and OpenCLIP-bigG
//! (`text_encoder_2`, 1280-wide, gelu). Both contribute their *penultimate*
//! hidden states (the output of the second-to-last transformer layer, before
//! any final layer norm — matching diffusers' `hidden_states[-2]`), which are
//! concatenated into the 2048-wide cross-attention context. `text_encoder_2`
//! additionally contributes a pooled, projected 1280-wide vector (from its
//! *full* depth + final layer norm, pooled at the EOS token position) used in
//! the UNet's micro-conditioning branch.
//!
//! Known deviation from the reference: `ClipTokenizer` always pads with EOS,
//! while `tokenizer_2`'s HF config nominally pads with `"!"` (id 0). Only
//! padding-position embeddings differ; content tokens are unaffected.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::{
    safetensors::{DType, SafeTensorFile, TensorView},
    tensor::Tensor,
};

const SEQUENCE: usize = 77;
const EPSILON: f32 = 1e-5;

pub struct ClipEncoderSpec {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub vocab: usize,
    pub gelu_exact: bool,
}

pub const CLIP_L: ClipEncoderSpec = ClipEncoderSpec {
    hidden: 768,
    heads: 12,
    layers: 12,
    vocab: 49_408,
    gelu_exact: false, // quick_gelu
};

pub const CLIP_BIGG: ClipEncoderSpec = ClipEncoderSpec {
    hidden: 1280,
    heads: 20,
    layers: 32,
    vocab: 49_408,
    gelu_exact: true, // gelu
};

/// Penultimate hidden states only (encoder 1's contribution to the concatenated context).
pub fn encode_penultimate(
    weights: &SafeTensorFile,
    tokens: &[u32; SEQUENCE],
    spec: &ClipEncoderSpec,
) -> Result<Tensor> {
    let (penultimate, _) = encode_layers(weights, tokens, spec)?;
    Ok(penultimate)
}

/// Penultimate hidden states plus the pooled+projected embedding (encoder 2's
/// contribution: both the context slice and the micro-conditioning vector).
pub fn encode_with_pool(
    weights: &SafeTensorFile,
    tokens: &[u32; SEQUENCE],
    spec: &ClipEncoderSpec,
) -> Result<(Tensor, Vec<f32>)> {
    let (penultimate, last_layer_input) = encode_layers(weights, tokens, spec)?;
    let final_prefix = format!("text_model.encoder.layers.{}", spec.layers - 1);
    let after_final_layer = encoder_layer(weights, &last_layer_input, &final_prefix, spec)?;
    let last_hidden_state = layer_norm(&after_final_layer, weights, "text_model.final_layer_norm")?;

    // CLIP's EOS id is fixed by the shared vocabulary (49407); SD1.5's tokenizer
    // already asserts this at load time, so it's safe to use as a plain constant here.
    const EOS_ID: u32 = 49_407;
    let eos_position = tokens
        .iter()
        .position(|&token| token == EOS_ID)
        .unwrap_or(SEQUENCE - 1);
    let pooled_raw = last_hidden_state.data()
        [eos_position * spec.hidden..(eos_position + 1) * spec.hidden]
        .to_vec();
    let pooled_raw = Tensor::new(vec![1, spec.hidden], pooled_raw)?;
    let projection = load_tensor(weights, "text_projection.weight")?;
    let pooled = pooled_raw.linear(&projection, None)?;
    Ok((penultimate, pooled.data().to_vec()))
}

/// Run embeddings + `layers - 1` transformer blocks. Returns (penultimate hidden
/// states, the same tensor — the input the caller needs to run one more layer
/// from, if it wants the full-depth pass too).
fn encode_layers(
    weights: &SafeTensorFile,
    tokens: &[u32; SEQUENCE],
    spec: &ClipEncoderSpec,
) -> Result<(Tensor, Tensor)> {
    let hidden = spec.hidden;
    let token_embedding = weights.view("text_model.embeddings.token_embedding.weight")?;
    let position_embedding = weights.view("text_model.embeddings.position_embedding.weight")?;
    require_view(
        token_embedding,
        "text_model.embeddings.token_embedding.weight",
        &[spec.vocab, hidden],
    )?;
    require_view(
        position_embedding,
        "text_model.embeddings.position_embedding.weight",
        &[SEQUENCE, hidden],
    )?;

    let mut embedding_data = vec![0.0f32; SEQUENCE * hidden];
    embedding_data
        .par_chunks_mut(hidden)
        .enumerate()
        .try_for_each(|(position, row)| -> Result<()> {
            let token = usize::try_from(tokens[position]).context("token ID does not fit usize")?;
            if token >= spec.vocab {
                bail!(
                    "CLIP token ID {token} is outside the {}-entry vocabulary",
                    spec.vocab
                );
            }
            for element in 0..hidden {
                row[element] = token_embedding.value(token * hidden + element)
                    + position_embedding.value(position * hidden + element);
            }
            Ok(())
        })?;
    let mut hidden_states = Tensor::new(vec![1, SEQUENCE, hidden], embedding_data)?;

    if spec.layers == 0 {
        bail!("CLIP encoder spec must have at least one layer");
    }
    for layer in 0..spec.layers - 1 {
        let prefix = format!("text_model.encoder.layers.{layer}");
        hidden_states = encoder_layer(weights, &hidden_states, &prefix, spec)?;
    }
    Ok((hidden_states.clone(), hidden_states))
}

fn encoder_layer(
    weights: &SafeTensorFile,
    hidden_states: &Tensor,
    prefix: &str,
    spec: &ClipEncoderSpec,
) -> Result<Tensor> {
    let residual = hidden_states.clone();
    let normalized = layer_norm(hidden_states, weights, &format!("{prefix}.layer_norm1"))?;
    let query = split_heads(
        &linear(&normalized, weights, &format!("{prefix}.self_attn.q_proj"))?,
        spec,
    )?;
    let key = split_heads(
        &linear(&normalized, weights, &format!("{prefix}.self_attn.k_proj"))?,
        spec,
    )?;
    let value = split_heads(
        &linear(&normalized, weights, &format!("{prefix}.self_attn.v_proj"))?,
        spec,
    )?;
    let attention = Tensor::attention_causal(&query, &key, &value)?;
    let attention = merge_heads(&attention, spec)?;
    let attention = linear(&attention, weights, &format!("{prefix}.self_attn.out_proj"))?;
    let hidden_states = residual.add(&attention)?;

    let residual = hidden_states.clone();
    let normalized = layer_norm(&hidden_states, weights, &format!("{prefix}.layer_norm2"))?;
    let intermediate = linear(&normalized, weights, &format!("{prefix}.mlp.fc1"))?;
    let intermediate = if spec.gelu_exact {
        intermediate.gelu()
    } else {
        intermediate.quick_gelu()
    };
    let output = linear(&intermediate, weights, &format!("{prefix}.mlp.fc2"))?;
    residual.add(&output)
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

fn split_heads(input: &Tensor, spec: &ClipEncoderSpec) -> Result<Tensor> {
    if input.shape() != [1, SEQUENCE, spec.hidden] {
        bail!(
            "CLIP head split requires [1, 77, {}], got {:?}",
            spec.hidden,
            input.shape()
        );
    }
    let width = spec.hidden / spec.heads;
    let mut output = vec![0.0f32; input.len()];
    for head in 0..spec.heads {
        for position in 0..SEQUENCE {
            for dimension in 0..width {
                output[(head * SEQUENCE + position) * width + dimension] =
                    input.data()[position * spec.hidden + head * width + dimension];
            }
        }
    }
    Tensor::new(vec![1, spec.heads, SEQUENCE, width], output)
}

fn merge_heads(input: &Tensor, spec: &ClipEncoderSpec) -> Result<Tensor> {
    let width = spec.hidden / spec.heads;
    if input.shape() != [1, spec.heads, SEQUENCE, width] {
        bail!(
            "CLIP head merge requires [1, {}, 77, {width}], got {:?}",
            spec.heads,
            input.shape()
        );
    }
    let mut output = vec![0.0f32; input.len()];
    for head in 0..spec.heads {
        for position in 0..SEQUENCE {
            for dimension in 0..width {
                output[position * spec.hidden + head * width + dimension] =
                    input.data()[(head * SEQUENCE + position) * width + dimension];
            }
        }
    }
    Tensor::new(vec![1, SEQUENCE, spec.hidden], output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-validates sdxl_clip's independently-written transformer machinery
    /// (attention, layer norm, quick_gelu, head split/merge) against sd_clip's
    /// golden-tested SD1.5 output. SD1.5's CLIP-L text encoder has the exact
    /// architecture `CLIP_L` describes (768/12L/quick_gelu), so running
    /// sdxl_clip's full-depth + final_layer_norm path (skipping only the
    /// projection step, which SD1.5's plain CLIPTextModel doesn't have) over
    /// the same weights and prompt should reproduce sd_clip's golden numbers.
    #[test]
    fn sdxl_clip_full_depth_matches_the_sd15_golden_output() {
        let Ok(root) = std::env::var("QUARTZ_SD15_MODEL_DIR") else {
            eprintln!("skipping: QUARTZ_SD15_MODEL_DIR not set");
            return;
        };
        let root = std::path::PathBuf::from(root);
        let weights =
            SafeTensorFile::open(root.join("text_encoder/model.fp16.safetensors")).unwrap();
        let tokenizer = crate::clip_tokenizer::ClipTokenizer::from_files(
            root.join("tokenizer/vocab.json"),
            root.join("tokenizer/merges.txt"),
        )
        .unwrap();
        let tokens = tokenizer.encode("A photo of a lion in the wild, ultra realistic");

        let (_, last_layer_input) = encode_layers(&weights, &tokens, &CLIP_L).unwrap();
        let final_prefix = format!("text_model.encoder.layers.{}", CLIP_L.layers - 1);
        let after_final_layer =
            encoder_layer(&weights, &last_layer_input, &final_prefix, &CLIP_L).unwrap();
        let last_hidden_state =
            layer_norm(&after_final_layer, &weights, "text_model.final_layer_norm").unwrap();

        let expected = [
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
            let actual = last_hidden_state.data()[index];
            assert!(
                (actual - expected).abs() < 2e-3,
                "sdxl_clip full-depth output {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn head_split_and_merge_round_trip_for_both_encoders() {
        for spec in [&CLIP_L, &CLIP_BIGG] {
            let input = Tensor::new(
                vec![1, SEQUENCE, spec.hidden],
                (0..SEQUENCE * spec.hidden)
                    .map(|value| value as f32)
                    .collect(),
            )
            .unwrap();
            assert_eq!(
                merge_heads(&split_heads(&input, spec).unwrap(), spec).unwrap(),
                input
            );
        }
    }
}
