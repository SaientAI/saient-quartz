//! SDXL base model-pack contract for Quartz.
//!
//! Same philosophy as sd15.rs: Quartz owns parsing, validation, tokenization,
//! scheduling, and execution; the pack directory is treated as untrusted
//! input data. Validation here checks the load-bearing config fields (the
//! ones the fixed sdxl_unet.rs/sdxl_clip.rs graphs assume) rather than
//! SD1.5's exhaustive per-tensor manifest — SDXL's real tensor names weren't
//! available to hardcode a manifest against at implementation time.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::{
    clip_tokenizer::ClipTokenizer,
    safetensors::{DType, SafeTensorFile},
    sd15::{
        expect_bool, expect_f64, expect_string, expect_u64, expect_u64_array, read_json,
        required_file,
    },
    sdxl_clip::{self, CLIP_BIGG, CLIP_L},
};

pub struct SdxlPack {
    root: PathBuf,
    text_encoder: SafeTensorFile,
    text_encoder_2: SafeTensorFile,
    unet: SafeTensorFile,
    vae: SafeTensorFile,
    tokenizer: ClipTokenizer,
    tokenizer_2: ClipTokenizer,
}

impl SdxlPack {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let requested_root = root.as_ref();
        let root = requested_root.canonicalize().with_context(|| {
            format!(
                "cannot resolve model directory {}",
                requested_root.display()
            )
        })?;
        if !root.is_dir() {
            bail!("model root is not a directory: {}", root.display());
        }

        let model_index = read_json(&required_file(&root, "model_index.json")?)?;
        validate_model_index(&model_index)?;
        validate_clip_config(
            &read_json(&required_file(&root, "text_encoder/config.json")?)?,
            &CLIP_L,
        )?;
        validate_clip_config(
            &read_json(&required_file(&root, "text_encoder_2/config.json")?)?,
            &CLIP_BIGG,
        )?;
        validate_unet_config(&read_json(&required_file(&root, "unet/config.json")?)?)?;
        validate_vae_config(&read_json(&required_file(&root, "vae/config.json")?)?)?;
        validate_scheduler_config(&read_json(&required_file(
            &root,
            "scheduler/scheduler_config.json",
        )?)?)?;

        let tokenizer = ClipTokenizer::from_files(
            required_file(&root, "tokenizer/vocab.json")?,
            required_file(&root, "tokenizer/merges.txt")?,
        )
        .context("invalid SDXL tokenizer")?;
        let tokenizer_2 = ClipTokenizer::from_files(
            required_file(&root, "tokenizer_2/vocab.json")?,
            required_file(&root, "tokenizer_2/merges.txt")?,
        )
        .context("invalid SDXL tokenizer_2")?;
        for (label, tokenizer) in [("tokenizer", &tokenizer), ("tokenizer_2", &tokenizer_2)] {
            let empty_prompt = tokenizer.encode("");
            if empty_prompt[0] != 49_406 || empty_prompt[1] != 49_407 {
                bail!(
                    "{label} special-token IDs are {}, {}; SDXL requires 49406, 49407",
                    empty_prompt[0],
                    empty_prompt[1]
                );
            }
        }

        let text_encoder =
            SafeTensorFile::open(required_file(&root, "text_encoder/model.fp16.safetensors")?)
                .context("cannot load SDXL text encoder")?;
        let text_encoder_2 = SafeTensorFile::open(required_file(
            &root,
            "text_encoder_2/model.fp16.safetensors",
        )?)
        .context("cannot load SDXL text encoder 2")?;
        let unet = SafeTensorFile::open(required_file(
            &root,
            "unet/diffusion_pytorch_model.fp16.safetensors",
        )?)
        .context("cannot load SDXL UNet")?;
        let vae = SafeTensorFile::open(required_file(
            &root,
            "vae/diffusion_pytorch_model.fp16.safetensors",
        )?)
        .context("cannot load SDXL VAE")?;

        require_all_f16(&text_encoder, "text encoder")?;
        require_all_f16(&text_encoder_2, "text encoder 2")?;
        require_all_f16(&unet, "UNet")?;
        require_all_f16(&vae, "VAE")?;
        require_tensor(&unet, "add_embedding.linear_1.weight", &[1_280, 2_816])?;
        require_tensor(&unet, "conv_in.weight", &[320, 4, 3, 3])?;
        require_tensor(&text_encoder_2, "text_projection.weight", &[1_280, 1_280])?;

        Ok(Self {
            root,
            text_encoder,
            text_encoder_2,
            unet,
            vae,
            tokenizer,
            tokenizer_2,
        })
    }

    pub fn print_summary(&self) {
        let total_bytes = self.text_encoder.mapped_len()
            + self.text_encoder_2.mapped_len()
            + self.unet.mapped_len()
            + self.vae.mapped_len();
        println!("Saient SDXL pack: {}", self.root.display());
        println!(
            "text encoder   : {:4} tensors  {:10} bytes",
            self.text_encoder.tensor_count(),
            self.text_encoder.mapped_len()
        );
        println!(
            "text encoder 2 : {:4} tensors  {:10} bytes",
            self.text_encoder_2.tensor_count(),
            self.text_encoder_2.mapped_len()
        );
        println!(
            "UNet           : {:4} tensors  {:10} bytes",
            self.unet.tensor_count(),
            self.unet.mapped_len()
        );
        println!(
            "VAE            : {:4} tensors  {:10} bytes",
            self.vae.tensor_count(),
            self.vae.mapped_len()
        );
        println!("weight total   : {total_bytes} bytes");
        println!("contract       : SDXL base / FP16 / 1024x1024 / dual CLIP-77");
        println!("validation     : passed");
    }

    /// Concatenated cross-attention context [1, 77, 2048] and the pooled
    /// micro-conditioning vector (1280 elements) from `text_encoder_2`.
    pub fn encode_prompt(&self, prompt: &str) -> Result<(crate::tensor::Tensor, Vec<f32>)> {
        let tokens = self.tokenizer.encode(prompt);
        let tokens_2 = self.tokenizer_2.encode(prompt);
        let context_1 = sdxl_clip::encode_penultimate(&self.text_encoder, &tokens, &CLIP_L)?;
        let (context_2, pooled) =
            sdxl_clip::encode_with_pool(&self.text_encoder_2, &tokens_2, &CLIP_BIGG)?;
        let context = concat_last_dim(&context_1, &context_2)?;
        Ok((context, pooled))
    }

    pub fn predict_noise(
        &self,
        sample: &crate::tensor::Tensor,
        timestep: f32,
        context: &crate::tensor::Tensor,
        pooled_text_embed: &[f32],
        time_ids: [f32; 6],
    ) -> Result<crate::tensor::Tensor> {
        crate::sdxl_unet::predict_noise(
            &self.unet,
            sample,
            timestep,
            context,
            pooled_text_embed,
            time_ids,
        )
    }

    pub fn decode_latents(&self, latents: &crate::tensor::Tensor) -> Result<crate::tensor::Tensor> {
        const SCALING_FACTOR: f32 = 0.13025;
        let mut scaled = latents.clone();
        for value in scaled.data_mut() {
            *value /= SCALING_FACTOR;
        }
        let image = crate::sd_vae::decode_unscaled(&self.vae, &scaled)?;
        if image.data().iter().any(|value| !value.is_finite()) {
            bail!(
                "SDXL VAE decode produced non-finite output (a known fp16 instability in the \
                 base VAE weights — the 'sdxl-vae-fp16-fix' community weights avoid this)"
            );
        }
        Ok(image)
    }

    pub fn generate(
        &self,
        request: &crate::sdxl_pipeline::GenerationRequest,
    ) -> Result<crate::tensor::Tensor> {
        crate::sdxl_pipeline::generate(self, request)
    }

    /// Opt this pack's UNet into staged Vulkan weight loading; see
    /// `Sd15Pack::enable_staged_vulkan_loading` for the mechanism.
    #[cfg(feature = "vulkan")]
    pub fn enable_staged_vulkan_loading(&self) -> Result<(usize, usize)> {
        let (budget_bytes, tensor_count_hint) =
            crate::sdxl_unet::staged_loading_budget(&self.unet)?;
        crate::vulkan::enable_staged_weight_loading(
            self.unet.mapping_key(),
            budget_bytes,
            tensor_count_hint,
        );
        Ok((budget_bytes, tensor_count_hint))
    }
}

/// Concatenate two [1, 77, hidden] tensors along the hidden dimension.
fn concat_last_dim(
    a: &crate::tensor::Tensor,
    b: &crate::tensor::Tensor,
) -> Result<crate::tensor::Tensor> {
    if a.shape()[..2] != b.shape()[..2] {
        bail!(
            "context concat shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
    }
    let (batch, sequence) = (a.shape()[0], a.shape()[1]);
    let (width_a, width_b) = (a.shape()[2], b.shape()[2]);
    let mut output = Vec::with_capacity(batch * sequence * (width_a + width_b));
    for position in 0..batch * sequence {
        output.extend_from_slice(&a.data()[position * width_a..(position + 1) * width_a]);
        output.extend_from_slice(&b.data()[position * width_b..(position + 1) * width_b]);
    }
    crate::tensor::Tensor::new(vec![batch, sequence, width_a + width_b], output)
}

fn require_all_f16(file: &SafeTensorFile, component: &str) -> Result<()> {
    for name in file.tensor_names() {
        let info = file.info(name).expect("name came from the same tensor map");
        if info.dtype != DType::F16 {
            bail!(
                "{component} tensor {name} is {:?}; the mobile pack requires F16",
                info.dtype
            );
        }
    }
    Ok(())
}

fn require_tensor(file: &SafeTensorFile, name: &str, shape: &[usize]) -> Result<()> {
    let info = file
        .info(name)
        .with_context(|| format!("required tensor is missing: {name}"))?;
    if info.dtype != DType::F16 {
        bail!("tensor {name} is {:?}; expected F16", info.dtype);
    }
    if info.shape != shape {
        bail!(
            "tensor {name} has shape {:?}; expected {shape:?}",
            info.shape
        );
    }
    Ok(())
}

fn validate_model_index(value: &Value) -> Result<()> {
    expect_string(value, "/_class_name", "StableDiffusionXLPipeline")?;
    expect_string(value, "/text_encoder/1", "CLIPTextModel")?;
    expect_string(value, "/text_encoder_2/1", "CLIPTextModelWithProjection")?;
    expect_string(value, "/tokenizer/1", "CLIPTokenizer")?;
    expect_string(value, "/tokenizer_2/1", "CLIPTokenizer")?;
    expect_string(value, "/unet/1", "UNet2DConditionModel")?;
    expect_string(value, "/vae/1", "AutoencoderKL")?;
    Ok(())
}

fn validate_clip_config(value: &Value, spec: &sdxl_clip::ClipEncoderSpec) -> Result<()> {
    expect_string(value, "/model_type", "clip_text_model")?;
    expect_string(
        value,
        "/hidden_act",
        if spec.gelu_exact {
            "gelu"
        } else {
            "quick_gelu"
        },
    )?;
    expect_u64(value, "/hidden_size", spec.hidden as u64)?;
    expect_u64(value, "/max_position_embeddings", 77)?;
    expect_u64(value, "/num_attention_heads", spec.heads as u64)?;
    expect_u64(value, "/num_hidden_layers", spec.layers as u64)?;
    expect_u64(value, "/vocab_size", spec.vocab as u64)?;
    Ok(())
}

fn validate_unet_config(value: &Value) -> Result<()> {
    expect_string(value, "/_class_name", "UNet2DConditionModel")?;
    expect_string(value, "/act_fn", "silu")?;
    expect_string(value, "/addition_embed_type", "text_time")?;
    expect_u64(value, "/addition_time_embed_dim", 256)?;
    expect_u64_array(value, "/attention_head_dim", &[5, 10, 20])?;
    expect_u64_array(value, "/block_out_channels", &[320, 640, 1_280])?;
    expect_u64(value, "/cross_attention_dim", 2_048)?;
    expect_u64(value, "/in_channels", 4)?;
    expect_u64(value, "/layers_per_block", 2)?;
    expect_f64(value, "/norm_eps", 1e-5)?;
    expect_u64(value, "/norm_num_groups", 32)?;
    expect_u64(value, "/out_channels", 4)?;
    expect_u64(value, "/projection_class_embeddings_input_dim", 2_816)?;
    expect_u64_array(value, "/transformer_layers_per_block", &[1, 2, 10])?;
    expect_bool(value, "/use_linear_projection", true)?;
    Ok(())
}

fn validate_vae_config(value: &Value) -> Result<()> {
    expect_string(value, "/_class_name", "AutoencoderKL")?;
    expect_string(value, "/act_fn", "silu")?;
    expect_u64_array(value, "/block_out_channels", &[128, 256, 512, 512])?;
    expect_u64(value, "/latent_channels", 4)?;
    expect_u64(value, "/layers_per_block", 2)?;
    expect_u64(value, "/norm_num_groups", 32)?;
    expect_u64(value, "/out_channels", 3)?;
    expect_f64(value, "/scaling_factor", 0.13025)?;
    Ok(())
}

fn validate_scheduler_config(value: &Value) -> Result<()> {
    expect_string(value, "/_class_name", "EulerDiscreteScheduler")?;
    expect_string(value, "/beta_schedule", "scaled_linear")?;
    expect_f64(value, "/beta_start", 0.00085)?;
    expect_f64(value, "/beta_end", 0.012)?;
    expect_string(value, "/prediction_type", "epsilon")?;
    expect_u64(value, "/num_train_timesteps", 1_000)?;
    expect_bool(value, "/use_karras_sigmas", false)?;
    expect_u64(value, "/steps_offset", 1)?;
    expect_string(value, "/timestep_spacing", "leading")?;
    Ok(())
}
