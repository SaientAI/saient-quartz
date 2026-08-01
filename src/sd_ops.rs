//! Shared mapped-weight operators for Quartz's fixed SD1.5 graphs.

use anyhow::{Result, bail};

use crate::{
    safetensors::{DType, SafeTensorFile},
    tensor::Tensor,
};

pub fn load_tensor(weights: &SafeTensorFile, name: &str) -> Result<Tensor> {
    let view = weights.view(name)?;
    if view.dtype != DType::F16 && view.dtype != DType::F32 && view.dtype != DType::BF16 {
        bail!("{name} is not a floating-point tensor");
    }
    let data = (0..view.len()).map(|index| view.value(index)).collect();
    Tensor::new(view.shape.to_vec(), data)
}

pub fn group_norm(
    input: &Tensor,
    weights: &SafeTensorFile,
    prefix: &str,
    groups: usize,
    epsilon: f32,
) -> Result<Tensor> {
    let weight = load_tensor(weights, &format!("{prefix}.weight"))?;
    let bias = load_tensor(weights, &format!("{prefix}.bias"))?;
    input.group_norm(groups, &weight, &bias, epsilon)
}

pub fn conv2d(
    input: &Tensor,
    weights: &SafeTensorFile,
    prefix: &str,
    padding: [usize; 2],
) -> Result<Tensor> {
    conv2d_full(input, weights, prefix, [1, 1], padding)
}

pub fn conv2d_full(
    input: &Tensor,
    weights: &SafeTensorFile,
    prefix: &str,
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    if crate::vulkan::sd_acceleration_requested() {
        let weight = weights.mapped(&format!("{prefix}.weight"))?;
        let bias = weights.view(&format!("{prefix}.bias"))?;
        return crate::vulkan::conv2d(input, weight, bias, stride, padding);
    }
    let weight = load_tensor(weights, &format!("{prefix}.weight"))?;
    let bias = load_tensor(weights, &format!("{prefix}.bias"))?;
    input.conv2d(&weight, Some(&bias), stride, padding, [1, 1], 1)
}

pub fn linear(input: &Tensor, weights: &SafeTensorFile, prefix: &str) -> Result<Tensor> {
    #[cfg(feature = "vulkan")]
    if crate::vulkan::sd_acceleration_requested() {
        let weight_name = format!("{prefix}.weight");
        let bias_name = format!("{prefix}.bias");
        let weight = weights.mapped(&weight_name)?;
        let bias = weights
            .info(&bias_name)
            .map(|_| weights.view(&bias_name))
            .transpose()?;
        return crate::vulkan::linear(input, weight, bias);
    }
    let weight = load_tensor(weights, &format!("{prefix}.weight"))?;
    let bias_name = format!("{prefix}.bias");
    let bias = weights
        .info(&bias_name)
        .map(|_| load_tensor(weights, &bias_name))
        .transpose()?;
    input.linear(&weight, bias.as_ref())
}

pub fn layer_norm(
    input: &Tensor,
    weights: &SafeTensorFile,
    prefix: &str,
    epsilon: f32,
) -> Result<Tensor> {
    let weight = load_tensor(weights, &format!("{prefix}.weight"))?;
    let bias = load_tensor(weights, &format!("{prefix}.bias"))?;
    input.layer_norm(&weight, &bias, epsilon)
}
