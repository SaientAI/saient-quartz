# Quartz Train — experimental implementation

## Stage 0: scalar reference operators
- lora.rs: trainable A/B matrices over a frozen quantized projection;
  explicit adapter-only input gradients.
- loss.rs: two-pass vocab-chunked cross entropy + hidden gradient.
- Independent finite-difference, stability and parity tests.

## Stage 1A: real-model output-head LoRA (quartz train-head)
- head.rs takes a real GGUF and JSONL dataset. Quartz's existing
  GGUF loader, tokenizer, dense transformer forward, RMSNorm and lazy
  quantized lm_head projection generate real frozen model features.
- Training updates only a rank-r output-head LoRA adapter. It uses Adam,
  clipped averaged gradients, and a frozen GGUF verified by SHA-256 before
  and after training.
- The training examples each have a single next-token target; the
  frozen final hidden states and base token scores are cached.
- A distinct held-out split is evaluated before/after training.
- Writes a real safetensors file (quartz-lora-head-v0) with two f32
  matrices, a base GGUF SHA-256 identifier, and rank/alpha metadata.
  The loader verifies that identifier and the output's numerical parity.
- The targeted Android ARM64 build uses CPU for correctness, not GPU.

Run:
  quartz train-head BASE.gguf DATA.jsonl OUT.safetensors \
    --steps 80 --rank 4 --lr 0.015 --max-tokens 16

Input JSONL line:
  {"prompt":"Review: excellent\nSentiment:","target":" positive","split":"train"}

Independent reload with novel examples:
  quartz eval-head BASE.gguf FRESH_HELDOUT.jsonl OUT.safetensors --max-tokens 16

Rules: 3..24 total examples, at least two train and one held out, 1..32
tokens in prompts (default 16), one-token target text. Up to 500 steps.

Scope limitation: this is real GGUF-backed on-device training, but
only an output-head adapter. It is NOT full transformer LoRA, does
NOT backpropagate through attention, and cannot change Q/V projections.
Caching one vocab vector per example is bounded by max 24 examples,
but is not the Stage 1–3 long-sequence streaming CE design.

## Next acceptance gates
1. Verify real-model train/heldout loss on the S26 CPU.
2. Enable a portable lora-apply inference path for head adapters and
   compare cached and loaded outputs with independent model inference.
3. Implement norms, attention, RoPE, SwiGLU and all WᵀdY backward
   operators, with numerical gradients and tiny GGUF end-to-end tests.
4. Move to last-K q/v LoRA, streaming long-sequence CE, and Android
   lifecycle/thermal management. Only then consider Vulkan acceleration.

No merged GGUF writer, full SFT, durable optimiser checkpoints or
background training service is claimed in this milestone.


## Stage 1B: real last-layer Q/V LoRA

Implemented in qv.rs, qv_cli.rs, qv_io.rs, qv_gradient.rs and ops.rs.
A frozen-prefix Qwen2.5-0.5B forward and backward through the full
last dense block updates both Q and V LoRA matrices. Validated on
the actual S26, with an independent 6-example evaluation and an
end-to-end numerical gradient check on real frozen GGUF weights.
See QV_STAGE1B.md and experiments/android-lora-smoke/results/
for source, limits, exact performance logs and model hashes.


## Stage 2: connected multi-block Q/V LoRA

Code: multi_ops.rs, multi_qv.rs, multi_backward.rs, multi_io.rs,
multi_cli.rs. Unlike the last-block-only trainer, this backpropagates
through every trainable layer and every relevant earlier causal token.
It uses true GGUF frozen W^T gradients and a single common loss.
Supports last 1..8 Q/V LoRA blocks, short single-target training.
See MULTI_QV_STAGE2.md for mathematical scope, checks and limitations.

For the Samsung S26 eight-block demonstration:
  quartz train-qv-suffix BASE.gguf TRAIN.jsonl OUT.safetensors     --layers 8 --steps 1 --rank 2 --lr 0.01
  quartz eval-qv-suffix BASE.gguf UNSEEN.jsonl OUT.safetensors     --layers 8

This does not fine-tune the original quantized model weights or
support full-sequence training, persisted optimizer-state resume,
production adapter-aware inference or GPU training yet.
