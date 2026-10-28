# Quartz Stage 2 — genuine multi-block Q/V LoRA, experimental

This is **not** the output-head adapter and **not** eight independent
per-layer optimizers fitted to frozen, cached features. It computes
the reverse-mode gradient **through the interconnected suffix blocks**
of a real GGUF transformer. The full forward and backwards paths use
native Quartz Rust and CPU, with no cloud training and no PyTorch.

## What the code does

- Load quantized, mmap-backed Qwen2.5-0.5B Instruct GGUF.
- Freeze the GGUF and the initial N-K transformer blocks completely.
  Prefix states are evaluated and cached once for every prompt token.
- Run ALL prompt tokens through K subsequent dense transformer blocks.
- In EACH adapted block: train rank-r Q and V LoRA A/B matrices.
  Frozen K, attention output, FFN gate/up/down, RMSNorm and head remain
  in the forward pass.
- Reverse gradients across EACH block and ALL causal token edges:
  frozen LM head, final norm, FFN residual, SwiGLU, gate/up/down
  transposes, attention output transpose, softmax/GQA dQ/dK/dV,
  inverse RoPE/QK norm, trainable Q/V LoRA and frozen Q/K/V
  transposes, attention-input norm, residual.
- Earlier-token keys/values accumulate dK/dV from all later queries,
  not just the final query. This distinction matters for correctness.
- Batched GGUF transpose GEMV dequantizes each frozen row only once
  for the prompt token batch, rather than materializing full FP32
  model weights.
- Test initial zero-adapter output against normal Quartz logits.
- Compute CE on one target token, update all K Q/V LoRA matrices.
- Save an atomic multi-tensor safetensors adapter including exact base
  GGUF SHA-256, adapted layer range, rank, alpha and update count.
- Reload in a fresh process for evaluation with separate heldout data.

## Verification

The real-model Qwen2.5-0.5B desktop two-block run passed:
- Normal Quartz forward parity (reported relative error 0).
- Eight analytic-vs-numerical gradient checks on Q.A, Q.B, V.A,
  V.B in **both** layers 22 and 23, including cross-layer gradients.
- One optimizer step changed both Q/V B matrices in both blocks.
- Train CE 1.0488070 -> 0.9286849.
- Two-token-label heldout CE 0.6603671 -> 0.5884012.
- Model SHA-256 unchanged; multi-layer safetensors roundtrip passed.

The eight-block Samsung S26 run PASSED on actual layers 16..23,
rank 2, using two training examples and one original heldout example.
- One step: train CE 1.0487998 -> 0.4235311; heldout
  0.6603754 -> 0.2987503; 55.796 s, thermal 0, battery 54% -> 54%.
- Four steps: train CE 1.0487998 -> 0.0111069; heldout
  0.6603754 -> 0.0062530; 98.539 s, thermal 0, battery 53% -> 53%.
  Training loss rose sharply on step 3 before recovering on step 4.
- Separate fresh phone processes evaluated both saved adapters on two
  unseen prompts. Mean CE was 1.1034586 unadapted, 0.3857160 with
  the one-step adapter and 0.0191992 with the four-step adapter.
- Numerical gradients through the REAL eight-block Qwen stack passed
  at both the earliest and latest adapted layers (16 and 23).
- The original GGUF hash stayed unchanged. The safetensors reload
  tests passed, and both trained adapters were SHA-256 verified on
  phone AND back on the PC. Original logs and per-step findings are
  in experiments/android-lora-smoke/results/2026-10-28-multi8*.


## CLI

  quartz train-qv-suffix MODEL.gguf TRAIN.jsonl OUT.safetensors \
    --layers 2 --steps 1 --rank 2 --lr 0.01 --max-tokens 16 \
    --gradient-check

  quartz train-qv-suffix MODEL.gguf TRAIN.jsonl OUT.safetensors \
    --layers 8 --steps 1 --rank 2 --lr 0.01 --max-tokens 16

  quartz eval-qv-suffix MODEL.gguf UNSEEN.jsonl ADAPTER.safetensors \
    --layers 8 --max-tokens 16

## Limits

- Frozen quantized **base weights** are never modified. This is LoRA,
  not full-parameter weight training.
- Only final K=1..8 blocks are trainable, and Q/V only.
- One supervised next-token target, short sequences (up to 16 tokens),
  six examples maximum in this prototype. Not yet full SFT,
  backpropagation into frozen prefix, gradient accumulation across
  longer contexts, training all model parameters or checkpoint/resume
  of optimizer moments.
- CPU reference correctness path, not GPU/Vulkan acceleration.
- Small synthetic sentiment classification exercises limited
  behaviour; broad language learning has not been established.
- Production inference binding for this multi-layer adapter format
  remains a separate integration task.
