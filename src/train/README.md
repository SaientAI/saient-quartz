# Quartz Train — Stage 0 (experimental)

This is an initial implementation, NOT a finished GGUF trainer.
It follows the Quartz LoRA design proposal.

Implemented:
- CPU reference LoRA A/B projection, compatible with a frozen base.
- Adapter dA, dB, and dX including the adapter A^T B^T dY term.
  The frozen W^T dY must still be added by the model-level caller.
- Two-pass vocab-chunked cross-entropy with callback projections.
- Finite-difference and numerical-parity unit tests.

Not yet implemented: GGUF forward/backprop integration, attention/RoPE/
normalization backward, AdamW checkpoints, Android thermal scheduler,
safetensors adapter writer, inference lora-apply, or model fine-tuning.

The Samsung APK smoke test separately demonstrated rank-2 LoRA on a
synthetic 6x8 layer. That result is not a speed estimate for 0.6–2B
models.

Run isolated dependency-free Rust tests:
  rustc --edition 2024 --test src/train/mod.rs -o /tmp/quartz-lora-tests
  /tmp/quartz-lora-tests

Source lives in a separate clone/branch to preserve the modified
/home/tiny/projects/tinyq4 working tree.
