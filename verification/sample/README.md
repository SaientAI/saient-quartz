# Public sample adapter: Qwen2.5-0.5B, last-eight-block Q/V LoRA

File: adapter-qwen25-qv16-23-r2-step4-s26.safetensors

- Format: Quartz experimental quartz-multi-qv-v0 safetensors.
- Target: 8 connected transformer layers, 16–23 of Qwen2.5-0.5B Instruct.
- Trainable tensors: 4 per layer, Q.A/Q.B/V.A/V.B, rank 2, alpha 4.
- 45,056 learned FP32 parameters, 182,936 bytes total.
- Base model GGUF SHA-256:
  6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653
- This sample adapter SHA-256:
  adfb838b23efefadf9423df0d3b6574920b378173c6ef026a31b5ec4a56c710d
- CPU training completed natively on Samsung SM-S948B Android 17,
  Qualcomm SM8850, four Adam steps, 98.5 seconds (measured).

**What is inside?** Only low-rank A/B weights fitted on the tiny,
authored sentiment JSONL fixture in this repository. There are no
user conversations, personal messages, user files, original GGUF
base model weights, API keys or model-source downloads in this adapter.

**This is NOT a standalone language model** and does not use the
generic PEFT/Transformers LoRA naming/metadata scheme. It is loaded
by Quartz's experimental eval-qv-suffix and train-qv-suffix paths.

To test it:

    cargo build --release
    export QUARTZ_GGUF=/path/to/exact/qwen05.gguf
    python3 verification/verify.py --mode audit
    python3 verification/verify.py --mode verify

Important: the byte-identical frozen GGUF used in the experiment
must be supplied separately. The adapter is intentionally bound to
the exact GGUF SHA-256 and Quartz will reject another conversion,
even if it is called Qwen2.5-0.5B Instruct. If you cannot source the
same bytes, use the included training JSONL to train your *own*
adapter against your own licensed GGUF, then create a separately
pinned verification manifest. Do not bypass identity verification.

The tiny two-label task proves only narrow pipeline behaviour.
Six held-out sentiment examples improved on average, but one became
worse, and the negation challenge exposed an error. Read the
published verification results and raw logs, not just the headline.

Software licence: MIT (root LICENSE). Base Qwen2.5 model licence:
Apache-2.0; base checkpoint not included. Trained adapter derives
from the licensed base model and the included synthetic training
examples; retain applicable upstream notices when redistributing.
