# Quartz

Quartz is the on-device inference engine behind [Saient](https://saient.co.uk) —
a from-scratch Rust runtime (no llama.cpp, no ggml, no Candle, no
PyTorch/libtorch) that runs LLM chat and image generation entirely on a
phone or desktop, no cloud round-trip.

**It's the engine that made this possible:** Saient mobile generates
full text-to-video clips — [Wan2.1 T2V](https://github.com/Wan-Video/Wan2.1),
1.3B parameters — running end-to-end, locally, on a stock Samsung Galaxy
S24's Vulkan GPU. Same phone also runs LLM chat and SDXL image
generation locally through this engine. No server, no API key, nothing
leaves the device.

📹 **[End-to-end proof video](https://youtu.be/ZXc5hBgoro0)** — a full
on-device generation recorded with Wi-Fi and mobile data switched off,
so there is no network path a cloud call could take.

(The Wan video model itself runs through a separate, MIT-licensed
companion binary — see [Relationship to the Wan video engine](#relationship-to-the-wan-video-engine)
below for how the two fit together. This repo, Quartz proper, is the
LLM chat + SDXL image half of the stack.)

This repo covers Quartz: a minimal, dependency-light Rust inference
runtime for GGUF-format transformer language models, with a bundled
Stable Diffusion (SD1.5/SDXL) image pipeline. CI-grade targets are a
4-core desktop CPU, a CUDA GPU, and an Android phone's Vulkan GPU, via
a single ~15k-line, mostly-`unsafe`-free codebase.

This project was previously developed under the name **tinyq4**. All crate,
binary, and environment-variable names now use `quartz` / `QUARTZ_*`; the
git history and any external references to `tinyq4` predate the rename.

## What's in the box

Quartz ships as a single binary (`quartz`) with two capabilities:

1. **LLM chat inference** — GGUF model loading, tokenization (native GGUF
   vocab or a `minijinja`-templated Jinja2 chat template, matching what
   the source model ships), forward pass, and an OpenAI-compatible HTTP
   server for streaming completions.
2. **Image generation (SD1.5 / SDXL)** — a from-scratch CLIP text encoder,
   UNet, VAE, and scheduler implementation, driven by the same binary via
   `--generate-sd15` / `--generate-sdxl`.

Video generation (Wan2.1 T2V) is a **separate** binary and a separate
codebase — see [Relationship to the Wan video engine](#relationship-to-the-wan-video-engine)
below. Don't conflate the two; they share the Quartz brand but not a
source tree.

### Backends

- **CPU** — portable Rust, with a hand-rolled AVX2/FMA dot-product path on
  x86_64 (auto-dispatched at runtime; falls back to scalar otherwise) and a
  hand-written NEON dot-product path for Q4_K on aarch64 (Android).
- **CUDA** — optional (`--features cuda`), custom GEMV kernels per
  quantization format (`src/cuda_kernels.cu`, `src/cuda.rs`), staged/paged
  weight loading, and a VRAM-aware KV cache that sizes itself to free VRAM
  rather than a hard token cap.
- **Vulkan** — optional (`--features vulkan`), used for both LLM and SDXL
  compute; this is the backend the Android build ships with, since phones
  have no CUDA.

### Quantization formats

Reads GGUF files directly (`src/gguf.rs`) and dequantizes on the fly:
Q4_K, Q6_K, Q8_0, Q5_0/Q5_1, Q4_0/Q4_1, and the IQ2/IQ3 k-quant formats
(grid/sign tables in `src/iq2_tables.rs` / `src/iq3_tables.rs`, generated
from upstream `ggml-quants.c` / `ggml-common.h` constant tables — the
tables are data, not linked code).

### HTTP API

`quartz --server` exposes an OpenAI-compatible surface:

```
GET  /health
GET  /v1/health
GET  /v1/models
POST /v1/chat/completions   (SSE streaming)
```

This is what both the desktop and mobile Saient clients speak — same
wire format either side.

## Building

```bash
cargo build --release                       # CPU only
cargo build --release --features cuda       # + CUDA GEMV kernels
cargo build --release --features vulkan     # + Vulkan (SD image gen, or Android)
```

Android (arm64-v8a), used by the Saient mobile app:

```bash
scripts/build-android-arm64.sh
# → target/aarch64-linux-android/release/quartz
```

Requires `ANDROID_NDK_HOME` (or a discoverable NDK under
`$ANDROID_SDK_ROOT/ndk`). `QUARTZ_ANDROID_API` (default 24) and
`QUARTZ_ANDROID_FEATURES` (default `vulkan`) are overridable.

## Environment variables

| Variable | Effect |
|---|---|
| `QUARTZ_BIND` | Server bind address. Defaults to `127.0.0.1` (loopback only); set to `0.0.0.0` to opt into LAN/network access. |
| `QUARTZ_CTX` | Upper-bounds the KV cache context length the host app requests. |
| `QUARTZ_CUDA_ARCH` | SASS target arch(es) for the CUDA build (`build.rs`); CI/shipping builds set `all-major`. |
| `QUARTZ_DENSE_FFN_CPU` | Forces dense-FFN compute onto CPU (CUDA path), for debugging/comparison. |
| `QUARTZ_DEBUG_PROMPT` | Dumps the fully-rendered prompt before tokenization. |
| `QUARTZ_NO_JINJA` | Skips the Jinja2 chat-template path even when the model has no native template. |
| `QUARTZ_NO_WARMUP` | Set by the Android host app before every engine start (see note below). |

**Note on `QUARTZ_NO_WARMUP`:** this flag is set by the mobile app
(`plugins/quartz/QuartzEngine.kt` in the Saient mobile repo) to try to
avoid a full-resident mmap warmup that can get the app OOM-killed on
memory-constrained phones. As of this writing the Quartz binary does not
actually read this variable anywhere — the mmap warmup always runs
unconditionally (see the `"quartz: mmap warmup complete"` log line in
`src/main.rs`). This was discovered while renaming `TINYQ4_NO_WARMUP` →
`QUARTZ_NO_WARMUP`; it is a pre-existing no-op on the engine side, not
something this rename introduced or fixed. Worth following up on if
low-memory devices are still seeing OOM kills at startup.

## Relationship to the Wan video engine

The Saient mobile app also ships `libquartz-wan.so`, a *second*,
unrelated binary used only for Wan2.1 text-to-video generation. It is
**not** built from this repository — it's a pinned fork of
[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)
(MIT-licensed, C++/ggml), patched with a small progress-reporting patch
and cross-compiled for Android/Vulkan. Its build lock lives in the mobile
repo at `engine/wan/BACKEND_LOCK.json` and `engine/wan/saient-progress.patch`.
Both binaries carry the "Quartz" name in the mobile app's UI/branding
(`QuartzService`, `QuartzEngine.kt` supervise both processes), but they
are two separate codebases with two separate license situations — this
repo (Quartz proper) is what's being open-sourced here; the Wan fork's
own license terms (MIT) and upstream attribution travel with it
separately. See the mobile repo's `docs/VIDEO_PIPELINE.md` for how the
two engines are orchestrated together on-device.

### A native Wan2.1 implementation is in progress here

Quartz now contains its own Wan2.1 implementation in Rust — the flow-matching
scheduler, the SentencePiece Unigram tokenizer, the UMT5-XXL encoder, 3D RoPE,
the 30-block DiT, and the stateful 3D causal VAE decoder — running on the same
Vulkan backend as the rest of this repo, with no external inference runtime.
Every stage was built against a numerical parity harness rather than by
eyeballing output frames, because a wrong RoPE, a wrong AdaLN modulation and a
wrong VAE scale all produce indistinguishable garbage. See
[`reference/README.md`](reference/README.md).

End-to-end parity passes, and the result is a stronger statement than "close
enough": measured against the C++ engine's own output, the Rust pipeline
agrees with it **more closely than that engine agrees with itself**.

The reference engine does not have one answer at this precision. Toggling only
`--diffusion-fa` — same binary, same weights, same seed, same prompt — moves
its decoded pixels by cosine `0.998562`, max abs `0.234375`. Quartz differs
from it by cosine `0.999008`, max abs `0.184277`: inside the engine's own
arithmetic spread on every metric. Both figures come from controlled runs that
reproduce the committed captures bit-exactly.

**It does not replace the C++ engine yet, and the app still ships that engine.**
Performance is the blocker, and it is not close: a 5-frame 416x240 clip takes
about **639 s** on a desktop RTX 5060 Ti, against roughly a second for the
C++/GPU engine. The Vulkan graph is deliberately unfused so that block-boundary
parity can identify the first divergent primitive. Fusing it is a separate and
substantial piece of work.

## Status / provenance note

This repo has uncommitted local changes beyond the rename performed for
open-sourcing (a chat-template fix, KV-cache sizing fix, NEON Q4_K dot
product, and the SD1.5/SDXL modules `sd15.rs`/`sdxl*.rs`/`tensor.rs`/
`safetensors.rs`/`vulkan.rs` are untracked as of this writing). Review
and commit those separately before or as part of publishing — this
README describes the working tree as it stands, not necessarily the
last pushed commit.

## License

MIT — see [`LICENSE`](LICENSE). Matches the license of the bundled Wan
video engine (see below), so the whole on-device stack is MIT.
