# Wan 2.1 3D causal VAE decoder reference

This document records the graph implemented by the pinned
`stable-diffusion.cpp` reference in `src/model/vae/wan_vae.hpp`. It covers the
Wan 2.1 decoder (`dim=96`, `z_dim=16`), not the different Wan 2.2 graph.

## Tensor layout and captured data

Quartz uses contiguous `[N, C, T, H, W]` tensors with `W` fastest. The SQD1
captures store the same bytes with ggml's fastest-first dimensions:

| Case | SQD1 dimensions | Quartz dimensions |
| --- | --- | --- |
| Small latent | `[8, 8, 2, 16, 1]` | `[1, 16, 2, 8, 8]` |
| Small pixels | `[64, 64, 5, 3, 1]` | `[1, 3, 5, 64, 64]` |
| Full latent | `[52, 30, 2, 16, 1]` | `[1, 16, 2, 30, 52]` |
| Full pixels | `[416, 240, 5, 3, 1]` | `[1, 3, 5, 240, 416]` |

`vae_in` is already transformed from diffusion space to VAE space. `vae_out`
is the public decoder result after `(x + 1) / 2` and clamping to `[0, 1]`.

## Decoder graph

The top-level `conv2` is a non-cached `1x1x1`, 16 to 16 channel convolution
over the complete latent. The resulting latent is then decoded one temporal
frame at a time. The decoder's feature-cache index is reset to zero for each
input chunk, while the cached tensors survive between chunks.

The channel progression is:

```text
latent 16
  -> conv2 16
  -> decoder.conv1 384
  -> middle residual 384
  -> spatial attention 384 (one independent H*W sequence per frame)
  -> middle residual 384
  -> three residuals 384
  -> temporal x2 + spatial x2 resample: 384 -> 192
  -> residual 192 -> 384, then two residuals 384
  -> temporal x2 + spatial x2 resample: 384 -> 192
  -> three residuals 192
  -> spatial x2 resample: 192 -> 96
  -> three residuals 96
  -> channel RMSNorm, SiLU, decoder.head.2 96 -> 3
```

Each residual is:

```text
shortcut = input or causal 1x1x1 convolution when Cin != Cout
hidden = RMSNorm(Cin) -> SiLU -> cached causal 3x3x3(Cin,Cout)
       -> RMSNorm(Cout) -> SiLU -> cached causal 3x3x3(Cout,Cout)
output = hidden + shortcut
```

RMSNorm is evaluated independently at every `(N,T,H,W)` location across the
channel axis, uses learned `gamma`, and has epsilon `1e-12`.

## Causal convolution

For configured padding `[Pt, Ph, Pw]`, the explicit padding is:

```text
time before = 2 * Pt
time after  = 0
height      = Ph before and after
width       = Pw before and after
```

When a feature cache is supplied, it is concatenated before the current input
and its temporal length is subtracted from `time before`. Every cached 3x3x3
convolution retains the latest two input frames. A 1x1x1 shortcut does not use
or consume a cache slot.

The decoder consumes 32 cache indices per chunk in this order:

1. `decoder.conv1` (1)
2. the two middle residuals (4)
3. twelve up-path residuals (24)
4. the two temporal upsample `time_conv` layers (2)
5. `decoder.head.2` (1)

The reference allocates 33 slots; one remains unused.

## Temporal upsampling branches

Each temporal resample first applies a causal `3x1x1` convolution from `C` to
`2C`, then rearranges the doubled channel axis onto time. Spatial nearest-x2
and a `3x3` convolution follow. The exact chunk branches are:

- `chunk_idx == 0`: skip `time_conv`, so one input frame produces one frame.
- `chunk_idx == 1`: preserve the last two current input frames as the cache,
  prepending zeros if fewer than two exist; run `time_conv` without a prior
  cache.
- `chunk_idx >= 2`: if the current input has one frame, combine the previous
  cache's last frame with it for the next cache; run `time_conv` with the full
  previous cache as causal context.

For a two-frame latent, the first chunk produces one pixel frame and the
second produces four, yielding five frames total.

## Spatial attention

Attention occurs only in `decoder.middle.1` at latent spatial resolution. It
uses channel RMSNorm, a `1x1` convolution producing concatenated Q/K/V, one
attention head of width 384 with the standard `1/sqrt(384)` scale, and a final
`1x1` projection. Time is treated as the batch dimension: frames do not attend
to one another in this block. The projected result is added to the input.

## Weight prefixes

- Top-level pre-convolution: `conv2`
- Decoder input: `decoder.conv1`
- Middle: `decoder.middle.{0,1,2}`
- Up path: `decoder.upsamples.0` through `decoder.upsamples.14`
- Output: `decoder.head.{0,2}`

The captured model contains BF16 weights in PyTorch order: Conv3D weights are
`[Cout, Cin, Kt, Kh, Kw]`, Conv2D weights are `[Cout, Cin, Kh, Kw]`, and biases
are `[Cout]`.
