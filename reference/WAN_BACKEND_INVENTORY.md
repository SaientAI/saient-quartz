# Wan 2.1 tensor-operation inventory

This inventory is taken from the verified scalar graph at tag
`wan-2.1-scalar-reference`. It separates model semantics from operations that
an execution backend must provide. Tensor layouts below are logical Quartz
layouts; the model layer remains responsible for deciding shapes, ordering,
cache indices, and chunk branches.

## Shared storage and layout requirements

- Activations are FP32 in the scalar baseline.
- UMT5 and DiT weights are read from GGUF and include quantized Q4_K/Q6_K
  matrices. Embedding-row lookup must operate directly on quantized storage.
- VAE weights are BF16 safetensors, decoded to FP32 by the scalar baseline.
- Sequence matrices use row-major `[rows, channels]` with channels contiguous.
- Attention uses `[batch, heads, queries, head_dim]` (BHQD).
- VAE image tensors use contiguous `[N,C,T,H,W]` (NCTHW); frame-local Conv2D
  temporarily views them as `[N*T,C,H,W]` (NCHW).
- Wan DiT latents use `[C,T,H,W]` at the model boundary and row-major
  `[tokens, channels]` after patch embedding.

## UMT5-XXL

Source graph: `src/umt5.rs`.

| Operation | Shape or rule | Model-specific semantics |
| --- | --- | --- |
| Quantized embedding gather | token IDs -> `[valid_tokens,4096]` | Read individual rows; never materialize the 4.2 GB FP32 table. |
| RMSNorm | rows of 4096, epsilon `1e-6` | Last-axis normalization with learned scale and FP32 reduction. |
| Quantized GEMM | `[N,K] x [M,K] -> [N,M]` | Q/K/V/O and gated FFN projections; no implicit activation. |
| QK transpose/BMM | per head `[N,64] x [N,64]^T` | No `1/sqrt(head_dim)` scale for T5 attention. |
| Relative-bias gather/add | `[N,N]` bucket IDs into `[32,64]` | A distinct bias table is used in every block. |
| Softmax | key axis | Stable max-subtracted FP32 softmax. |
| Attention-value BMM | `[N,N] x [N,64]` | Bidirectional over valid tokens only. |
| Residual add | `[N,4096]` | After attention output and FFN output. |
| GELU (tanh approximation) | `[N,10240]` | Gate branch only. |
| Elementwise multiply | two `[N,10240]` tensors | `gelu(gate) * up`. |
| Pad/zero fill | valid rows -> `[512,4096]` | Padding remains exactly zero; this is graph policy. |

Relative-position bucketing is integer model logic and can remain on the CPU;
the resulting bucket indices or gathered bias values are backend inputs.

## Wan DiT, 30 blocks

Source graph: `src/wan_dit.rs` and RoPE construction in `src/wan_rope.rs`.

| Operation | Shape or rule | Model-specific semantics |
| --- | --- | --- |
| Patch Conv3D | `[16,T,H,W] -> [T,H/2,W/2,1536]` | Kernel and stride `[1,2,2]`, no overlap; equivalent to patch gather plus GEMM and bias. |
| Quantized GEMM + optional bias | row-major `[N,K] -> [N,M]` | Time/text embeddings, Q/K/V/O, MLP, and output head. |
| SiLU | timestep projections | Exact scalar formula `x/(1+exp(-x))`. |
| GELU (tanh approximation) | text embedding and MLP | Same approximation as UMT5. |
| LayerNorm | last axis 1536, epsilon `1e-6` | Both affine and non-affine variants are required. |
| RMSNorm | Q and K rows of 1536, epsilon `1e-6` | Normalize the full model dimension before splitting heads. |
| 3D RoPE | 12 heads x 128 | Axis split `44+42+42`; rotate Q and K only for self-attention. |
| Attention score BMM | `[12,N,128] x K^T` | Scale by `1/sqrt(128)`; both self- and cross-attention. |
| Softmax | key axis | Stable FP32 reference reduction. |
| Attention-value BMM | probabilities x V | Head merge preserves the 1536-channel order. |
| Broadcast affine | `[N,1536]` plus row vectors | AdaLN shift/scale and head modulation. |
| Gated residual update | `[N,1536]` | Per-channel gate multiply followed by add. |
| Patch scatter | tokens -> `[16,T,H,W]` | Patch-major/channel-minor head output ordering. |

Timestep embedding, RoPE-frequency construction, patch grid sizes, block order,
and modulation-vector slicing are model-layer responsibilities.

## Wan stateful causal VAE decoder

Source graph: `src/wan_vae.rs`; the complete graph and cache order are in
`WAN_VAE.md`.

| Operation | Shape or rule | Model-specific semantics |
| --- | --- | --- |
| Causal Conv3D | NCTHW with OICTHW weights | Explicit padding before `[2*Pt,Ph,Pw]`, after `[0,Ph,Pw]`; arbitrary stride/dilation/groups. |
| Frame-local Conv2D | `[N*T,C,H,W]`, OIHW weights | Used by spatial resampling and VAE attention projections. |
| Channel RMSNorm | NCTHW, epsilon `1e-12` | Reduce across C independently for every `(N,T,H,W)` location. |
| SiLU | NCTHW | Used after every VAE RMSNorm. |
| Residual add | matching NCTHW tensors | Includes optional causal `1x1x1` shortcut in graph logic. |
| Spatial attention | `[N*T,1,H*W,C]` | Standard scale `1/sqrt(C)`; frames must never attend across time. |
| Nearest spatial upsample | H and W x2 | Executed before the resample Conv2D. |
| Temporal concatenate/slice | NCTHW time axis | Used for feature-cache context and chunk assembly. |
| Channel-to-time shuffle | `[N,2C,T,H,W] -> [N,C,2T,H,W]` | Channel halves become adjacent output frames. |
| Zero prepend | time axis | Only the `chunk_idx == 1` temporal-upsample branch. |
| Output affine/clamp | NCTHW | `(x+1)*0.5`, then clamp to `[0,1]`. |

Feature-cache ownership, its 32 used indices, chunk sequencing, and the three
temporal-upsample branches stay in the Wan model layer. A backend executes the
convolution, transforms, and arithmetic selected by that control flow; it does
not choose the control flow.

## Backend operation surface

The complete execution surface implied by the three graphs is:

1. Storage: allocate, zero, copy, view, reshape, permute, concatenate, slice,
   host/device transfer, and persistent weight handles.
2. Weight access: typed dense weights, quantized row gather, deterministic
   dequantization, and quantized/dense GEMM.
3. Elementwise: add, multiply, scale, affine, SiLU, tanh-GELU, clamp, and type
   conversion.
4. Reductions/norms: RMSNorm (last axis and VAE channel axis), LayerNorm
   (affine and non-affine), stable softmax, and finite-value counts.
5. Dense math: GEMM, batched GEMM, bias addition, and broadcast modulation.
6. Convolution: Conv2D and Conv3D with explicit stride, dilation, grouping, and
   asymmetric per-axis padding.
7. Attention helpers: head split/merge, RoPE, QK multiplication, softmax, and
   probability-value multiplication.
8. Video transforms: patch gather/scatter, nearest spatial upsample,
   channel-to-time shuffle, and temporal cache concatenation.

The initial backend abstraction covers elementwise arithmetic, SiLU,
tanh-GELU, VAE channel RMSNorm, and dense linear projection. Its first parity
milestone used host-owned `Tensor` inputs and outputs. The resident milestone
below extends that boundary while keeping GGUF, safetensors, and Vulkan types
outside the Wan graph API.

## Resident execution milestone

The backend boundary now includes opaque `DeviceTensor` and
`LinearWeightHandle` values. Wan code can prepare a linear layer once and
compose resident linear and activation operations without seeing Vulkan buffer
identifiers or allocation types.

The first resident graph is the complete Wan time-embedding path:

`sinusoidal input -> linear -> SiLU -> linear -> SiLU -> time projection`

Its current Vulkan storage policy is FP32 activations, FP16 matrix weights, and
FP32 fused bias. Prepared matrices and biases remain in persistent Vulkan-owned
buffers; intermediate outputs remain resident until their last handle is
dropped. Each prepared linear uses one host-visible staging buffer and one queue
submission to copy its matrix and bias into separate `DEVICE_LOCAL` storage
buffers. Activations remain host-visible/coherent until the memory planner adds
device-local activation arenas and explicit output readback. Runtime accounting
tracks both logical device-local bytes and Vulkan allocation-requirement bytes
without changing the Wan-facing handle API.

The next resident graph extends the time embedding through block 0's
pre-self-attention normalization:

`e0 + block modulation -> LayerNorm(x) -> x * (1 + scale[1]) + shift[0]`

The shared `e0` projection stays resident between the time projection and the
block. LayerNorm reduces independently over the final 1536-value token axis
with epsilon `1e-6`; its shader supports both the non-affine norm1/norm2 form
and prepared FP32 affine weight/bias vectors for norm3. Block modulation is a
prepared device-local FP32 vector of shape `[6 * 1536]`. Addition, LayerNorm,
and broadcast modulation remain separate dispatches for first-divergence
parity testing.

The resident block boundary now continues through the self-attention inputs:

`modulated x -> Q/K/V projections -> full-width Q/K RMSNorm -> per-head 3D RoPE`

Q, K, V, and output-projection matrices are prepared once as device-local FP16
weights with FP32 biases. Q/K normalization weights remain device-local FP32.
RMSNorm reduces the complete 1536-value row before the 12 heads are interpreted;
RoPE then applies the same position-local 64-pair rotation matrix independently
to each 128-value head. The position matrix is a model-layer tensor and is not
reconstructed inside the Vulkan backend.

The first complete resident block uses an unfused attention ladder:

`QK score BMM -> stable softmax -> probability/value BMM -> output projection`

Scores use head-major `[heads, queries, keys]` storage; values and merged
contexts use row-major `[rows, 1536]`. The self-attention projection is gated
by modulation chunk 2 before its residual add. Cross-attention applies affine
norm3, attends to the separately resident text context without RoPE, and adds
its projection directly. The FFN applies non-affine norm2, modulation chunks
3/4, `1536 -> 8960 -> 1536` projections with tanh-GELU, and modulation chunk 5
before the final residual. These operations remain separate dispatches so
block-boundary parity identifies the first divergent primitive.

The resident DiT graph now includes its complete envelope and a bounded-memory
30-block loop:

`patch gather -> patch projection -> text/time projections -> 30 staged blocks -> head norm/modulation -> head projection -> patch scatter`

Patch gather/scatter and final head modulation have dedicated FP32 Vulkan
kernels. Each block owns 17 prepared resources and is dropped before the next
block is staged. On the captured `[16,2,30,52]` latent (780 tokens, 512 text
rows), the Vulkan output matched the captured velocity at cosine
`0.999946270`, maximum absolute error `0.050617695`, and mean absolute error
`0.008702695`. Staging plus execution took `173.962 s` on the detected RTX 5060
Ti; peak resident Vulkan memory was `277,077,248` bytes. The flow scheduler's
resident Euler update is a separate scale-plus-add pair and matched scalar
exactly on `[3,5,7]`.

The VAE port has begun at the primitive boundary. `Conv3dWeightHandle` owns an
FP16 OICTHW matrix and FP32 bias while activations remain FP32. The Vulkan
Conv3D accepts explicit padding-before and padding-after arrays; the causal
test `[1,2,3,4,5]` with `[3,2,3,3,3]` weights and temporal padding `2/0`
matched scalar exactly. The first real VAE prelude convolution on captured
`[1,16,2,8,8]` input matched at cosine `1.0`, maximum error `4.77e-7`, and mean
error `4.7e-8`.

## Resident VAE temporal-cache milestone

The backend now owns four FP32 NCTHW layout operations without host
reconstruction: temporal slice, ordered temporal concatenation, causal
zero-time prepend, and Wan's channel-to-time shuffle. The shuffle mapping is
fixed as

`input[n, half*C+c, t, h, w] -> output[n, c, 2*t+half, h, w]`.

Exact-equality fixtures cover first/last/middle/full slices of
`[2,3,5,2,3]`, a three-input concatenation, a two-frame zero prefix, and a
multi-batch `[2,4,2,2,3] -> [2,2,4,2,3]` shuffle. That focused Vulkan test
performed three host uploads (the source, shuffle source, and one deliberately
mismatched validation tensor) and seven deliberate comparison downloads; no
operation downloaded and reconstructed an intermediate tensor.

`DeviceFeatureCache` has exactly 32 independently replaceable slots. It
validates NCTHW shape, Vulkan ownership, matching non-temporal axes, and the
two-frame prefix limit. It reports logical current/peak bytes, occupancy,
replacement/eviction counts, resident ownership, and memory class. Reset drops
every slot and returns the active index to zero, so a new decode cannot retain
stale feature tensors. Vulkan activation outputs currently remain resident in
host-visible/coherent allocations; cache reporting therefore says
`all_slots_resident=true` and `all_slots_device_local=false` until the future
activation arena changes their memory class.

The accepted cached Conv3D fixture uses input `[1,2,5,2,3]`, weight
`[3,2,3,3,3]`, output `[1,3,5,2,3]`, causal padding-before `[2,1,1]`,
padding-after `[0,1,1]`, unit stride/dilation, and chunks `1+2+2`. Incremental
Vulkan execution matched one-shot scalar with cosine `1.0`, maximum error `0`,
and mean error `0`. The focused execution took `4.983 ms`, retained a 96-byte
two-frame cache prefix, and held 1,392 logical Vulkan bytes at comparison. It
used one activation upload, one prepared-weight upload, and one final download.

Temporal upsampling preserves its three reference branches rather than
collapsing them into one reshape. A deterministic `[1,2,3,1,2]` sequence
verified `chunk_idx == 0`, `chunk_idx == 1`, and `chunk_idx >= 2` independently.
Their output temporal lengths were `1`, `2`, and `2`; every output and cache
write matched scalar exactly (cosine `1.0`, maximum/mean error `0`). The focused
run took `5.385 ms`, retained 32 cache bytes, held 224 logical Vulkan bytes,
and used one activation upload, one prepared-weight upload, and five deliberate
downloads for three outputs plus the two populated cache states.

The complete VAE residual path is now resident: channel-axis RMSNorm with
epsilon `1e-12`, SiLU, two cached causal Conv3D operations, an optional learned
`1x1x1` projection, and the residual add. Device-local staging now pads each
transfer to Vulkan's four-byte copy granularity while retaining the tensor's
logical element and byte counts; this is required for FP16 convolution weights
with odd element counts. A deterministic channel-changing block processed
three one-frame chunks from `[1,2,3,2,2]` to `[1,3,3,2,2]`, used both cache
slots and a learned shortcut, and matched scalar at cosine `1.0`, maximum error
`4.77e-7`, and mean error `1.49e-7`. The focused run took `16.213 ms`, held
1,422 logical Vulkan bytes and 160 peak cache bytes, and used one activation
upload, five prepared-weight uploads, and one final download.

The real `decoder.upsamples.12` residual block from the preserved Wan VAE was
also checked on non-square `[1,96,1,2,3]` input. Its `[1,96,1,2,3]` output
matched scalar at cosine `1.0`, maximum error `1.431e-6`, and mean error
`1.60e-7`. Scalar execution took `16.149 ms`; Vulkan execution took `3.246 ms`.
The graph held 1,006,080 logical Vulkan bytes and 4,608 peak cache bytes, with
one activation upload, four prepared-weight uploads, and one final download.

## Resident VAE spatial milestone

The generic resident Conv2D path uses FP32 NCHW activations, prepared FP16
OIHW weights, FP32 bias, arbitrary spatial extents, stride, and symmetric
padding. It is a dedicated 2D shader, not a temporal Conv3D wrapper. The
awkward `[2,3,4,5]` fixture with `[3,3,3,3]` weights, stride `[2,1]`, and
padding `[1,1]` produced `[2,3,2,5]` exactly: cosine `1.0`, maximum/mean error
`0`, and `0.943 ms` focused runtime. Its odd 81-value FP16 weight payload also
verifies the padded Vulkan staging path. The run held 894 logical Vulkan bytes,
174 device-local logical bytes, one activation upload, one prepared-weight
upload, and one final download.

VAE spatial attention has explicit resident layout kernels for NCTHW frame
packing, frame reconstruction, Q/K/V channel splitting into
`[N*T,H*W,C]`, and sequence reconstruction. The attention ladder is unfused:
batched QK scores, row softmax, probability/value multiplication, projection,
and residual add. The batch axis is the flattened frame identity, so no score
or value operation can cross a temporal-frame boundary. The first trace found
and corrected a descriptor-contract error at the frame-packing output; the
accepted trace compares every intermediate boundary.

The deterministic `[2,2,2,2,3]` attention fixture (four frames, six spatial
positions) matched scalar at every boundary with final cosine `1.0`, maximum
error `2.38e-7`, and mean error `6.1e-8`. Changing only frame 1 left frame 0
bit-identical and changed the targeted frames by an aggregate `159.598557`.
The primary graph took `4.946 ms`; the test held 456 logical Vulkan bytes and
72 device-local logical bytes after its second pass, used two activation
uploads and three prepared-weight uploads, and performed fourteen deliberate
downloads for the thirteen-boundary trace plus the isolation output.

The real `decoder.middle.1` attention block was checked on
`[1,384,1,2,3]`. Its output matched scalar at cosine `1.0`, maximum error
`2.38e-7`, and mean error `3.9e-8`. Scalar execution took `91.069 ms`; Vulkan
execution took `10.004 ms`. The graph held 1,205,760 logical Vulkan bytes,
1,187,328 device-local logical bytes, one activation upload, three prepared
weight uploads, and one final download.

The first assembled real decoder section now runs as one resident graph:

`conv2 prelude -> cached decoder.conv1 -> middle.0 -> middle.1 attention -> middle.2`

On `[1,16,1,2,3]` input it produced `[1,384,1,2,3]`, consumed the expected
five independent cache slots, and matched scalar at cosine `1.0`, maximum
error `2.384e-6`, and mean error `2.36e-7`. Scalar execution took `548.257 ms`;
Vulkan execution took `29.323 ms`. The assembled graph held 33,430,848 logical
Vulkan bytes, 33,384,000 device-local logical bytes, and 37,248 cache bytes. It
used one activation upload, thirteen prepared-weight uploads, and one final
download.
