// quartz CUDA kernels — dequant + GEMV for each quantisation format.
// Compiled with nvcc -arch=sm_120 (Blackwell / RTX 5060 Ti).
//
// Each kernel: one warp (32 threads) per output row.
// Threads stride over blocks so the full in_dim is covered regardless of size.

#include <stdint.h>
#include <cuda_fp16.h>
#include <string.h>  // memcpy
#include <stdio.h>

// ── Helpers ───────────────────────────────────────────────────────────────────

__device__ __forceinline__ float f16_to_f32_dev(uint16_t bits) {
    __half h;
    memcpy(&h, &bits, 2);
    return __half2float(h);
}

__device__ __forceinline__ void warp_reduce(float &v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, off);
}

// ── IQ4_NL ────────────────────────────────────────────────────────────────────
// Block: [d: f16 (2 B)][qs: 16 B] = 18 bytes, 32 values.
// 4-bit indices into non-linear lookup table; two per byte (low nibble first).
// value = d * table[idx]   (d already encodes the per-block scale)

__device__ __constant__ float iq4nl_table[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
       1.f,   13.f,  25.f,  38.f,  53.f,  69.f,  89.f, 113.f
};

__global__ void gemv_iq4nl(const uint8_t* __restrict__ data,
                            const float*   __restrict__ x,
                            float*         __restrict__ y,
                            int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const int BLOCK = 32, BSIZ = 18;
    const int bpr = in_dim / BLOCK;
    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;

    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        const float d = f16_to_f32_dev(d_bits);
        const uint8_t* qs = blk + 2;
        const float*   xi = x + b * BLOCK;
        #pragma unroll 16
        for (int k = 0; k < 16; k++) {
            acc += xi[k]      * d * iq4nl_table[qs[k] & 0x0F];
            acc += xi[k + 16] * d * iq4nl_table[qs[k] >>    4];
        }
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── BF16 ──────────────────────────────────────────────────────────────────────
// Row-major, 2 bytes per element.  value = (bits << 16) reinterpreted as f32.

__global__ void gemv_bf16(const uint8_t* __restrict__ data,
                           const float*   __restrict__ x,
                           float*         __restrict__ y,
                           int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const uint16_t* rd = (const uint16_t*)(data + (size_t)row * in_dim * 2);

    float acc = 0.f;
    for (int k = lane; k < in_dim; k += 32) {
        uint32_t bits = (uint32_t)rd[k] << 16;
        float w; memcpy(&w, &bits, 4);
        acc += x[k] * w;
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── F32/F16 ───────────────────────────────────────────────────────────────────

__global__ void gemv_f32(const uint8_t* __restrict__ data,
                         const float*   __restrict__ x,
                         float*         __restrict__ y,
                         int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const float* rd = (const float*)(data + (size_t)row * in_dim * 4);

    float acc = 0.f;
    for (int k = lane; k < in_dim; k += 32)
        acc += x[k] * rd[k];
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

__global__ void gemv_f16(const uint8_t* __restrict__ data,
                         const float*   __restrict__ x,
                         float*         __restrict__ y,
                         int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const uint16_t* rd = (const uint16_t*)(data + (size_t)row * in_dim * 2);

    float acc = 0.f;
    for (int k = lane; k < in_dim; k += 32)
        acc += x[k] * f16_to_f32_dev(rd[k]);
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q5_0 ──────────────────────────────────────────────────────────────────────
// Block: [d: f16][qh: 4 B][qs: 16 B] = 22 bytes, 32 values.
// value = (q5 - 16) * d  (signed 5-bit centered at 0; no m offset)

__global__ void gemv_q5_0_v4(const uint8_t* __restrict__ data,
                               const float*   __restrict__ x,
                               float*         __restrict__ y,
                               int in_dim, int out_dim)
{
    const int row  = blockIdx.x * 2 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= out_dim) return;
    const int BSIZ = 22, VALS = 32;
    const int bpr  = in_dim / VALS;
    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;
    float acc = 0.f;
    for (int b = 0; b < bpr; b++) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits;
        memcpy(&d_bits, blk, 2);
        const float dv = f16_to_f32_dev(d_bits);
        uint32_t qh; memcpy(&qh, blk + 2, 4);
        const uint8_t* qs = blk + 6;
        uint8_t nibble = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        uint8_t hi = (qh >> lane) & 1u;
        float q = (float)(int8_t)((nibble | (hi << 4)) - 16);
        float xi = x[b * VALS + lane];
        acc += xi * (q * dv);
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q5_1 ──────────────────────────────────────────────────────────────────────
// Block: [d: f16][m: f16][qh: 4 B][qs: 16 B] = 24 bytes, 32 values.
// value = q5 * d + m  (unsigned 5-bit; low 4 bits from qs, bit 4 from qh)

__global__ void gemv_q5_1(const uint8_t* __restrict__ data,
                           const float*   __restrict__ x,
                           float*         __restrict__ y,
                           int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const int BLOCK = 32, BSIZ = 24;
    const int bpr = in_dim / BLOCK;
    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;

    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk,     2);
        memcpy(&m_bits, blk + 2, 2);
        const float d = f16_to_f32_dev(d_bits);
        const float m = f16_to_f32_dev(m_bits);
        uint32_t qh; memcpy(&qh, blk + 4, 4);
        const uint8_t* qs = blk + 8;
        const float*   xi = x + b * BLOCK;
        #pragma unroll 16
        for (int k = 0; k < 16; k++) {
            uint32_t xh0 = ((qh >>  k     ) << 4) & 0x10;
            uint32_t xh1 = ((qh >> (k + 12))    ) & 0x10;
            float q0 = (float)((qs[k] & 0x0F) | xh0);
            float q1 = (float)((qs[k] >>   4) | xh1);
            acc += xi[k]      * (q0 * d + m);
            acc += xi[k + 16] * (q1 * d + m);
        }
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q5_1 GEMV v3: coalesced x access — each outer step all 32 threads ─────────
// Kept for reference; v4 (below) supersedes it for serial calls.
__global__ void gemv_q5_1_v3(const uint8_t* __restrict__ data,
                               const float*   __restrict__ x,
                               float*         __restrict__ y,
                               int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;   // 0..31

    const int BSIZ = 24, VALS = 32;
    const int bpr  = in_dim / VALS;

    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;

    float acc = 0.f;
    for (int b = 0; b < bpr; b++) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk,     2);
        memcpy(&m_bits, blk + 2, 2);
        const float dv = f16_to_f32_dev(d_bits);
        const float mv = f16_to_f32_dev(m_bits);
        uint32_t qh; memcpy(&qh, blk + 4, 4);
        const uint8_t* qs = blk + 8;
        uint8_t nibble = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        uint8_t hi     = (qh >> lane) & 1u;
        float   q      = (float)(nibble | (hi << 4));
        float xi = x[b * VALS + lane];
        acc += xi * (q * dv + mv);
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q5_1 GEMV v4: 2 rows per block, 64 threads — doubles warp occupancy ───────
// sm_120: max 24 blocks/SM, max 48 warps/SM.
// v3 (32 threads/block): 24 blocks → 24 warps/SM (50% warp occupancy).
// v4 (64 threads/block): 24 blocks → 48 warps/SM (100% warp occupancy).
// More warps/SM means the scheduler can hide DRAM latency better.
// grid: ((out_dim+1)/2), block: 64.
// Warp 0 (threads  0-31) → row blockIdx.x*2
// Warp 1 (threads 32-63) → row blockIdx.x*2 + 1
__global__ void gemv_q5_1_v4(const uint8_t* __restrict__ data,
                               const float*   __restrict__ x,
                               float*         __restrict__ y,
                               int in_dim, int out_dim)
{
    const int row  = blockIdx.x * 2 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= out_dim) return;

    const int BSIZ = 24, VALS = 32;
    const int bpr  = in_dim / VALS;
    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;

    float acc = 0.f;
    for (int b = 0; b < bpr; b++) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk,     2);
        memcpy(&m_bits, blk + 2, 2);
        const float dv = f16_to_f32_dev(d_bits);
        const float mv = f16_to_f32_dev(m_bits);
        uint32_t qh; memcpy(&qh, blk + 4, 4);
        const uint8_t* qs = blk + 8;
        uint8_t nibble = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        uint8_t hi     = (qh >> lane) & 1u;
        float   q      = (float)(nibble | (hi << 4));
        float xi = x[b * VALS + lane];
        acc += xi * (q * dv + mv);
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q5_1 GEMV v4 batched: 4 experts × 2 rows/block ────────────────────────────
// Runs up to 4 independent GEMV calls in parallel via a 2D grid.
// grid: ((out_dim+1)/2, n_act), block: 64.
// For gate/up: all xi0..xi3 point to the same d_norm_out.
// For down:    xi0..xi3 point to per-expert intermediate buffers.
__global__ void gemv_q5_1_v4_moe4(
    const uint8_t* w0, const uint8_t* w1,
    const uint8_t* w2, const uint8_t* w3,
    const float* xi0, const float* xi1,
    const float* xi2, const float* xi3,
    float* o0, float* o1, float* o2, float* o3,
    int n_act, int in_dim, int out_dim)
{
    const int eid  = blockIdx.y;
    const int row  = blockIdx.x * 2 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (eid >= n_act || row >= out_dim) return;

    const uint8_t* w;
    const float*   xi;
    float*         out;
    switch (eid) {
        case 0: w = w0; xi = xi0; out = o0; break;
        case 1: w = w1; xi = xi1; out = o1; break;
        case 2: w = w2; xi = xi2; out = o2; break;
        default: w = w3; xi = xi3; out = o3; break;
    }

    const int BSIZ = 24, VALS = 32;
    const int bpr  = in_dim / VALS;
    const uint8_t* rd = w + (size_t)row * bpr * BSIZ;

    float acc = 0.f;
    for (int b = 0; b < bpr; b++) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits, m_bits;
        memcpy(&d_bits, blk,     2);
        memcpy(&m_bits, blk + 2, 2);
        const float dv = f16_to_f32_dev(d_bits);
        const float mv = f16_to_f32_dev(m_bits);
        uint32_t qh; memcpy(&qh, blk + 4, 4);
        const uint8_t* qs = blk + 8;
        uint8_t nibble = (lane < 16) ? (qs[lane] & 0x0F) : (qs[lane - 16] >> 4);
        uint8_t hi     = (qh >> lane) & 1u;
        float   q      = (float)(nibble | (hi << 4));
        float xv = xi[b * VALS + lane];
        acc += xv * (q * dv + mv);
    }
    warp_reduce(acc);
    if (lane == 0) out[row] = acc;
}

// ── Q4_K ──────────────────────────────────────────────────────────────────────
// Block: [d: f16][dmin: f16][scales: 12 B][qs: 128 B] = 144 bytes, 256 values.

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* sc,
                                                  uint8_t* out_sc, uint8_t* out_min)
{
    if (j < 4) {
        *out_sc  = sc[j]     & 63;
        *out_min = sc[j + 4] & 63;
    } else {
        *out_sc  = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        *out_min = (sc[j + 4] >>   4) | ((sc[j]     >> 6) << 4);
    }
}

__global__ void gemv_q4_k(const uint8_t* __restrict__ data,
                           const float*   __restrict__ x,
                           float*         __restrict__ y,
                           int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const int BLOCK = 256, BSIZ = 144;
    const int bpr = in_dim / BLOCK;
    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;

    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk,     2);
        memcpy(&dmin_bits, blk + 2, 2);
        const float d    = f16_to_f32_dev(d_bits);
        const float dmin = f16_to_f32_dev(dmin_bits);
        const uint8_t* scales = blk + 4;
        const uint8_t* qs     = blk + 16;
        const float*   xi     = x + b * BLOCK;

        int q_off = 0;
        int x_off = 0;
        for (int is = 0; is < 8; is += 2) {
            uint8_t sc1, m1, sc2, m2;
            get_scale_min_k4(is,     scales, &sc1, &m1);
            get_scale_min_k4(is + 1, scales, &sc2, &m2);
            float d1 = d * sc1, md1 = dmin * m1;
            float d2 = d * sc2, md2 = dmin * m2;
            #pragma unroll 32
            for (int l = 0; l < 32; l++) {
                acc += xi[x_off + l]      * (d1 * (float)(qs[q_off + l] & 0x0F) - md1);
                acc += xi[x_off + l + 32] * (d2 * (float)(qs[q_off + l] >>    4) - md2);
            }
            q_off += 32;
            x_off += 64;
        }
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q4_K v4: 2 rows per block, 64 threads ─────────────────────────────────────
// Cooperative layout: all 32 lanes work on every 256-weight block together
// (the old "one block per lane" left 18/32 lanes idle when in_dim/256 < 32, e.g.
// 3584 → bpr=14). Each lane loads ONE uint32 of qs (a coalesced 128 B/warp read,
// 8 nibbles = 8 weights) plus two float4 of x, so loads are fully vectorized.
//
// Lane → weight mapping inside a block:
//   pair p = lane / 8       → sub-blocks (2p, 2p+1)
//   j0     = (lane*4) % 32  → 4-weight offset within each sub-block (16 B aligned)
//   qs[4*lane .. 4*lane+3]: low nibbles → sub 2p, high nibbles → sub 2p+1.
__global__ void gemv_q4_k_v4(const uint8_t* __restrict__ data,
                               const float*   __restrict__ x,
                               float*         __restrict__ y,
                               int in_dim, int out_dim)
{
    const int row  = blockIdx.x * 2 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= out_dim) return;
    const int BSIZ = 144;
    const int bpr  = in_dim >> 8;            // in_dim / 256
    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;

    const int p   = lane >> 3;               // 0..3 — which sub-block pair
    const int j0  = (lane << 2) & 31;        // 0,4,...,28 — offset within sub-block
    const int xa0 = (2 * p)     * 32 + j0;   // x index for low-nibble sub-block
    const int xb0 = (2 * p + 1) * 32 + j0;   // x index for high-nibble sub-block

    float acc = 0.f;
    for (int b = 0; b < bpr; b++) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk,     2);
        memcpy(&dmin_bits, blk + 2, 2);
        const float d    = f16_to_f32_dev(d_bits);
        const float dmin = f16_to_f32_dev(dmin_bits);
        const uint8_t* scales = blk + 4;

        uint8_t scA, mA, scB, mB;
        get_scale_min_k4(2 * p,     scales, &scA, &mA);
        get_scale_min_k4(2 * p + 1, scales, &scB, &mB);
        const float dA = d * scA, mdA = dmin * mA;
        const float dB = d * scB, mdB = dmin * mB;

        // One coalesced 4-byte load = 4 qs bytes = 8 weights.
        const uint32_t packed = ((const uint32_t*)(blk + 16))[lane];
        const float* xi = x + b * 256;
        const float4 xa = *(const float4*)(xi + xa0);
        const float4 xb = *(const float4*)(xi + xb0);

        // low nibbles → sub A (weights j0..j0+3), high nibbles → sub B
        acc += xa.x * (dA * (float)( packed        & 0xF) - mdA);
        acc += xa.y * (dA * (float)((packed >>  8) & 0xF) - mdA);
        acc += xa.z * (dA * (float)((packed >> 16) & 0xF) - mdA);
        acc += xa.w * (dA * (float)((packed >> 24) & 0xF) - mdA);
        acc += xb.x * (dB * (float)((packed >>  4) & 0xF) - mdB);
        acc += xb.y * (dB * (float)((packed >> 12) & 0xF) - mdB);
        acc += xb.z * (dB * (float)((packed >> 20) & 0xF) - mdB);
        acc += xb.w * (dB * (float)((packed >> 28) & 0xF) - mdB);
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q6_K ──────────────────────────────────────────────────────────────────────
// Block: [ql: 128 B][qh: 64 B][scales: 16 × i8][d: f16] = 210 bytes.

__device__ __forceinline__ void gemv_q6_k_body(
    const uint8_t* rd, const float* x, float& acc, int bpr, int lane)
{
    // Cooperative: lane = weight index l in [0,31]; all 32 lanes work every block.
    // Each lane owns {l, l+32, l+64, l+96} within each 128-weight chunk (2/block),
    // so the warp is always full — the old "one block per lane" left 18/32 lanes
    // idle when bpr < 32 (e.g. lm_head in_dim=3584 → bpr=14). warp_reduce sums the
    // per-lane partials afterwards. Loads stay coalesced (lane l reads byte l).
    const int l  = lane;
    const int is = l >> 4;
    for (int b = 0; b < bpr; b++) {
        const uint8_t* blk = rd + b * 210;
        const int8_t*  sc_base = (const int8_t*)(blk + 192);
        uint16_t d_bits; memcpy(&d_bits, blk + 208, 2);
        const float d = f16_to_f32_dev(d_bits);
        const float* xi = x + b * 256;
        #pragma unroll 2
        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8_t* ql = blk + chunk * 64;
            const uint8_t* qh = blk + 128 + chunk * 32;
            const int8_t*  sc = sc_base + chunk * 8;
            const float* xic = xi + chunk * 128;
            const uint8_t qll = ql[l];
            const uint8_t qlh = ql[l + 32];
            const uint8_t qhv = qh[l];
            int q1 = (int)((qll & 0x0F) | (((qhv >> 0) & 3) << 4)) - 32;
            int q2 = (int)((qlh & 0x0F) | (((qhv >> 2) & 3) << 4)) - 32;
            int q3 = (int)((qll >>   4) | (((qhv >> 4) & 3) << 4)) - 32;
            int q4 = (int)((qlh >>   4) | (((qhv >> 6) & 3) << 4)) - 32;
            acc += xic[l +  0] * d * (float)sc[is + 0] * (float)q1;
            acc += xic[l + 32] * d * (float)sc[is + 2] * (float)q2;
            acc += xic[l + 64] * d * (float)sc[is + 4] * (float)q3;
            acc += xic[l + 96] * d * (float)sc[is + 6] * (float)q4;
        }
    }
}

__global__ void gemv_q6_k(const uint8_t* __restrict__ data,
                          const float*   __restrict__ x,
                          float*         __restrict__ y,
                          int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const int bpr = in_dim / 256;
    float acc = 0.f;
    gemv_q6_k_body(data + (size_t)row * bpr * 210, x, acc, bpr, lane);
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q6_K v4: 2 rows per block, 64 threads — doubles warp occupancy ────────────
__global__ void gemv_q6_k_v4(const uint8_t* __restrict__ data,
                               const float*   __restrict__ x,
                               float*         __restrict__ y,
                               int in_dim, int out_dim)
{
    const int row  = blockIdx.x * 2 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= out_dim) return;
    const int bpr = in_dim / 256;
    float acc = 0.f;
    gemv_q6_k_body(data + (size_t)row * bpr * 210, x, acc, bpr, lane);
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Q8_0 ──────────────────────────────────────────────────────────────────────
// Block: [d: f16][qs: 32 × i8] = 34 bytes, 32 values.

__global__ void gemv_q8_0(const uint8_t* __restrict__ data,
                           const float*   __restrict__ x,
                           float*         __restrict__ y,
                           int in_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const int BLOCK = 32, BSIZ = 34;
    const int bpr = in_dim / BLOCK;
    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;

    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits; memcpy(&d_bits, blk, 2);
        float d = f16_to_f32_dev(d_bits);
        const int8_t* qs = (const int8_t*)(blk + 2);
        const float*  xi = x + b * BLOCK;
        #pragma unroll 32
        for (int k = 0; k < 32; k++)
            acc += xi[k] * d * (float)qs[k];
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── Full GPU forward pass kernels ─────────────────────────────────────────────

// vec_add: dst[i] += src[i]
__global__ void vec_add_f32(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// vec_scale_add: dst[i] += scale * src[i]
__global__ void vec_scale_add_f32(float* dst, const float* src, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += scale * src[i];
}

// vec_zero: dst[i] = 0
__global__ void vec_zero_f32(float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = 0.f;
}

// RMS norm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
__global__ void rms_norm_f32(float* out, const float* x, const float* w, int n) {
    extern __shared__ float sm[];
    float local = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];
    sm[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sm[0] / (float)n + 1e-5f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * rms * w[i];
}

// RoPE YARN (NeoX style): modifies x[head * head_dim .. + head_dim] in-place.
// Launch: grid(n_heads), block(head_dim/2)
__global__ void rope_yarn_f32(float* x, int head_dim, int pos,
                               float theta, float yarn_scale, int yarn_orig_ctx, int neox)
{
    const float PI = 3.14159265358979f;
    int i    = threadIdx.x;        // 0 .. head_dim/2
    int half = head_dim >> 1;
    float* xh = x + blockIdx.x * head_dim;

    float std_inv = 1.f / powf(theta, 2.f * (float)i / (float)head_dim);
    float eff_inv = std_inv;
    if (yarn_scale > 1.f && yarn_orig_ctx > 0) {
        float hfw     = (float)yarn_orig_ctx / 32.f;   // beta_fast = 32
        float lfw     = (float)yarn_orig_ctx / 1.f;    // beta_slow = 1
        float wavelen = 2.f * PI / std_inv;
        float ramp    = fmaxf(0.f, fminf(1.f, (lfw - wavelen) / (lfw - hfw)));
        eff_inv = ramp * std_inv + (1.f - ramp) * (std_inv / yarn_scale);
    }
    float angle = (float)pos * eff_inv;
    float s, c;
    sincosf(angle, &s, &c);
    // NEOX (qwen2/gpt-oss/olmoe): rotate halves (i, i+half).
    // NORM (llama/mistral): rotate adjacent pairs (2i, 2i+1) — their GGUF Q/K weights are
    // stored for this convention, so applying NEOX corrupts attention.
    int i0 = neox ? i        : 2 * i;
    int i1 = neox ? i + half : 2 * i + 1;
    float x0 = xh[i0];
    float x1 = xh[i1];
    xh[i0] = x0 * c - x1 * s;
    xh[i1] = x0 * s + x1 * c;
}

// Append K and V into their KV caches at sequence position pos.
// k_cache, v_cache: [max_seq × kv_dim]
__global__ void kv_append_f32(float* k_cache, float* v_cache,
                               const float* k, const float* v,
                               int pos, int kv_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        k_cache[pos * kv_dim + i] = k[i];
        v_cache[pos * kv_dim + i] = v[i];
    }
}

// Attention scores for one head: scores[t] = dot(q, k_cache[t][kv_head_off:]) * scale
// grid(seq_len), block(32) — one warp per past token
__global__ void attn_score_f32(float* scores, const float* q, const float* k_cache,
                                int head_dim, int kv_dim, int kv_head_off, float scale)
{
    int t    = blockIdx.x;
    int lane = threadIdx.x;
    const float* kt = k_cache + t * kv_dim + kv_head_off;
    float acc = 0.f;
    for (int i = lane; i < head_dim; i += 32) acc += q[i] * kt[i];
    warp_reduce(acc);
    if (lane == 0) scores[t] = acc * scale;
}

// Softmax in-place; optionally adds a sink to the partition function (not to x).
// sink == -1e30f means no sink. One block, shared mem = blockDim.x * sizeof(float).
__global__ void softmax_sink_f32(float* x, int n, float sink) {
    extern __shared__ float sm[];
    float mx = -1e30f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, x[i]);
    sm[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    mx = fmaxf(sm[0], (sink > -1e29f) ? sink : -1e30f);
    __syncthreads();

    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    sm[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    float total = sm[0] + ((sink > -1e29f) ? expf(sink - mx) : 0.f);
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[i] /= total;
}

// Weighted V sum for one head: out[i] = sum_t(scores[t] * v_cache[t][kv_head_off+i])
// grid(head_dim), block(32)
__global__ void attn_values_f32(float* out, const float* scores, const float* v_cache,
                                  int seq_len, int kv_dim, int kv_head_off)
{
    int i    = blockIdx.x;
    int lane = threadIdx.x;
    float acc = 0.f;
    for (int t = lane; t < seq_len; t += 32)
        acc += scores[t] * v_cache[t * kv_dim + kv_head_off + i];
    warp_reduce(acc);
    if (lane == 0) out[i] = acc;
}

// GPT-oss SwiGLU: out[i] = (gate[i].min(7) / (1+exp(-1.702*gate[i].min(7)))) * (up[i].clamp(-7,7)+1)
__global__ void swiglu_oai_f32(float* out, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float ALPHA = 1.702f, LIMIT = 7.f;
    float x = fminf(gate[i], LIMIT);
    float y = fmaxf(-LIMIT, fminf(LIMIT, up[i]));
    out[i] = (x / (1.f + expf(-ALPHA * x))) * (y + 1.f);
}

// LLaMA/Qwen dense SwiGLU: out[i] = silu(gate[i]) * up[i].
__global__ void swiglu_f32(float* out, const float* gate, const float* up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.f + expf(-g))) * up[i];
}

// Embed lookup Q5_1: dequant one row, grid(embed_dim/32), block(32)
__global__ void embed_lookup_q5_1(float* out, const uint8_t* data,
                                   int token, int embed_dim)
{
    const int BSIZ = 24;
    int b = blockIdx.x, k = threadIdx.x;
    int bpr = embed_dim / 32;
    const uint8_t* blk = data + (size_t)token * bpr * BSIZ + b * BSIZ;
    uint16_t d_bits, m_bits;
    memcpy(&d_bits, blk,   2);
    memcpy(&m_bits, blk+2, 2);
    float d = f16_to_f32_dev(d_bits), m = f16_to_f32_dev(m_bits);
    uint32_t qh; memcpy(&qh, blk+4, 4);
    const uint8_t* qs = blk + 8;
    float val;
    if (k < 16) {
        uint32_t xh = ((qh >> k) << 4) & 0x10;
        val = (float)((qs[k] & 0x0F) | xh) * d + m;
    } else {
        int j = k - 16;
        uint32_t xh = (qh >> (j + 12)) & 0x10;
        val = (float)((qs[j] >> 4) | xh) * d + m;
    }
    out[b * 32 + k] = val;
}

// Embed lookup IQ4_NL: grid(embed_dim/32), block(32)
__global__ void embed_lookup_iq4nl(float* out, const uint8_t* data,
                                    int token, int embed_dim)
{
    const int BSIZ = 18;
    int b = blockIdx.x, k = threadIdx.x;
    int bpr = embed_dim / 32;
    const uint8_t* blk = data + (size_t)token * bpr * BSIZ + b * BSIZ;
    uint16_t d_bits; memcpy(&d_bits, blk, 2);
    float d = f16_to_f32_dev(d_bits);
    const uint8_t* qs = blk + 2;
    int byte_idx = k & 15;
    float val = d * iq4nl_table[(k < 16) ? (qs[byte_idx] & 0x0F) : (qs[byte_idx] >> 4)];
    out[b * 32 + k] = val;
}

// Embed lookup BF16: grid((embed_dim+255)/256), block(256)
__global__ void embed_lookup_bf16(float* out, const uint16_t* data,
                                   int token, int embed_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= embed_dim) return;
    uint32_t bits = (uint32_t)data[(size_t)token * embed_dim + i] << 16;
    memcpy(&out[i], &bits, 4);
}

// Q5_K fused GEMV — Q4_K layout plus a 1-bit-per-weight high plane (qh).
// block = 176B: d(f16,2) dmin(f16,2) scales(12) qh(32) qs(128). 5-bit weight = qs nibble | (qh bit << 4).
__global__ void gemv_q5_k_v4(const uint8_t* __restrict__ data,
                              const float*   __restrict__ x,
                              float*         __restrict__ y,
                              int in_dim, int out_dim)
{
    const int row  = blockIdx.x * 2 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= out_dim) return;
    const int BSIZ = 176;
    const int bpr  = in_dim >> 8;
    const uint8_t* rd = data + (size_t)row * bpr * BSIZ;

    const int p   = lane >> 3;               // group 0..3
    const int j0  = (lane << 2) & 31;        // 0,4,...,28 — qh byte offset within the 32-byte plane
    const int xa0 = (2 * p)     * 32 + j0;
    const int xb0 = (2 * p + 1) * 32 + j0;
    const uint8_t u1 = (uint8_t)(1u << (2 * p));      // qh bit for low-nibble sub-block
    const uint8_t u2 = (uint8_t)(1u << (2 * p + 1));  // qh bit for high-nibble sub-block

    float acc = 0.f;
    for (int b = 0; b < bpr; b++) {
        const uint8_t* blk = rd + b * BSIZ;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    blk,     2);
        memcpy(&dmin_bits, blk + 2, 2);
        const float d    = f16_to_f32_dev(d_bits);
        const float dmin = f16_to_f32_dev(dmin_bits);
        const uint8_t* scales = blk + 4;

        uint8_t scA, mA, scB, mB;
        get_scale_min_k4(2 * p,     scales, &scA, &mA);
        get_scale_min_k4(2 * p + 1, scales, &scB, &mB);
        const float dA = d * scA, mdA = dmin * mA;
        const float dB = d * scB, mdB = dmin * mB;

        const uint32_t packed = *(const uint32_t*)(blk + 48 + lane * 4);  // 4 qs bytes
        const uint32_t qhp    = *(const uint32_t*)(blk + 16 + j0);        // 4 qh bytes (l=j0..j0+3)
        const uint8_t h0 = (qhp      ) & 0xFF, h1 = (qhp >>  8) & 0xFF;
        const uint8_t h2 = (qhp >> 16) & 0xFF, h3 = (qhp >> 24) & 0xFF;

        const float* xi = x + b * 256;
        const float4 xa = *(const float4*)(xi + xa0);
        const float4 xb = *(const float4*)(xi + xb0);

        acc += xa.x * (dA * ((float)( packed        & 0xF) + ((h0 & u1) ? 16.f : 0.f)) - mdA);
        acc += xa.y * (dA * ((float)((packed >>  8) & 0xF) + ((h1 & u1) ? 16.f : 0.f)) - mdA);
        acc += xa.z * (dA * ((float)((packed >> 16) & 0xF) + ((h2 & u1) ? 16.f : 0.f)) - mdA);
        acc += xa.w * (dA * ((float)((packed >> 24) & 0xF) + ((h3 & u1) ? 16.f : 0.f)) - mdA);
        acc += xb.x * (dB * ((float)((packed >>  4) & 0xF) + ((h0 & u2) ? 16.f : 0.f)) - mdB);
        acc += xb.y * (dB * ((float)((packed >> 12) & 0xF) + ((h1 & u2) ? 16.f : 0.f)) - mdB);
        acc += xb.z * (dB * ((float)((packed >> 20) & 0xF) + ((h2 & u2) ? 16.f : 0.f)) - mdB);
        acc += xb.w * (dB * ((float)((packed >> 28) & 0xF) + ((h3 & u2) ? 16.f : 0.f)) - mdB);
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── IQ grid codebooks (uploaded once from the Rust tables) ───────────────────
__device__ uint32_t g_iq3s_grid[512];
__device__ uint64_t g_iq2s_grid[1024];
__device__ uint64_t g_iq2xxs_grid[256];
__device__ uint64_t g_iq2xs_grid[512];
__device__ uint8_t  g_ksigns[128];

// Q2_K fused GEMV — one warp per row. block = 84B: scales(16) qs(64) d(f16,2) dmin(f16,2).
// weight = d*scale*(2-bit q) - dmin*min.
__global__ void gemv_q2_k(const uint8_t* __restrict__ data,
                          const float* __restrict__ x, float* __restrict__ y,
                          int in_dim, int out_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= out_dim) return;
    const int bpr = in_dim >> 8;
    const uint8_t* rd = data + (size_t)row * bpr * 84;
    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * 84;
        const uint8_t* scales = blk;
        const uint8_t* qs = blk + 16;
        uint16_t db, dmb; memcpy(&db, blk + 80, 2); memcpy(&dmb, blk + 82, 2);
        const float d = f16_to_f32_dev(db), dmin = f16_to_f32_dev(dmb);
        const float* xi = x + b * 256;
        int is = 0, k = 0;
        for (int n = 0; n < 256; n += 128) {
            const uint8_t* q = qs + (n / 128) * 32;
            for (int j = 0; j < 4; j++) {
                const int shift = 2 * j;
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xf), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) acc += (dl * (float)((q[l]    >> shift) & 3) - ml) * xi[k++];
                sc = scales[is++];
                dl = d * (sc & 0xf); ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; l++) acc += (dl * (float)((q[l+16] >> shift) & 3) - ml) * xi[k++];
            }
        }
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// IQ2_XXS fused GEMV — block = 66B: d(f16,2) qs(64 = u16[32]). 4 grid idx + packed signs/scale per group.
__global__ void gemv_iq2_xxs(const uint8_t* __restrict__ data,
                             const float* __restrict__ x, float* __restrict__ y,
                             int in_dim, int out_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= out_dim) return;
    const int bpr = in_dim >> 8;
    const uint8_t* rd = data + (size_t)row * bpr * 66;
    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * 66;
        uint16_t dbits; memcpy(&dbits, blk, 2);
        const float d = f16_to_f32_dev(dbits);
        const uint8_t* qs = blk + 2;
        const float* xi = x + b * 256;
        int k = 0;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            const int bo = 8 * ib32;
            uint32_t a1; memcpy(&a1, qs + bo + 4, 4);
            const float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint64_t g = g_iq2xxs_grid[qs[bo + l]];
                const uint8_t signs = g_ksigns[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++) {
                    int8_t gv = (int8_t)((g >> (8*j)) & 0xFF);
                    acc += db * (float)gv * ((signs & (1 << j)) ? -1.f : 1.f) * xi[k++];
                }
            }
        }
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// IQ2_XS fused GEMV — block = 74B: d(f16,2) qs(64 = u16[32]) scales(8). q = 9-bit grid idx | 7-bit sign idx.
__global__ void gemv_iq2_xs(const uint8_t* __restrict__ data,
                            const float* __restrict__ x, float* __restrict__ y,
                            int in_dim, int out_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= out_dim) return;
    const int bpr = in_dim >> 8;
    const uint8_t* rd = data + (size_t)row * bpr * 74;
    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * 74;
        uint16_t dbits; memcpy(&dbits, blk, 2);
        const float d = f16_to_f32_dev(dbits);
        const uint8_t* qs = blk + 2;
        const uint8_t* sc = blk + 66;
        const float* xi = x + b * 256;
        int k = 0;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            const float db0 = d * (0.5f + (float)(sc[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (float)(sc[ib32] >> 4))  * 0.25f;
            for (int l = 0; l < 4; l++) {
                uint16_t q; memcpy(&q, qs + 2*(4*ib32 + l), 2);
                const uint64_t g = g_iq2xs_grid[q & 511];
                const uint8_t signs = g_ksigns[q >> 9];
                const float db = (l < 2) ? db0 : db1;
                for (int j = 0; j < 8; j++) {
                    int8_t gv = (int8_t)((g >> (8*j)) & 0xFF);
                    acc += db * (float)gv * ((signs & (1 << j)) ? -1.f : 1.f) * xi[k++];
                }
            }
        }
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// IQ2_S fused GEMV — one warp per row; direct port of dequant_iq2_s.
// block = 82B: d(f16,2) qs(64: 32 idx + 32 signs) qh(8) scales(8).
__global__ void gemv_iq2_s(const uint8_t* __restrict__ data,
                           const float* __restrict__ x, float* __restrict__ y,
                           int in_dim, int out_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= out_dim) return;
    const int bpr = in_dim >> 8;
    const uint8_t* rd = data + (size_t)row * bpr * 82;
    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * 82;
        uint16_t dbits; memcpy(&dbits, blk, 2);
        const float d = f16_to_f32_dev(dbits);
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t* sc = blk + 74;
        const float* xi = x + b * 256;
        int k = 0;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            const float db0 = d * (0.5f + (float)(sc[ib32] & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (float)(sc[ib32] >> 4))  * 0.25f;
            for (int l = 0; l < 4; l++) {
                const float db = (l < 2) ? db0 : db1;
                const int idx = qs[4*ib32 + l] | ((qh[ib32] << (8 - 2*l)) & 0x300);
                const uint64_t g = g_iq2s_grid[idx];
                const uint8_t signs = qs[32 + 4*ib32 + l];
                for (int j = 0; j < 8; j++) {
                    int8_t gv = (int8_t)((g >> (8*j)) & 0xFF);
                    acc += db * (float)gv * ((signs & (1 << j)) ? -1.f : 1.f) * xi[k++];
                }
            }
        }
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// IQ3_S fused GEMV — one warp per row; each lane dequantizes whole 256-weight blocks
// (direct port of dequant_iq3_s) and accumulates its blocks' dot products.
// block = 110B: d(f16,2) qs(64) qh(8) signs(32) scales(4).
__global__ void gemv_iq3_s(const uint8_t* __restrict__ data,
                           const float* __restrict__ x, float* __restrict__ y,
                           int in_dim, int out_dim)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;             // 0..31
    if (row >= out_dim) return;
    const int bpr = in_dim >> 8;
    const uint8_t* rd = data + (size_t)row * bpr * 110;
    float acc = 0.f;
    for (int b = lane; b < bpr; b += 32) {
        const uint8_t* blk = rd + b * 110;
        uint16_t dbits; memcpy(&dbits, blk, 2);
        const float d  = f16_to_f32_dev(dbits);
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t* sg = blk + 74;
        const uint8_t* sc = blk + 106;
        const float* xi = x + b * 256;
        int k = 0;
        for (int it = 0; it < 4; it++) {
            const int qo = it*16, so = it*8, ho = it*2;
            const float db1 = d * (1.f + 2.f * (float)(sc[it] & 0xf));
            const float db2 = d * (1.f + 2.f * (float)(sc[it] >> 4));
            for (int half = 0; half < 2; half++) {
                const float db = half == 0 ? db1 : db2;
                const int qhb  = qh[ho + half];
                const int qbase = qo + half*8;
                const int sbase = so + half*4;
                for (int l = 0; l < 4; l++) {
                    const int i0 = qs[qbase + 2*l]     | ((qhb << (8 - 2*l)) & 256);
                    const int i1 = qs[qbase + 2*l + 1] | ((qhb << (7 - 2*l)) & 256);
                    const uint32_t g1 = g_iq3s_grid[i0];
                    const uint32_t g2 = g_iq3s_grid[i1];
                    const uint8_t s = sg[sbase + l];
                    for (int j = 0; j < 4; j++) {
                        int8_t gv = (int8_t)((g1 >> (8*j)) & 0xFF);
                        acc += db * (float)gv * ((s & (1 << j))     ? -1.f : 1.f) * xi[k++];
                    }
                    for (int j = 0; j < 4; j++) {
                        int8_t gv = (int8_t)((g2 >> (8*j)) & 0xFF);
                        acc += db * (float)gv * ((s & (1 << (j+4))) ? -1.f : 1.f) * xi[k++];
                    }
                }
            }
        }
    }
    warp_reduce(acc);
    if (lane == 0) y[row] = acc;
}

// ── C-callable dispatch ───────────────────────────────────────────────────────

extern "C" {

void cuda_set_iq3s_grid(const uint32_t* host) {
    cudaMemcpyToSymbol(g_iq3s_grid, host, 512 * sizeof(uint32_t));
}
void cuda_set_iq2s_grid(const uint64_t* host) {
    cudaMemcpyToSymbol(g_iq2s_grid, host, 1024 * sizeof(uint64_t));
}
void cuda_set_iq2xxs_grid(const uint64_t* host) {
    cudaMemcpyToSymbol(g_iq2xxs_grid, host, 256 * sizeof(uint64_t));
}
void cuda_set_iq2xs_grid(const uint64_t* host) {
    cudaMemcpyToSymbol(g_iq2xs_grid, host, 512 * sizeof(uint64_t));
}
void cuda_set_ksigns(const uint8_t* host) {
    cudaMemcpyToSymbol(g_ksigns, host, 128);
}
void cuda_mem_info(size_t* free_b, size_t* total_b) {
    cudaMemGetInfo(free_b, total_b);
}

void cuda_gemv(uint32_t ggml_type,
               const uint8_t* d_data,
               const float*   d_x,
               float*         d_y,
               int in_dim, int out_dim)
{
    switch (ggml_type) {
        case  0: { dim3 g(out_dim), b(32); gemv_f32  <<<g,b>>>(d_data,d_x,d_y,in_dim); break; }
        case  1: { dim3 g(out_dim), b(32); gemv_f16  <<<g,b>>>(d_data,d_x,d_y,in_dim); break; }
        // Q5_1: v4 — 64-thread blocks (2 rows each) for full warp occupancy
        case  7: {
            dim3 grid((out_dim + 1) / 2), block(64);
            gemv_q5_1_v4<<<grid, block>>>(d_data, d_x, d_y, in_dim, out_dim);
            break;
        }
        case  6: { dim3 grid((out_dim+1)/2), block(64); gemv_q5_0_v4<<<grid,block>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case  8: { dim3 g(out_dim), b(32); gemv_q8_0 <<<g,b>>>(d_data,d_x,d_y,in_dim); break; }
        case 10: { dim3 g(out_dim), b(32); gemv_q2_k<<<g,b>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case 12: { dim3 grid((out_dim+1)/2), block(64); gemv_q4_k_v4<<<grid,block>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case 13: { dim3 grid((out_dim+1)/2), block(64); gemv_q5_k_v4<<<grid,block>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case 14: { dim3 grid((out_dim+1)/2), block(64); gemv_q6_k_v4<<<grid,block>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case 21: { dim3 g(out_dim), b(32); gemv_iq3_s<<<g,b>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case 22: { dim3 g(out_dim), b(32); gemv_iq2_s<<<g,b>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case 16: { dim3 g(out_dim), b(32); gemv_iq2_xxs<<<g,b>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case 17: { dim3 g(out_dim), b(32); gemv_iq2_xs<<<g,b>>>(d_data,d_x,d_y,in_dim,out_dim); break; }
        case 20: { dim3 g(out_dim), b(32); gemv_iq4nl<<<g,b>>>(d_data,d_x,d_y,in_dim); break; }
        case 30: { dim3 g(out_dim), b(32); gemv_bf16 <<<g,b>>>(d_data,d_x,d_y,in_dim); break; }
        default: {
            cudaMemset(d_y, 0, (size_t)out_dim * sizeof(float));
            fprintf(stderr, "cuda_gemv: unsupported ggml_type %u\n", ggml_type);
            break;
        }
    }
}

// Batched expert GEMV: 4 experts run simultaneously in a 2D grid.
// For gate/up: xi0..xi3 all point to the same input (d_norm_out).
// For down:    xi0..xi3 point to per-expert swiglu outputs.
// Unused expert slots (eid >= n_act) are masked out by the kernel.
void cuda_gemv_moe4(
    uint32_t ggml_type,
    const uint8_t* w0, const uint8_t* w1,
    const uint8_t* w2, const uint8_t* w3,
    const float* xi0, const float* xi1,
    const float* xi2, const float* xi3,
    float* o0, float* o1, float* o2, float* o3,
    int n_act, int in_dim, int out_dim)
{
    if (ggml_type == 7) {
        dim3 grid((out_dim + 1) / 2, n_act), block(64);
        gemv_q5_1_v4_moe4<<<grid, block>>>(
            w0, w1, w2, w3, xi0, xi1, xi2, xi3,
            o0, o1, o2, o3, n_act, in_dim, out_dim);
    } else {
        // Fallback: run serially via cuda_gemv for non-Q5_1 types
        const uint8_t* ws[4]  = {w0, w1, w2, w3};
        const float*   xis[4] = {xi0, xi1, xi2, xi3};
        float*         os[4]  = {o0, o1, o2, o3};
        for (int e = 0; e < n_act; e++)
            cuda_gemv(ggml_type, ws[e], xis[e], os[e], in_dim, out_dim);
    }
}

void* cuda_alloc(size_t bytes) {
    void* p = nullptr;
    cudaMalloc(&p, bytes);
    return p;
}

void cuda_drop(void* p) { cudaFree(p); }

void cuda_h2d(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

void cuda_d2h(void* dst, const void* src, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

void cuda_sync() { cudaDeviceSynchronize(); }

// ── GPU forward pass helpers ──────────────────────────────────────────────────

void cuda_vec_add(float* dst, const float* src, int n) {
    vec_add_f32<<<(n+255)/256, 256>>>(dst, src, n);
}
void cuda_vec_scale_add(float* dst, const float* src, float scale, int n) {
    vec_scale_add_f32<<<(n+255)/256, 256>>>(dst, src, scale, n);
}
void cuda_vec_zero(float* dst, int n) {
    vec_zero_f32<<<(n+255)/256, 256>>>(dst, n);
}
void cuda_rms_norm(float* out, const float* x, const float* w, int n) {
    int blk = (n < 512) ? 64 : 256;
    rms_norm_f32<<<1, blk, blk * sizeof(float)>>>(out, x, w, n);
}
void cuda_rope_yarn(float* x, int n_heads, int head_dim, int pos,
                    float theta, float yarn_scale, int yarn_orig_ctx, int neox)
{
    rope_yarn_f32<<<n_heads, head_dim/2>>>(x, head_dim, pos, theta, yarn_scale, yarn_orig_ctx, neox);
}
void cuda_kv_append(float* k_cache, float* v_cache, const float* k, const float* v,
                    int pos, int kv_dim)
{
    kv_append_f32<<<(kv_dim+255)/256, 256>>>(k_cache, v_cache, k, v, pos, kv_dim);
}
void cuda_attn_score(float* scores, const float* q, const float* k_cache,
                     int seq_len, int head_dim, int kv_dim, int kv_head_off, float scale)
{
    attn_score_f32<<<seq_len, 32>>>(scores, q, k_cache, head_dim, kv_dim, kv_head_off, scale);
}
void cuda_softmax_sink(float* x, int n, float sink) {
    int blk = (n < 512) ? 64 : 256;
    softmax_sink_f32<<<1, blk, blk * sizeof(float)>>>(x, n, sink);
}
void cuda_attn_values(float* out, const float* scores, const float* v_cache,
                      int seq_len, int head_dim, int kv_dim, int kv_head_off)
{
    attn_values_f32<<<head_dim, 32>>>(out, scores, v_cache, seq_len, kv_dim, kv_head_off);
}
void cuda_swiglu_oai(float* out, const float* gate, const float* up, int n) {
    swiglu_oai_f32<<<(n+255)/256, 256>>>(out, gate, up, n);
}
void cuda_swiglu(float* out, const float* gate, const float* up, int n) {
    swiglu_f32<<<(n+255)/256, 256>>>(out, gate, up, n);
}
void cuda_embed_lookup(float* out, const uint8_t* data, uint32_t ggml_type,
                        int token, int embed_dim)
{
    int bpr = embed_dim / 32;
    switch (ggml_type) {
        case  7: embed_lookup_q5_1 <<<bpr, 32>>>(out, data, token, embed_dim); break;
        case 20: embed_lookup_iq4nl<<<bpr, 32>>>(out, data, token, embed_dim); break;
        case 30: {
            int blk = 256;
            embed_lookup_bf16<<<(embed_dim+blk-1)/blk, blk>>>(
                out, (const uint16_t*)data, token, embed_dim);
            break;
        }
    }
}

// ── Batched multi-head attention ──────────────────────────────────────────────
// These replace the per-head loop (64×3 = 192 kernel launches per layer) with
// 3 launches that process all heads in parallel.

// Attention scores: scores[head * max_seq + t] = dot(q[head], k_cache[t][kv_head]) * scale
// grid(n_heads, seq_len), block(32)
__global__ void attn_score_all_f32(float* scores, const float* q, const float* k_cache,
                                    int n_heads, int n_kv_heads, int seq_len,
                                    int head_dim, int kv_dim, int max_seq, float scale)
{
    int head = blockIdx.x;
    int t    = blockIdx.y;
    int lane = threadIdx.x;
    if (head >= n_heads || t >= seq_len) return;
    int kv_head = (head * n_kv_heads) / n_heads;
    const float* qh = q + head * head_dim;
    const float* kt = k_cache + (size_t)t * kv_dim + kv_head * head_dim;
    float acc = 0.f;
    for (int i = lane; i < head_dim; i += 32) acc += qh[i] * kt[i];
    warp_reduce(acc);
    if (lane == 0) scores[head * max_seq + t] = acc * scale;
}

// In-place softmax with per-head attention sink for ALL heads.
// scores: [n_heads, max_seq], sinks: [n_heads]
// grid(n_heads), block(blk) with shared mem = blk * sizeof(float)
__global__ void softmax_sink_all_f32(float* scores, int seq_len, int max_seq,
                                      const float* sinks)
{
    extern __shared__ float sm[];
    int head = blockIdx.x;
    float* x = scores + head * max_seq;
    float  sink = sinks[head];

    float mx = -1e30f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) mx = fmaxf(mx, x[i]);
    sm[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x+s]);
        __syncthreads();
    }
    mx = fmaxf(sm[0], (sink > -1e29f) ? sink : -1e30f);
    __syncthreads();

    float sum = 0.f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    sm[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x+s];
        __syncthreads();
    }
    float total = sm[0] + ((sink > -1e29f) ? expf(sink - mx) : 0.f);
    __syncthreads();
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) x[i] /= total;
}

// Weighted V-sum for ALL heads: out[head*head_dim + i] = sum_t(scores[head,t]*v[t,kv_head,i])
// grid(n_heads, head_dim), block(32)
__global__ void attn_values_all_f32(float* out, const float* scores, const float* v_cache,
                                     int n_heads, int n_kv_heads, int seq_len,
                                     int head_dim, int kv_dim, int max_seq)
{
    int head    = blockIdx.x;
    int i       = blockIdx.y;
    int lane    = threadIdx.x;
    int kv_head = (head * n_kv_heads) / n_heads;
    const float* s = scores + head * max_seq;
    float acc = 0.f;
    for (int t = lane; t < seq_len; t += 32)
        acc += s[t] * v_cache[(size_t)t * kv_dim + kv_head * head_dim + i];
    warp_reduce(acc);
    if (lane == 0) out[head * head_dim + i] = acc;
}

void cuda_attn_score_all(float* scores, const float* q, const float* k_cache,
                          int n_heads, int n_kv_heads, int seq_len,
                          int head_dim, int kv_dim, int max_seq, float scale)
{
    dim3 grid(n_heads, seq_len);
    attn_score_all_f32<<<grid, 32>>>(scores, q, k_cache, n_heads, n_kv_heads,
                                      seq_len, head_dim, kv_dim, max_seq, scale);
}

void cuda_softmax_sink_all(float* scores, int n_heads, int seq_len,
                            int max_seq, const float* sinks)
{
    int blk = (seq_len <= 64) ? 64 : 256;
    softmax_sink_all_f32<<<n_heads, blk, blk * sizeof(float)>>>(
        scores, seq_len, max_seq, sinks);
}

void cuda_attn_values_all(float* out, const float* scores, const float* v_cache,
                           int n_heads, int n_kv_heads, int seq_len,
                           int head_dim, int kv_dim, int max_seq)
{
    dim3 grid(n_heads, head_dim);
    attn_values_all_f32<<<grid, 32>>>(out, scores, v_cache, n_heads, n_kv_heads,
                                       seq_len, head_dim, kv_dim, max_seq);
}

} // extern "C"
