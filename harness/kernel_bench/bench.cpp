// Standalone microbench: AVX2 (reference) vs AVX-512 (new) for
// ggml_vec_dot_q2_K_q8_K and ggml_vec_dot_q3_K_q8_K
//
// Build:
//   source ~/miniforge3/etc/profile.d/conda.sh && conda activate llamacpp-cu131
//   g++ -O3 -march=native -fopenmp bench.cpp -o bench
//   taskset -c 0 ./bench
//
// NOTE: -fopenmp is only linked for convenience; the timed kernels are single
// thread and we pin with taskset -c 0.

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>

// ----------------------------------------------------------------------------
// ggml block layout (copied verbatim from ggml-common.h, QK_K=256)
// ----------------------------------------------------------------------------
#define QK_K 256
typedef uint16_t ggml_half;

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct { ggml_half d; ggml_half dmin; };
        uint32_t dm;
    };
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4, "q2_K size");

typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_half) + QK_K/4 + QK_K/8 + 12, "q3_K size");

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "q8_K size");

#define GGML_RESTRICT __restrict__
#define UNUSED(x) (void)(x)

// FP16 -> FP32 via F16C (enabled by -march=native on Zen5)
static inline float GGML_CPU_FP16_TO_FP32(ggml_half h) {
    return _cvtsh_ss(h);
}

// some compilers don't provide _mm256_set_m128i
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

// ============================================================================
//  Q2_K  AVX2 reference (verbatim from quants.c)
// ============================================================================
void ggml_vec_dot_q2_K_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q2_K * GGML_RESTRICT x = (const block_q2_K *)vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);
}

// ============================================================================
//  Q2_K  AVX-512  (NEW: 256->512 widening)
// ----------------------------------------------------------------------------
// Strategy:
//   One Q2_K super-block = 256 weights = 64 bytes of qs = exactly one zmm.
//   The 256 weights split into 16 sub-blocks of 16 elements, each with its own
//   4-bit scale. The AVX2 code processes the super-block in 2 halves of 128
//   weights, with 4 maddubs of 32 lanes each. We process the WHOLE super-block
//   per shift in 512-bit lanes: one zmm of q2bits -> 4 shifts -> 4 maddubs of
//   64 lanes each = the full 256 weights, halving instruction count vs AVX2.
//
//   Lane layout: a 512-bit load of qs holds bytes [0..63]. Byte b (b in 0..63)
//   holds, at shift s (s in 0,2,4,6), the weight for element  (s/2)*64 + b ...
//   matching the q8 load order [0..63],[64..127],[128..191],[192..255].
//   So q8 for shift s is the zmm covering elements [s/2*64 .. s/2*64+63].
//
//   The 16 per-sub-block scales must be broadcast to the right 16-element
//   groups. After maddubs we have 16-bit products in 32 lanes per zmm; the
//   madd_epi16 with a per-element-pair scale collapses to 32-bit. We build a
//   512-bit scale vector for each shift by shuffling 16-bit scales into the
//   right byte positions (same idea as the AVX2 get_scale_shuffle_q3k, widened).
//
// VNNI note: vpdpbusd would fuse maddubs+madd-by-ones into one op, but here the
// post-maddubs values must each be multiplied by a *per-16-element scale* before
// the horizontal add-by-pairs. A naive vpdpbusd folds with implicit ones, which
// does NOT apply the scale, so it is mathematically invalid here. We keep the
// maddubs + (scale)madd structure. VNNI is therefore NOT used for Q2_K.
// ============================================================================
void ggml_vec_dot_q2_K_q8_K_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const block_q2_K * GGML_RESTRICT x = (const block_q2_K *)vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    const __m512i m3_512 = _mm512_set1_epi8(3);
    const __m128i m4     = _mm_set1_epi8(0xF);

    // The 64-byte qs of one super-block = [ half0 (bytes 0..31) | half1 (bytes
    // 32..63) ]. For shift s (0,2,4,6 -> idx 0..3), half0 weights pair with q8
    // elements [s_idx*32 .. +31] and half1 with q8 elements [128 + s_idx*32 ..].
    // So per shift we build:
    //   q8v   = [ q8(s_idx*32) | q8(128 + s_idx*32) ]   (256|256)
    //   q2v   = (whole64 >> 2*s_idx) & 3                 (already [half0|half1])
    // After maddubs (32x int16 per zmm): lanes 0..7 of EACH 256-half come from
    // bytes 0..15 (scale sc[is]), lanes 8..15 from bytes 16..31 (scale sc[is+1]).
    // AVX2 picks these with get_scale_shuffle_q3k(s_idx) against scales[k] where
    // scales[0]=dup(l_scales)=sub 0..7, scales[1]=dup(h_scales)=sub 8..15.
    // So the 512 scale operand = [ scales[0] | scales[1] ] and the shuffle =
    // get_scale_shuffle_q3k(s_idx) duplicated across both 256-halves.

    __m512 acc = _mm512_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const float d    =  y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8   = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);

        // mins contribution: -dmin * sum(mins16 * bsums)  (exactly AVX2, 256-bit)
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));
        const __m512  prodf = _mm512_castps256_ps512(_mm256_cvtepi32_ps(prod));
        acc = _mm512_mask_add_ps(acc, 0x00FF, acc, _mm512_mul_ps(_mm512_set1_ps(dmin), prodf));

        // 512 scale operand = [ dup(l_scales) | dup(h_scales) ] (sub 0..7 | 8..15)
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales0_256 = MM256_SET_M128I(l_scales, l_scales);
        const __m256i scales1_256 = MM256_SET_M128I(h_scales, h_scales);
        const __m512i scale_512 = _mm512_inserti64x4(_mm512_castsi256_si512(scales0_256), scales1_256, 1);

        // whole 64-byte qs = [half0 | half1]
        const __m512i q2all = _mm512_loadu_si512((const __m512i*)q2);

        __m512i sumi = _mm512_setzero_si512();

        for (int sidx = 0; sidx < 4; ++sidx) {
            const __m512i q2v = _mm512_and_si512(_mm512_srli_epi16(q2all, 2*sidx), m3_512);
            const __m256i q8lo = _mm256_loadu_si256((const __m256i*)(q8 +       sidx*32));
            const __m256i q8hi = _mm256_loadu_si256((const __m256i*)(q8 + 128 + sidx*32));
            const __m512i q8v  = _mm512_inserti64x4(_mm512_castsi256_si512(q8lo), q8hi, 1);

            __m512i p = _mm512_maddubs_epi16(q2v, q8v);

            // scale shuffle: get_scale_shuffle_q3k(sidx) in BOTH 256-halves;
            // applied to scale_512 -> half0 picks from l_scales, half1 from h_scales
            const __m256i sh256 = get_scale_shuffle_q3k(sidx);
            const __m512i shuf  = _mm512_inserti64x4(_mm512_castsi256_si512(sh256), sh256, 1);
            p = _mm512_madd_epi16(_mm512_shuffle_epi8(scale_512, shuf), p);

            sumi = _mm512_add_epi32(sumi, p);
        }

        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }

    *s = _mm512_reduce_add_ps(acc);
}

// ============================================================================
//  Q3_K  AVX2 reference (verbatim from quants.c)
// ============================================================================
void ggml_vec_dot_q3_K_q8_K_avx2(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = (const block_q3_K *)vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();
    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;
        int is  = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi  = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
            is += 4;
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);
}

// ============================================================================
//  Q3_K  AVX-512  (NEW: 256->512 widening)
// ----------------------------------------------------------------------------
// Same structure as AVX2 but each of the two j-iterations is fused into one
// 512-bit pass. The AVX2 inner loop runs j=0,1 (QK_K/128=2), each handling 128
// elements via 4 maddubs of 32 lanes. Here we process all 256 elements per
// super-block: low-2-bit and high-bit handling identical, just 512-bit.
//
// hbits: 32 bytes (QK_K/8). For the AVX2 code, the SAME 32-byte hbits vector is
// reused for both j-halves but with different bit indices (j*4 + k). In 512-bit
// we need hbits covering all 256 elements -> but hmask is only 32 bytes total.
// The 8 bits of each hmask byte map across the 8 (bit) positions; element e in
// [0..255] uses hmask byte (e & 31)?? Actually AVX2 reuses the 32-byte hbits
// for BOTH halves: q8 for half j covers elements [j*128 .. j*128+127] but the
// hbits 32-byte vector is the same and indexed by 'bit' = j*4+k. So the high
// bit for a 32-lane group at (j,k) is hmask_byte[lane] bit (j*4+k).
// To go 512-bit we must replicate the 32-byte hbits into a 64-byte zmm so the
// low 256 elements (j=0, q8_0..q8_3 lanes) and high 256 elements share lanes.
// But q8_0(512) = elements[0..63], q8_1=[64..127] (these are j=0 in AVX2),
// q8_2=[128..191], q8_3=[192..255] (these are j=1 in AVX2). The hmask byte for
// element e is hmask[e & 31] (the 32-byte mask tiles every 32 elements within a
// 256 super-block? NO). Careful: AVX2 q8_0 in half j is _mm256 of 32 elements;
// across half j it loads q8 sequentially: half0 -> elems 0..127, half1 ->
// 128..255. The hbits __m256i is the 32-byte hmask, and lane L (0..31) within a
// 32-element q8 group corresponds to hmask byte L. Since both halves reuse the
// same 32-byte hbits, hmask byte index = (element index) & 31. So to build the
// 512-bit q8_0 (elems 0..63) we need hmask bytes [0..31] then [0..31] again =
// broadcast the 32-byte hmask into a 64-byte zmm. Same for all four 512 groups.
//
// bit index per 512-group: q8_0(elems0..63) -> shift within byte = 0 (k=0,j=0)
// for its FIRST 32 lanes and ... wait, each 512 group spans 64 elements = two
// AVX2 32-lane groups with DIFFERENT bit. q8_0(0..63): lanes 0..31 are AVX2
// (j=0,k=0) bit0, lanes 32..63 are (j=0,k=1) bit1. So a single 512 group needs
// TWO different bit positions in its two 256-bit halves. We therefore build the
// high-bit operand with a per-half bit using a 64-byte 'mone<<bit' where the bit
// differs between the low and high 256-bit halves. Equivalent: process the
// q3 low bits as: group g (0..3) handles elements [g*64..g*64+63]; its low half
// (256-bit) uses q3 shift (2*( (g*64)/... )). This gets messy; simplest exact
// port: keep the j loop but make each iteration 512-bit by pairing (k,k+...).
//
// CLEANEST EXACT APPROACH (used below): mirror AVX2 lane-for-lane. We keep the
// 4 maddubs but widen each to operate on a 512-bit operand that concatenates the
// AVX2 j=0 and j=1 operands for the SAME k. i.e.
//   zmm_k = [ avx2_half0_k (256) | avx2_half1_k (256) ]
// q3 low bits: half0_k uses q3bits0 (bytes 0..31) shift 2k; half1_k uses
//   q3bits1 (bytes 32..63) shift 2k. So q3_512 = whole 64-byte qs load, shift
//   2k, mask 3 -> gives [half0_k | half1_k] directly since the 64-byte load is
//   [q3bits0 | q3bits1].
// q8: half0_k = q8 elems [k*32 .. k*32+31]; half1_k = q8 elems [128 + k*32 ..].
//   So q8_k_512 = concat(q8[ k*32 ], q8[ 128 + k*32 ]).
// hbits: half0 byte = hmask[lane] bit (k); half1 byte = hmask[lane] bit (4+k).
//   So hbits_512 = [hmask(256-broadcast) | hmask(256-broadcast)], and the
//   bit selector zmm = [ mone<<k (low256) | mone<<(4+k) (high256) ].
// scales: half0_k uses scales[0] shuffle(is0=k), half1_k uses scales[1]
//   shuffle(k). So scale_512 = [ scales[0] | scales[1] ] with the per-128-lane
//   shuffle get_scale_shuffle_q3k(k) applied in each half.
// ============================================================================
void ggml_vec_dot_q3_K_q8_K_avx512(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = (const block_q3_K *)vx;
    const block_q8_K * GGML_RESTRICT y = (const block_q8_K *)vy;
    const int nb = n / QK_K;

    const __m512i m3   = _mm512_set1_epi8(3);
    const __m512i mone = _mm512_set1_epi8(1);
    const __m128i m32  = _mm_set1_epi8(32);

    __m512 acc = _mm512_setzero_ps();
    uint32_t aux[3];

    // bit-selector zmm for k=0..3: low 256 = mone<<k, high 256 = mone<<(4+k)
    // built per-iteration from mone via slli.

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        // scale_512 = [ scales[0]=dup(l) | scales[1]=dup(h) ] across the two
        // 256-bit halves; within each 256-bit half the 128 is duplicated so a
        // within-128 shuffle works in all four 128-lanes.
        const __m256i scales0_256 = MM256_SET_M128I(l_scales, l_scales);
        const __m256i scales1_256 = MM256_SET_M128I(h_scales, h_scales);
        const __m512i scale_512 = _mm512_inserti64x4(_mm512_castsi256_si512(scales0_256), scales1_256, 1);

        // hbits broadcast: [hmask(256) | hmask(256)] -> 64-byte zmm
        const __m256i hbits256 = _mm256_loadu_si256((const __m256i*)x[i].hmask);
        const __m512i hbits = _mm512_inserti64x4(_mm512_castsi256_si512(hbits256), hbits256, 1);

        // whole 64-byte qs load = [q3bits0 | q3bits1]
        const __m512i q3all = _mm512_loadu_si512((const __m512i*)q3);

        __m512i sumi = _mm512_setzero_si512();

        for (int k = 0; k < 4; ++k) {
            // low 2 bits for this k: shift 2k, mask 3
            const __m512i q3l = _mm512_and_si512(_mm512_srli_epi16(q3all, 2*k), m3);

            // high-bit operand: low256 uses bit k, high256 uses bit 4+k.
            // selector = [ mone<<k | mone<<(4+k) ]  -> build via two slli + blend
            const __m512i sel_lo = _mm512_slli_epi16(mone, k);      // bit k in all lanes
            const __m512i sel_hi = _mm512_slli_epi16(mone, 4 + k);  // bit 4+k in all lanes
            // combine: low 256 = sel_lo, high 256 = sel_hi
            const __m512i sel = _mm512_inserti64x4(sel_lo, _mm512_extracti64x4_epi64(sel_hi, 1), 1);
            // q3h = ((~hbits & sel) >> bit) << 2 ; but bit differs per half.
            // (~hbits & sel) gives sel-or-0 per lane; >> bit normalizes to 0/1.
            // Since bit differs (k low, 4+k high) we shift each half separately.
            __m512i masked = _mm512_andnot_si512(hbits, sel);
            // normalize: low half >> k, high half >> (4+k), then <<2
            __m256i mlo = _mm512_castsi512_si256(masked);
            __m256i mhi = _mm512_extracti64x4_epi64(masked, 1);
            mlo = _mm256_srli_epi16(mlo, k);
            mhi = _mm256_srli_epi16(mhi, 4 + k);
            __m512i q3h = _mm512_inserti64x4(_mm512_castsi256_si512(mlo), mhi, 1);
            q3h = _mm512_slli_epi16(q3h, 2);

            // q8 operand: [ q8 elems(k*32..k*32+31) | q8 elems(128+k*32 ..) ]
            const __m256i q8lo = _mm256_loadu_si256((const __m256i*)(q8 +       k*32));
            const __m256i q8hi = _mm256_loadu_si256((const __m256i*)(q8 + 128 + k*32));
            const __m512i q8v  = _mm512_inserti64x4(_mm512_castsi256_si512(q8lo), q8hi, 1);

            __m512i q8s = _mm512_maddubs_epi16(q3h, q8v);
            __m512i p16 = _mm512_maddubs_epi16(q3l, q8v);
            p16 = _mm512_sub_epi16(p16, q8s);

            // scale shuffle: within-128-lane get_scale_shuffle_q3k(k) applied to
            // scale_512 (low256 has scales[0], high256 has scales[1]) -> picks
            // scale[k] in low256 and scale[4+k]?? NO. AVX2: half0 uses
            // get_scale_shuffle_q3k(is+ k) with is=0 -> scales[0] index k.
            // half1 (j=1) uses is=4 -> scales[1] index (4+k)?? get_scale_shuffle
            // index 4..7 maps into the 256-bit 'scales[1]' which holds
            // h_scales (sub 8..15). shuffle index (is + k) where is=4 -> 4+k,
            // and get_scale_shuffle_q3k returns a within-256 mask. Since
            // scale1_256 broadcasts h_scales in BOTH 128 lanes, the shuffle
            // index (4+k) picks element (4+k) of an 8-elem set... but h_scales
            // only has 8 int16 (indices 0..7 = sub 8..15). get_scale_shuffle_q3k
            // index 4+k addresses byte pairs (8+2k..) i.e. element 4+k of 16.
            // That's WRONG for a broadcast 128. We must replicate AVX2 exactly:
            // AVX2 scales[1] = dup(h_scales) is a 256 with h_scales in BOTH
            // halves; get_scale_shuffle_q3k(4+k) indexes byte 2*(4+k).. which is
            // within a 128 (16 bytes) only valid for index 0..7. For k up to 3,
            // 4+k up to 7 -> byte 8..15, valid within 128. Good: it picks
            // h_scales element (4+k) which = sub-block (8 + 4 + k) = sub 12+k for
            // the SECOND half... Let's just trust AVX2 layout: build the 512
            // shuffle by concatenating get_scale_shuffle_q3k(k) (low) and
            // get_scale_shuffle_q3k(4+k) (high).
            const __m256i shlo = get_scale_shuffle_q3k(k);
            const __m256i shhi = get_scale_shuffle_q3k(4 + k);
            const __m512i shuf = _mm512_inserti64x4(_mm512_castsi256_si512(shlo), shhi, 1);
            p16 = _mm512_madd_epi16(_mm512_shuffle_epi8(scale_512, shuf), p16);

            sumi = _mm512_add_epi32(sumi, p16);
        }

        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }

    *s = _mm512_reduce_add_ps(acc);
}

// ============================================================================
//  Test harness
// ============================================================================
static uint64_t g_lcg = 0x2545F4914F6CDD1DULL;
static inline uint32_t lcg() {
    g_lcg = g_lcg * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(g_lcg >> 33);
}
static inline int rnd_range(int lo, int hi) { return lo + (int)(lcg() % (uint32_t)(hi - lo + 1)); }

static double now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fill_q8(block_q8_K * b) {
    b->d = (float)(0.005 + (lcg() % 1000) * 1e-5);
    for (int j = 0; j < QK_K; ++j) b->qs[j] = (int8_t)(rnd_range(-127, 127));
    for (int g = 0; g < QK_K/16; ++g) {
        int s16 = 0;
        for (int t = 0; t < 16; ++t) s16 += b->qs[g*16 + t];
        b->bsums[g] = (int16_t)s16;
    }
}
static void fill_q2(block_q2_K * b) {
    b->d    = _cvtss_sh((float)(0.01 + (lcg()%500)*1e-5), 0);
    b->dmin = _cvtss_sh((float)(0.005 + (lcg()%500)*1e-5), 0);
    for (int j = 0; j < QK_K/16; ++j) b->scales[j] = (uint8_t)(lcg() & 0xFF); // 4b scale + 4b min
    for (int j = 0; j < QK_K/4;  ++j) b->qs[j]     = (uint8_t)(lcg() & 0xFF);
}
static void fill_q3(block_q3_K * b) {
    b->d = _cvtss_sh((float)(0.01 + (lcg()%500)*1e-5), 0);
    for (int j = 0; j < QK_K/8;  ++j) b->hmask[j]  = (uint8_t)(lcg() & 0xFF);
    for (int j = 0; j < QK_K/4;  ++j) b->qs[j]     = (uint8_t)(lcg() & 0xFF);
    for (int j = 0; j < 12;      ++j) b->scales[j] = (uint8_t)(lcg() & 0xFF);
}

struct Stats { double cosine; double max_rel; };
static Stats compare(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0, maxrel = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i]*b[i];
        na  += (double)a[i]*a[i];
        nb  += (double)b[i]*b[i];
        double den = std::max(1e-12, std::fabs((double)a[i]));
        double rel = std::fabs((double)a[i]-(double)b[i]) / den;
        if (rel > maxrel) maxrel = rel;
    }
    return { dot / (std::sqrt(na)*std::sqrt(nb) + 1e-30), maxrel };
}

template <typename FK, typename T>
static double time_kernel(FK kern, int n, const std::vector<T>& wrows, int rows,
                          int row_blocks, const std::vector<block_q8_K>& act,
                          std::vector<float>& out, int iters) {
    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
        double t0 = now_sec();
        for (int r = 0; r < rows; ++r) {
            kern(n, &out[r], 0, &wrows[(size_t)r*row_blocks], 0, act.data(), 0, 1);
        }
        double t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best * 1e3; // ms
}

template <typename T, typename FK>
static void run_case(const char* name, int n, int rows,
                     void(*fillw)(T*), FK kavx2, FK kavx512, bool l2fit=false) {
    const int row_blocks = n / QK_K;
    // l2fit mode: shrink rows so weights fit in L2 (1MB/core) -> isolate compute
    if (l2fit) {
        size_t blk_bytes = sizeof(T);
        size_t target = 700*1024; // stay under 1MB L2
        rows = (int)std::max((size_t)8, target / (blk_bytes * row_blocks));
    }
    std::vector<T> wrows((size_t)rows * row_blocks);
    std::vector<block_q8_K> act(row_blocks);

    for (auto& w : wrows) fillw(&w);
    for (auto& a : act) fill_q8(&a);

    std::vector<float> out2(rows), out5(rows);

    // correctness (single pass)
    for (int r = 0; r < rows; ++r) {
        kavx2 (n, &out2[r], 0, &wrows[(size_t)r*row_blocks], 0, act.data(), 0, 1);
        kavx512(n, &out5[r], 0, &wrows[(size_t)r*row_blocks], 0, act.data(), 0, 1);
    }
    Stats st = compare(out2, out5);

    // warm + time (best of iters); scale iters up for tiny L2-fit cases
    const int iters = l2fit ? 4000 : 60;
    double ms2 = time_kernel(kavx2,   n, wrows, rows, row_blocks, act, out2, iters);
    double ms5 = time_kernel(kavx512, n, wrows, rows, row_blocks, act, out5, iters);

    printf("=== %s  (n=%d, rows=%d, blocks/row=%d) ===\n", name, n, rows, row_blocks);
    printf("  AVX2   : %8.3f ms\n", ms2);
    printf("  AVX512 : %8.3f ms\n", ms5);
    printf("  speedup: %6.3fx\n", ms2 / ms5);
    printf("  cosine : %.10f\n", st.cosine);
    printf("  max_rel: %.3e\n", st.max_rel);
    printf("  sample out[0]: avx2=%.6f avx512=%.6f\n", out2[0], out5[0]);
    printf("  GATE   : %s\n\n", (st.cosine > 0.99999) ? "PASS" : "FAIL");
}

int main() {
    printf("Q2_K / Q3_K AVX2 vs AVX-512 microbench (single thread, taskset -c 0)\n\n");

    // Q2_K: gate/up shape n=6144 ; rows chosen so wall ~100ms+
    run_case<block_q2_K>("Q2_K n=6144 (gate/up)", 6144, 4096, fill_q2,
                         ggml_vec_dot_q2_K_q8_K_avx2, ggml_vec_dot_q2_K_q8_K_avx512);
    run_case<block_q2_K>("Q2_K n=2048 (down)",     2048, 8192, fill_q2,
                         ggml_vec_dot_q2_K_q8_K_avx2, ggml_vec_dot_q2_K_q8_K_avx512);

    // Q3_K: down shape n=2048 (primary), plus n=6144
    run_case<block_q3_K>("Q3_K n=2048 (down)",     2048, 8192, fill_q3,
                         ggml_vec_dot_q3_K_q8_K_avx2, ggml_vec_dot_q3_K_q8_K_avx512);
    run_case<block_q3_K>("Q3_K n=6144 (gate/up)",  6144, 4096, fill_q3,
                         ggml_vec_dot_q3_K_q8_K_avx2, ggml_vec_dot_q3_K_q8_K_avx512);

    printf("--- L2-resident (compute-ceiling) variants ---\n\n");
    run_case<block_q2_K>("Q2_K n=2048 L2-fit", 2048, 0, fill_q2,
                         ggml_vec_dot_q2_K_q8_K_avx2, ggml_vec_dot_q2_K_q8_K_avx512, true);
    run_case<block_q3_K>("Q3_K n=2048 L2-fit", 2048, 0, fill_q3,
                         ggml_vec_dot_q3_K_q8_K_avx2, ggml_vec_dot_q3_K_q8_K_avx512, true);

    return 0;
}
