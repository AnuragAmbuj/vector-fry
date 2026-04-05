#pragma once

// distance_avx2
//
// x86_64 AVX2 + FMA SIMD distance kernels for Intel and AMD CPUs.
//
// AVX2 (Advanced Vector Extensions 2) is the SIMD extension for x86_64.
// It provides 256-bit vector registers that hold 8 x float32 simultaneously.
// One AVX2 instruction does the work of 8 scalar instructions — 2× NEON's
// throughput per instruction on paper (though real gains depend on memory).
//
// FMA (Fused Multiply-Add) is a companion extension enabled with -mfma.
// It provides _mm256_fmadd_ps: acc += a * b in a single fused instruction,
// giving better accuracy and throughput than separate multiply + add.
//
// ── Key types and intrinsics ──────────────────────────────────────────────────
//
//   __m256                   — a 256-bit register holding 8 floats
//                              [lane0 | lane1 | lane2 | lane3 | lane4 | ... | lane7]
//
//   _mm256_setzero_ps()      — set all 8 lanes to 0.0f
//                              result: [0, 0, 0, 0, 0, 0, 0, 0]
//
//   _mm256_loadu_ps(ptr)     — load 8 consecutive floats from memory (unaligned)
//                              result: [ptr[0], ptr[1], ..., ptr[7]]
//                              'u' = unaligned: safe even if ptr % 32 != 0
//
//   _mm256_sub_ps(a, b)      — lane-wise subtraction
//                              result: [a0-b0, a1-b1, ..., a7-b7]
//
//   _mm256_fmadd_ps(a, b, c) — fused multiply-add: c[i] + a[i] * b[i]
//                              result: [c0+a0*b0, c1+a1*b1, ..., c7+a7*b7]
//                              FMA = one rounding error instead of two
//
//   _mm256_mul_ps(a, b)      — lane-wise multiplication (non-FMA alternative)
//                              result: [a0*b0, a1*b1, ..., a7*b7]
//
//   _mm256_add_ps(a, b)      — lane-wise addition
//                              result: [a0+b0, a1+b1, ..., a7+b7]
//
// ── Horizontal sum (reducing 8 lanes → 1 scalar) ─────────────────────────────
//
// AVX2 has no single "horizontal sum all 8 lanes" instruction unlike NEON's
// vaddvq_f32. We must reduce in multiple steps using the hsum256 helper below.
//
//   Step 1: _mm256_hadd_ps(v, v)
//           Horizontal add WITHIN each 128-bit half independently.
//           If v = [a0, a1, a2, a3, a4, a5, a6, a7], then:
//           output: [a0+a1, a2+a3, a0+a1, a2+a3 | a4+a5, a6+a7, a4+a5, a6+a7]
//           (each 128-bit half reduces independently — note the duplication)
//
//   Step 2: _mm256_hadd_ps(h1, h1)
//           Second hadd reduces each half to one value in lane [0] and [4]:
//           output: [a0+a1+a2+a3, *, *, * | a4+a5+a6+a7, *, *, *]
//
//   Step 3: _mm256_extractf128_ps(h2, 1)
//           Extract the HIGH 128-bit half as a __m128.
//           lane[0] of this __m128 = a4+a5+a6+a7
//
//   Step 4: _mm_add_ps(low128, high128)
//           Add low half (a0..a3 sum) + high half (a4..a7 sum).
//           lane[0] now holds the complete sum a0+a1+...+a7.
//
//   Step 5: _mm_cvtss_f32(result)
//           Extract lane[0] as a scalar float.
//
// ── Loop structure (both functions) ──────────────────────────────────────────
//
//   Main loop : processes 8 floats per iteration (i += 8)
//               runs while i + 8 <= dim  (a full 8-wide block must remain)
//
//   Scalar tail: handles the remaining dim % 8 elements
//               e.g. dim=9  → main loop once (i=0..7), tail handles i=8
//               e.g. dim=8  → main loop once, no tail
//               e.g. dim=5  → main loop skipped entirely, tail handles all
//
// ── AVX2 vs NEON comparison ───────────────────────────────────────────────────
//
//   Feature          NEON (ARM64)       AVX2 (x86_64)
//   ─────────────────────────────────────────────────
//   Register width   128-bit            256-bit
//   Floats/register  4                  8
//   Horizontal sum   vaddvq_f32 (1 op)  5-step hadd/extract/add sequence
//   FMA support      vfmaq_f32          _mm256_fmadd_ps (with -mfma)
//   Always present?  Yes on arm64       No — requires -mavx2 -mfma flags
//
// ── File guard ────────────────────────────────────────────────────────────────
// Only compiled when FRY_HAVE_AVX2 is defined by CMake (x86_64 platform).
// Including this on ARM would fail — <immintrin.h> does not exist there.

#ifdef FRY_HAVE_AVX2

#include <immintrin.h>
#include <cstddef>

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
// Rationale: SIMD kernels intentionally take two raw const float* pointers.
// They sit below the Vector<float> abstraction and are never called directly
// by user code — only through distance_simd.hpp. The pointer names a/b
// are unambiguous in this context.

namespace fry {
    // hsum256(__m256)
    //
    // Reduces all 8 lanes of a __m256 accumulator to a single float scalar.
    // See the "Horizontal sum" section at the top for full step-by-step breakdown.
    //
    // Marked inline so the compiler can eliminate the function call overhead and
    // emit the reduction instructions directly inside the calling loop.
    //
    inline auto hsum256(__m256 v) -> float {
        __m256 h1 = _mm256_hadd_ps(v, v);
        __m256 h2 = _mm256_hadd_ps(h1, h1);
        __m128 hi = _mm256_extractf128_ps(h2, 1);
        __m128 lo = _mm256_castps256_ps128(h2);
        __m128 sum = _mm_add_ps(lo, hi);
        return _mm_cvtss_f32(sum);
    }

    // l2_squared_avx2()
    //
    // Computes squared L2 distance between two float arrays using AVX2 + FMA.
    //
    //   result = Σ (a[i] - b[i])²   for i in [0, dim)
    //
    // No sqrt — for nearest-neighbour comparison, squared distance preserves
    // the same ordering as true distance and is significantly cheaper.
    //
    // Processes 8 floats per AVX2 iteration, scalar tail for dim % 8 remainder.
    //
    inline auto l2_squared_avx2(const float *a, const float *b, std::size_t dim) -> float {
        // Accumulator: 8 partial sums running in parallel across 8 lanes.
        // hsum256 collapses them into one scalar after the main loop.
        __m256 acc = _mm256_setzero_ps();
        std::size_t i = 0;

        // Main AVX2 loop
        for (; i + 8 <= dim; i += 8) {
            // load 8 floats (2x compared to neon) from a[i,i+7] and b[i,i+7]
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            // accumulate diff² using FMA: acc[k] += diff[k] * diff[k]
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }


        // Horizontal reduction - collapsing the 8 accumulator lanes into a single float.
        float result = hsum256(acc);

        // Handle elements that didn't fit into an 8-wide block (0–7 elements).
        for (; i < dim; ++i) {
            const float d = a[i] - b[i];
            result += d * d;
        }
        return result;
    }

    // inner_product_avx2
    //
    // Computes the inner product (dot product) of two float arrays using AVX2 + FMA.
    //
    //   result = Σ a[i] * b[i]   for i in [0, dim)
    //
    // Used in MIPS (Maximum Inner Product Search) — recommendation systems
    // find the item vector with the highest dot product with the user vector.
    // Unlike L2, HIGHER inner product = MORE similar (no negation needed).
    //
    // Processes 8 floats per AVX2 iteration, scalar tail for dim % 8 remainder.
    //
    inline auto inner_product_avx2(const float *a, const float *b, std::size_t dim) -> float {
        __m256 acc = _mm256_setzero_ps();

        std::size_t i = 0;

        // Main AVX2 loop
        for (; i + 8 <= dim; i += 8) {
            // load 8 floats from a[i..i+7] and b[i..i+7]
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);

            // accumulate a[k]*b[k] using FMA: acc[k] += a[k] * b[k]
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        // Horizontal reduction
        float result = hsum256(acc);

        // Scalar tail
        for (; i < dim; ++i) {
            result += a[i] * b[i];
        }

        return result;
    }
} // namespace fry

// NOLINTEND(bugprone-easily-swappable-parameters)

#endif // FRY_HAVE_AVX2
