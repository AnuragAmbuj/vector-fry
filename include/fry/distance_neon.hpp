#pragma once

// ── distance_neon.hpp ─────────────────────────────────────────────────────────
//
// ARM NEON SIMD distance kernels for Apple M-series and other arm64 CPUs.
//
// NEON is the SIMD (Single Instruction, Multiple Data) extension for ARM.
// It provides 128-bit vector registers that hold 4 x float32 simultaneously.
// One NEON instruction does the work of 4 scalar instructions.
//
// ── Key types and intrinsics ──────────────────────────────────────────────────
//
//   float32x4_t              — a 128-bit register holding 4 floats
//                              [lane0 | lane1 | lane2 | lane3]
//
//   vdupq_n_f32(x)           — broadcast scalar x into all 4 lanes
//                              result: [x, x, x, x]
//
//   vld1q_f32(ptr)           — load 4 consecutive floats from memory
//                              result: [ptr[0], ptr[1], ptr[2], ptr[3]]
//
//   vsubq_f32(a, b)          — lane-wise subtraction
//                              result: [a0-b0, a1-b1, a2-b2, a3-b3]
//
//   vfmaq_f32(acc, a, b)     — fused multiply-add: acc[i] += a[i] * b[i]
//                              "fused" means one rounding error instead of two
//                              result: [acc0+a0*b0, acc1+a1*b1, ...]
//
//   vaddvq_f32(v)            — horizontal sum: v[0]+v[1]+v[2]+v[3] → scalar
//
// ── Loop structure (both functions) ──────────────────────────────────────────
//
//   Main loop : processes 4 floats per iteration (i += 4)
//               runs while i + 4 <= dim  (i.e. a full 4-wide block remains)
//
//   Scalar tail: handles the remaining dim % 4 elements
//               e.g. dim=7 → main loop runs once (i=0..3), tail handles i=4,5,6
//               e.g. dim=4 → main loop runs once, no tail
//               e.g. dim=3 → main loop skipped entirely, tail handles all
//
// ── File guard ────────────────────────────────────────────────────────────────
// Only compiled when FRY_HAVE_NEON is defined by CMake (arm64 platform).
// Including this on x86 would fail — <arm_neon.h> does not exist there.

#ifdef FRY_HAVE_NEON

#include <arm_neon.h>
#include <cstddef>

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
// Rationale: SIMD kernels intentionally take two raw const float* pointers.
// They sit below the Vector<float> abstraction and are never called directly
// by user code — only through distance_simd.hpp. The pointer names a/b
// are unambiguous in this context.

namespace fry {

// ── l2_squared_neon ───────────────────────────────────────────────────────────
//
// Computes squared L2 distance between two float arrays using NEON.
//
//   result = Σ (a[i] - b[i])²   for i in [0, dim)
//
// No sqrt — for nearest-neighbour comparison, squared distance preserves
// the same ordering as true distance and is significantly cheaper.
//
// Processes 4 floats per NEON iteration, scalar tail for dim % 4 remainder.
//
inline auto l2_squared_neon(const float* a, const float* b, std::size_t dim) -> float {
    // Accumulator: 4 partial sums running in parallel across 4 lanes.
    // At the end, vaddvq_f32 collapses them into one scalar.
    float32x4_t acc = vdupq_n_f32(0.0f);

    std::size_t i = 0;

    // ── Main NEON loop ────────────────────────────────────────────────────────
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va   = vld1q_f32(a + i);        // load a[i..i+3]
        const float32x4_t vb   = vld1q_f32(b + i);        // load b[i..i+3]
        const float32x4_t diff = vsubq_f32(va, vb);       // diff[k] = a[k]-b[k]
        acc = vfmaq_f32(acc, diff, diff);                  // acc[k] += diff[k]²
    }

    // ── Horizontal reduction ──────────────────────────────────────────────────
    // Collapse the 4 accumulator lanes into a single float.
    float result = vaddvq_f32(acc);

    // ── Scalar tail ───────────────────────────────────────────────────────────
    // Handle elements that didn't fit into a 4-wide block (0–3 elements).
    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        result += d * d;
    }

    return result;
}

// ── inner_product_neon ────────────────────────────────────────────────────────
//
// Computes the inner product (dot product) of two float arrays using NEON.
//
//   result = Σ a[i] * b[i]   for i in [0, dim)
//
// Used in MIPS (Maximum Inner Product Search) — recommendation systems
// find the item vector with the highest dot product with the user vector.
// Unlike L2, HIGHER inner product = MORE similar (no negation needed).
//
// Processes 4 floats per NEON iteration, scalar tail for dim % 4 remainder.
//
inline auto inner_product_neon(const float* a, const float* b, std::size_t dim) -> float {
    float32x4_t acc = vdupq_n_f32(0.0f);

    std::size_t i = 0;

    // ── Main NEON loop ────────────────────────────────────────────────────────
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);          // load a[i..i+3]
        const float32x4_t vb = vld1q_f32(b + i);          // load b[i..i+3]
        acc = vfmaq_f32(acc, va, vb);                      // acc[k] += a[k]*b[k]
        // Note: no subtraction — inner product multiplies directly,
        // unlike L2 which needs the difference first.
    }

    // ── Horizontal reduction ──────────────────────────────────────────────────
    float result = vaddvq_f32(acc);

    // ── Scalar tail ───────────────────────────────────────────────────────────
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

} // namespace fry

// NOLINTEND(bugprone-easily-swappable-parameters)

#endif // FRY_HAVE_NEON
