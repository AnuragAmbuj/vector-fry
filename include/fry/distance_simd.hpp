#pragma once

// ── distance_simd.hpp ─────────────────────────────────────────────────────────
//
// Unified SIMD dispatcher — selects the right backend at compile time.
//
// This is the ONLY header user code (and distance.hpp dispatch<M>) should
// include for accelerated distance computation. It hides the platform
// details behind two simple functions:
//
//   fry::l2_squared_simd(a, b, dim)      — squared L2, best available kernel
//   fry::inner_product_simd(a, b, dim)   — dot product, best available kernel
//
// ── How the dispatch works ────────────────────────────────────────────────────
//
// CMake defines exactly one of the following macros at compile time based on
// the detected CPU architecture (see CMakeLists.txt, fry_simd target):
//
//   FRY_HAVE_NEON   — ARM64 (Apple M-series, Raspberry Pi 4, etc.)
//   FRY_HAVE_AVX2   — x86_64 with AVX2 + FMA support (Intel Haswell+, AMD Ryzen)
//   (neither)       — Unknown/unsupported architecture → scalar fallback
//
// The dispatcher uses #if / #elif / #else at compile time, so there is zero
// runtime overhead — no function pointer, no branch, no vtable lookup.
// The correct kernel is baked in at compile time for that binary.
//
// ── Fallback behaviour ────────────────────────────────────────────────────────
//
// When neither NEON nor AVX2 is available, the dispatcher falls back to the
// scalar implementations in distance.hpp. This guarantees:
//   1. Correctness on any platform (CI, WASM, exotic embedded targets)
//   2. A single test suite (test_distance_simd.cpp) tests all three paths
//      by including this header — each platform compiles only what it has.
//
// ── Dependency graph ─────────────────────────────────────────────────────────
//
//   distance_simd.hpp          ← this file (the public API)
//       ├── distance_neon.hpp  (only on FRY_HAVE_NEON)
//       ├── distance_avx2.hpp  (only on FRY_HAVE_AVX2)
//       └── distance.hpp       (always — for scalar fallback)
//
// ── Usage ─────────────────────────────────────────────────────────────────────
//
//   #include "fry/distance_simd.hpp"
//
//   float dist = fry::l2_squared_simd(a.data(), b.data(), dim);
//   float dot  = fry::inner_product_simd(a.data(), b.data(), dim);
//
// These are drop-in replacements for fry::l2_squared / fry::inner_product.
// The interface (argument types, return type) is identical — only the
// implementation changes based on compile-time platform detection.

#include "distance.hpp"          // scalar fallback: l2_squared, inner_product

#if defined(FRY_HAVE_NEON)
#   include "distance_neon.hpp"  // ARM NEON kernels
#elif defined(FRY_HAVE_AVX2)
#   include "distance_avx2.hpp"  // x86 AVX2 + FMA kernels
#endif

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
// Rationale: Dispatcher functions take two raw const float* pointers,
// matching the signature of the underlying NEON/AVX2/scalar kernels.
// These are never called directly by user code — they sit one layer below
// the Vector<float> abstraction. Names a/b are unambiguous here.

#include <cstddef>

namespace fry {

// ── l2_squared_simd ───────────────────────────────────────────────────────────
//
// Dispatches to the best available squared L2 kernel for this platform:
//
//   FRY_HAVE_NEON  → l2_squared_neon   (4 floats/iteration, 128-bit NEON)
//   FRY_HAVE_AVX2  → l2_squared_avx2   (8 floats/iteration, 256-bit AVX2)
//   (fallback)     → l2_squared        (1 float/iteration, portable scalar)
//
// All three produce bit-for-bit identical results for typical inputs
// (differences arise only from floating-point reassociation, within 1e-5).
//
inline auto l2_squared_simd(const float* a, const float* b, std::size_t dim) -> float {
#if defined(FRY_HAVE_NEON)
    return l2_squared_neon(a, b, dim);
#elif defined(FRY_HAVE_AVX2)
    return l2_squared_avx2(a, b, dim);
#else
    // Scalar fallback — correct on all platforms, no SIMD required.
    return l2_squared(a, b, dim);
#endif
}

// ── inner_product_simd ────────────────────────────────────────────────────────
//
// Dispatches to the best available dot-product kernel for this platform:
//
//   FRY_HAVE_NEON  → inner_product_neon   (4 floats/iteration, 128-bit NEON)
//   FRY_HAVE_AVX2  → inner_product_avx2   (8 floats/iteration, 256-bit AVX2)
//   (fallback)     → inner_product        (1 float/iteration, portable scalar)
//
inline auto inner_product_simd(const float* a, const float* b, std::size_t dim) -> float {
#if defined(FRY_HAVE_NEON)
    return inner_product_neon(a, b, dim);
#elif defined(FRY_HAVE_AVX2)
    return inner_product_avx2(a, b, dim);
#else
    // Scalar fallback — correct on all platforms, no SIMD required.
    return inner_product(a, b, dim);
#endif
}

} // namespace fry

// NOLINTEND(bugprone-easily-swappable-parameters)
