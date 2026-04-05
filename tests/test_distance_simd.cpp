#include "fry/distance.hpp"
#include "fry/distance_simd.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

// ── Helper ────────────────────────────────────────────────────────────────────

static bool approx_eq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

// ── SIMD tests are gated on platform macros ───────────────────────────────────
// They compile and run only on the platform where the kernel exists.
// On other platforms the scalar path is tested via distance_simd.hpp instead.

// ═══════════════════════════════════════════════════════════════════════════════
// NEON kernels (ARM64 / Apple M-series)
// ═══════════════════════════════════════════════════════════════════════════════

#if defined(FRY_HAVE_NEON)

TEST_CASE("NEON l2_squared: matches scalar — dim=4 (exact multiple)", "[simd][neon]") {
    // dim=4 — one full NEON iteration, no tail.
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {4.0f, 3.0f, 2.0f, 1.0f};
    REQUIRE(approx_eq(fry::l2_squared_neon(a, b, 4),
                      fry::l2_squared(a, b, 4)));
}

TEST_CASE("NEON l2_squared: matches scalar — dim=5 (tail=1)", "[simd][neon]") {
    // dim=5 — one full NEON iteration + 1 scalar tail element.
    float a[] = {1.0f, 0.0f, 0.0f, 0.0f, 3.0f};
    float b[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    REQUIRE(approx_eq(fry::l2_squared_neon(a, b, 5),
                      fry::l2_squared(a, b, 5)));
}

TEST_CASE("NEON l2_squared: matches scalar — dim=3 (no full iteration, scalar only)", "[simd][neon]") {
    // dim=3 < 4 — the NEON loop does nothing, all work is in the tail.
    // Tests that the tail handles the entire computation correctly.
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    REQUIRE(approx_eq(fry::l2_squared_neon(a, b, 3),
                      fry::l2_squared(a, b, 3)));
}

TEST_CASE("NEON l2_squared: matches scalar — dim=128 (typical embedding)", "[simd][neon]") {
    // dim=128 — 32 full NEON iterations, no tail. Typical embedding size.
    std::vector<float> a(128), b(128);
    for (int i = 0; i < 128; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(128 - i);
    }
    REQUIRE(approx_eq(fry::l2_squared_neon(a.data(), b.data(), 128),
                      fry::l2_squared(a.data(), b.data(), 128)));
}

TEST_CASE("NEON l2_squared: identical vectors → 0", "[simd][neon]") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE(fry::l2_squared_neon(a, a, 4) == 0.0f);
}

TEST_CASE("NEON inner_product: matches scalar — dim=4", "[simd][neon]") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {4.0f, 3.0f, 2.0f, 1.0f};
    // 1*4 + 2*3 + 3*2 + 4*1 = 4+6+6+4 = 20
    REQUIRE(approx_eq(fry::inner_product_neon(a, b, 4),
                      fry::inner_product(a, b, 4)));
}

TEST_CASE("NEON inner_product: matches scalar — dim=7 (tail=3)", "[simd][neon]") {
    std::vector<float> a(7, 1.0f);
    std::vector<float> b(7, 2.0f);
    // All 1s dot all 2s = 7 * 2 = 14
    REQUIRE(approx_eq(fry::inner_product_neon(a.data(), b.data(), 7),
                      fry::inner_product(a.data(), b.data(), 7)));
}

TEST_CASE("NEON inner_product: matches scalar — dim=128", "[simd][neon]") {
    std::vector<float> a(128, 1.0f);
    std::vector<float> b(128, 1.0f);
    REQUIRE(approx_eq(fry::inner_product_neon(a.data(), b.data(), 128),
                      fry::inner_product(a.data(), b.data(), 128)));
}

#endif // FRY_HAVE_NEON

// ═══════════════════════════════════════════════════════════════════════════════
// AVX2 kernels (x86_64)
// ═══════════════════════════════════════════════════════════════════════════════

#if defined(FRY_HAVE_AVX2)

TEST_CASE("AVX2 l2_squared: matches scalar — dim=8 (exact multiple)", "[simd][avx2]") {
    float a[] = {1,2,3,4,5,6,7,8};
    float b[] = {8,7,6,5,4,3,2,1};
    REQUIRE(approx_eq(fry::l2_squared_avx2(a, b, 8),
                      fry::l2_squared(a, b, 8)));
}

TEST_CASE("AVX2 l2_squared: matches scalar — dim=9 (tail=1)", "[simd][avx2]") {
    float a[] = {1,2,3,4,5,6,7,8,9};
    float b[] = {0,0,0,0,0,0,0,0,0};
    REQUIRE(approx_eq(fry::l2_squared_avx2(a, b, 9),
                      fry::l2_squared(a, b, 9)));
}

TEST_CASE("AVX2 l2_squared: matches scalar — dim=128", "[simd][avx2]") {
    std::vector<float> a(128), b(128);
    for (int i = 0; i < 128; ++i) { a[i] = static_cast<float>(i); b[i] = 1.0f; }
    REQUIRE(approx_eq(fry::l2_squared_avx2(a.data(), b.data(), 128),
                      fry::l2_squared(a.data(), b.data(), 128)));
}

TEST_CASE("AVX2 inner_product: matches scalar — dim=8", "[simd][avx2]") {
    std::vector<float> a(8, 2.0f);
    std::vector<float> b(8, 3.0f);
    REQUIRE(approx_eq(fry::inner_product_avx2(a.data(), b.data(), 8),
                      fry::inner_product(a.data(), b.data(), 8)));
}

TEST_CASE("AVX2 inner_product: matches scalar — dim=128", "[simd][avx2]") {
    std::vector<float> a(128, 1.0f);
    std::vector<float> b(128, 1.0f);
    REQUIRE(approx_eq(fry::inner_product_avx2(a.data(), b.data(), 128),
                      fry::inner_product(a.data(), b.data(), 128)));
}

#endif // FRY_HAVE_AVX2

// ═══════════════════════════════════════════════════════════════════════════════
// Unified dispatcher — runs on all platforms
// Tests that distance_simd.hpp calls the right backend transparently.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("SIMD dispatcher: l2_squared_simd matches scalar", "[simd][dispatcher]") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    REQUIRE(approx_eq(fry::l2_squared_simd(a, b, 5),
                      fry::l2_squared(a, b, 5)));
}

TEST_CASE("SIMD dispatcher: inner_product_simd matches scalar", "[simd][dispatcher]") {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {4.0f, 3.0f, 2.0f, 1.0f};
    REQUIRE(approx_eq(fry::inner_product_simd(a, b, 4),
                      fry::inner_product(a, b, 4)));
}
