#include "fry/distance.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>

// ── Helper ────────────────────────────────────────────────────────────────────

static bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

// ═══════════════════════════════════════════════════════════════════════════════
// l2_squared
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("l2_squared: identical vectors → 0", "[distance][l2]") {
    float a[] = {1.0f, 2.0f, 3.0f};
    REQUIRE(fry::l2_squared(a, a, 3) == 0.0f);
}

TEST_CASE("l2_squared: dim=1 positive values", "[distance][l2]") {
    float a[] = {3.0f};
    float b[] = {0.0f};
    REQUIRE(approx_eq(fry::l2_squared(a, b, 1), 9.0f));
}

TEST_CASE("l2_squared: dim=1 negative values", "[distance][l2]") {
    float a[] = {-1.0f};
    float b[] = { 1.0f};
    REQUIRE(approx_eq(fry::l2_squared(a, b, 1), 4.0f));
}

TEST_CASE("l2_squared: dim=3 known result", "[distance][l2]") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 6.0f, 3.0f};
    // (1-4)^2 + (2-6)^2 + (3-3)^2 = 9 + 16 + 0 = 25
    REQUIRE(approx_eq(fry::l2_squared(a, b, 3), 25.0f));
}

TEST_CASE("l2_squared: symmetry — d(a,b) == d(b,a)", "[distance][l2]") {
    float a[] = {1.0f, 0.0f, 0.0f};
    float b[] = {0.0f, 1.0f, 0.0f};
    REQUIRE(approx_eq(fry::l2_squared(a, b, 3),
                      fry::l2_squared(b, a, 3)));
}

TEST_CASE("l2_squared: all-negative components", "[distance][l2]") {
    float a[] = {-3.0f, -4.0f};
    float b[] = { 0.0f,  0.0f};
    REQUIRE(approx_eq(fry::l2_squared(a, b, 2), 25.0f));
}

TEST_CASE("l2_squared: both vectors negative", "[distance][l2]") {
    float a[] = {-1.0f};
    float b[] = {-3.0f};
    REQUIRE(approx_eq(fry::l2_squared(a, b, 1), 4.0f));
}

TEST_CASE("l2_squared: dim=128 unit vectors", "[distance][l2]") {
    float a[128] = {};
    float b[128];
    for (int i = 0; i < 128; ++i) b[i] = 1.0f;
    REQUIRE(approx_eq(fry::l2_squared(a, b, 128), 128.0f));
}

TEST_CASE("l2_squared: ordering — closer vector has smaller distance", "[distance][l2]") {
    float q[]     = {0.9f, 0.1f};
    float near_[] = {1.0f, 0.0f};
    float far_[]  = {0.0f, 1.0f};
    REQUIRE(fry::l2_squared(q, near_, 2) < fry::l2_squared(q, far_, 2));
}

// ═══════════════════════════════════════════════════════════════════════════════
// cosine_similarity
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("cosine_similarity: identical unit vectors → 1.0", "[distance][cosine]") {
    // Same direction, unit length → maximum similarity.
    float a[] = {1.0f, 0.0f, 0.0f};
    REQUIRE(approx_eq(fry::cosine_similarity(a, a, 3), 1.0f));
}

TEST_CASE("cosine_similarity: opposite vectors → -1.0", "[distance][cosine]") {
    // Exactly opposite direction → minimum similarity.
    float a[] = { 1.0f, 0.0f};
    float b[] = {-1.0f, 0.0f};
    REQUIRE(approx_eq(fry::cosine_similarity(a, b, 2), -1.0f));
}

TEST_CASE("cosine_similarity: perpendicular vectors → 0.0", "[distance][cosine]") {
    // 90 degrees apart → zero similarity.
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    REQUIRE(approx_eq(fry::cosine_similarity(a, b, 2), 0.0f));
}

TEST_CASE("cosine_similarity: same direction different magnitude → 1.0", "[distance][cosine]") {
    // Cosine only measures angle, not length.
    // {1,0} and {5,0} point the same way → similarity = 1.
    float a[] = {1.0f, 0.0f};
    float b[] = {5.0f, 0.0f};
    REQUIRE(approx_eq(fry::cosine_similarity(a, b, 2), 1.0f));
}

TEST_CASE("cosine_similarity: symmetry — sim(a,b) == sim(b,a)", "[distance][cosine]") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    REQUIRE(approx_eq(fry::cosine_similarity(a, b, 3),
                      fry::cosine_similarity(b, a, 3)));
}

TEST_CASE("cosine_similarity: zero vector → 0.0", "[distance][cosine]") {
    // Dividing by zero magnitude is undefined — must return 0.0 safely.
    float a[] = {0.0f, 0.0f, 0.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    REQUIRE(fry::cosine_similarity(a, b, 3) == 0.0f);
}

TEST_CASE("cosine_similarity: result is in [-1, 1]", "[distance][cosine]") {
    // Result must never exceed these bounds regardless of input.
    float a[] = {3.0f, 4.0f};
    float b[] = {-1.0f, 2.0f};
    float s = fry::cosine_similarity(a, b, 2);
    REQUIRE(s >= -1.0f);
    REQUIRE(s <=  1.0f);
}

TEST_CASE("cosine_similarity: known 45-degree angle", "[distance][cosine]") {
    // {1,0} and {1,1}/sqrt(2) are 45 degrees apart.
    // cosine(45°) = sqrt(2)/2 ≈ 0.7071
    float a[] = {1.0f, 0.0f};
    float b[] = {1.0f, 1.0f};
    REQUIRE(approx_eq(fry::cosine_similarity(a, b, 2), 0.70710678f, 1e-4f));
}

// ═══════════════════════════════════════════════════════════════════════════════
// inner_product
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("inner_product: orthogonal vectors → 0", "[distance][inner_product]") {
    // Perpendicular vectors have zero dot product.
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    REQUIRE(fry::inner_product(a, b, 2) == 0.0f);
}

TEST_CASE("inner_product: known result dim=3", "[distance][inner_product]") {
    // (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    REQUIRE(approx_eq(fry::inner_product(a, b, 3), 32.0f));
}

TEST_CASE("inner_product: symmetry — ip(a,b) == ip(b,a)", "[distance][inner_product]") {
    float a[] = {3.0f, -1.0f, 2.0f};
    float b[] = {1.0f,  4.0f, 0.0f};
    REQUIRE(approx_eq(fry::inner_product(a, b, 3),
                      fry::inner_product(b, a, 3)));
}

TEST_CASE("inner_product: negative components", "[distance][inner_product]") {
    // (-1*2) + (-2*-3) = -2 + 6 = 4
    float a[] = {-1.0f, -2.0f};
    float b[] = { 2.0f, -3.0f};
    REQUIRE(approx_eq(fry::inner_product(a, b, 2), 4.0f));
}

TEST_CASE("inner_product: zero vector → 0", "[distance][inner_product]") {
    float a[] = {0.0f, 0.0f, 0.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    REQUIRE(fry::inner_product(a, b, 3) == 0.0f);
}

TEST_CASE("inner_product: parallel vectors equal product of magnitudes", "[distance][inner_product]") {
    // For unit vectors in the same direction: ip(a,a) = |a|^2 = 1
    float a[] = {1.0f, 0.0f};
    REQUIRE(approx_eq(fry::inner_product(a, a, 2), 1.0f));
}

TEST_CASE("inner_product: higher score for more-aligned vector", "[distance][inner_product]") {
    // In MIPS, we want the vector with the HIGHEST inner product.
    // query = {1,0}: aligned has ip=1, misaligned has ip=0.5
    float q[]          = {1.0f, 0.0f};
    float aligned[]    = {1.0f, 0.0f};
    float misaligned[] = {0.5f, 0.5f};
    REQUIRE(fry::inner_product(q, aligned, 2) >
            fry::inner_product(q, misaligned, 2));
}

// ═══════════════════════════════════════════════════════════════════════════════
// dispatch<Metric>
//
// Template dispatch selects the correct distance function at compile time.
// Each test calls dispatch<M>(...) and checks it matches the direct function.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("dispatch<L2>: matches l2_squared", "[distance][dispatch]") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 6.0f, 3.0f};
    REQUIRE(fry::dispatch<fry::Metric::L2>(a, b, 3) ==
            fry::l2_squared(a, b, 3));
}

TEST_CASE("dispatch<Cosine>: matches cosine_similarity", "[distance][dispatch]") {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    REQUIRE(fry::dispatch<fry::Metric::Cosine>(a, b, 2) ==
            fry::cosine_similarity(a, b, 2));
}

TEST_CASE("dispatch<InnerProduct>: matches inner_product", "[distance][dispatch]") {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    REQUIRE(fry::dispatch<fry::Metric::InnerProduct>(a, b, 3) ==
            fry::inner_product(a, b, 3));
}

TEST_CASE("dispatch<L2>: identical vectors → 0", "[distance][dispatch]") {
    // Sanity check: dispatch preserves the underlying function's behaviour.
    float a[] = {3.0f, 1.0f, 4.0f};
    REQUIRE(fry::dispatch<fry::Metric::L2>(a, a, 3) == 0.0f);
}

TEST_CASE("dispatch<Cosine>: perpendicular → 0", "[distance][dispatch]") {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    REQUIRE(fry::dispatch<fry::Metric::Cosine>(a, b, 2) == 0.0f);
}

TEST_CASE("dispatch<InnerProduct>: orthogonal → 0", "[distance][dispatch]") {
    float a[] = {1.0f, 0.0f};
    float b[] = {0.0f, 1.0f};
    REQUIRE(fry::dispatch<fry::Metric::InnerProduct>(a, b, 2) == 0.0f);
}
