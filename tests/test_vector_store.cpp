#include "fry/result.hpp"
#include "fry/types.hpp"
#include "fry/vector_store.h"

#include <catch2/catch_test_macros.hpp>

// ═══════════════════════════════════════════════════════════════════════════════
// Vector<T> tests
//
// We test the type itself, independently of VectorStore.
// This "test the unit in isolation" discipline makes failures easier to diagnose.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Vector: construction from initialiser list", "[types]") {
    fry::Vector<float> v{1.0f, 2.0f, 3.0f};

    REQUIRE(v.dim() == 3);
    REQUIRE(v[0] == 1.0f);
    REQUIRE(v[1] == 2.0f);
    REQUIRE(v[2] == 3.0f);
}

TEST_CASE("Vector: construction from std::vector", "[types]") {
    std::vector<float> raw = {4.0f, 5.0f};
    fry::Vector<float> v{std::move(raw)};

    REQUIRE(v.dim() == 2);
    REQUIRE(v.data() != nullptr);
    // v.data() + v.dim() defines the half-open range of elements.
    // In C++20 we would use std::span here; in C++11 we use the two-part
    // "pointer + size" contract that data() and dim() together provide.
    REQUIRE(v.data()[0] == 4.0f);
    REQUIRE(v.data()[1] == 5.0f);
}

TEST_CASE("Vector: zero-dimension throws", "[types]") {
    // An empty initialiser list produces a zero-dimension vector — invalid.
    REQUIRE_THROWS_AS((fry::Vector<float>{}), std::invalid_argument);
}

TEST_CASE("Vector: equality comparison", "[types]") {
    fry::Vector<float> a{1.0f, 0.0f};
    fry::Vector<float> b{1.0f, 0.0f};
    fry::Vector<float> c{0.0f, 1.0f};

    REQUIRE(a == b);
    REQUIRE_FALSE(a == c);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result<T> / std::expected tests
//
// These test the error-handling machinery, not the business logic.
// If Result is wrong, every other test that uses it is also suspect.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("Result: make_ok holds value", "[result]") {
    fry::Result<int> r = fry::make_ok(42);

    REQUIRE(r.has_value());
    REQUIRE(*r == 42);
}

TEST_CASE("Result: make_error holds error", "[result]") {
    fry::Result<int> r = fry::make_error<int>(fry::Error::store_empty());

    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().code == fry::Error::Code::StoreEmpty);
    REQUIRE_FALSE(r.error().message.empty());
}

TEST_CASE("Result: dimension_mismatch error contains sizes", "[result]") {
    fry::Result<fry::VectorId> r =
        fry::make_error<fry::VectorId>(fry::Error::dimension_mismatch(3, 128));

    REQUIRE(r.error().code == fry::Error::Code::DimensionMismatch);
    // The message should mention both the got and expected dimensions.
    // We don't test the exact string — just that both numbers appear.
    REQUIRE(r.error().message.find("3")   != std::string::npos);
    REQUIRE(r.error().message.find("128") != std::string::npos);
}

// ═══════════════════════════════════════════════════════════════════════════════
// VectorStore integration tests
//
// These test VectorStore with the new Result-returning API.
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("VectorStore: rejects zero dimension", "[vector_store]") {
    REQUIRE_THROWS_AS(fry::VectorStore(0), std::invalid_argument);
}

TEST_CASE("VectorStore: insert succeeds and returns sequential ids", "[vector_store]") {
    fry::VectorStore store(2);
    REQUIRE(store.size() == 0);

    auto r0 = store.insert(fry::Vector<float>{1.0f, 0.0f});
    auto r1 = store.insert(fry::Vector<float>{0.0f, 1.0f});

    REQUIRE(r0.has_value());
    REQUIRE(r1.has_value());
    REQUIRE(*r0 == 0);
    REQUIRE(*r1 == 1);
    REQUIRE(store.size() == 2);
}

TEST_CASE("VectorStore: insert rejects mismatched dimension", "[vector_store]") {
    fry::VectorStore store(3);
    auto result = store.insert(fry::Vector<float>{1.0f, 0.0f}); // dim 2, store expects 3

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().code == fry::Error::Code::DimensionMismatch);
    // Store must be unchanged after a failed insert.
    REQUIRE(store.size() == 0);
}

TEST_CASE("VectorStore: nearest-neighbour search", "[vector_store]") {
    fry::VectorStore store(2);
    const fry::VectorId id0 = store.insert(fry::Vector<float>{1.0f, 0.0f}).value();
    const fry::VectorId id1 = store.insert(fry::Vector<float>{0.0f, 1.0f}).value();

    // Query close to (1,0) → should return id0
    auto r0 = store.search(fry::Vector<float>{0.9f, 0.1f});
    REQUIRE(r0.has_value());
    REQUIRE(*r0 == id0);

    // Query close to (0,1) → should return id1
    auto r1 = store.search(fry::Vector<float>{0.1f, 0.9f});
    REQUIRE(r1.has_value());
    REQUIRE(*r1 == id1);
}

TEST_CASE("VectorStore: search on empty store returns error", "[vector_store]") {
    fry::VectorStore store(2);
    auto result = store.search(fry::Vector<float>{1.0f, 0.0f});

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().code == fry::Error::Code::StoreEmpty);
}

TEST_CASE("VectorStore: search rejects mismatched query dimension", "[vector_store]") {
    fry::VectorStore store(3);
    store.insert(fry::Vector<float>{1.0f, 0.0f, 0.0f}).value();

    auto result = store.search(fry::Vector<float>{1.0f, 0.0f}); // dim 2, store expects 3

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().code == fry::Error::Code::DimensionMismatch);
}

// ─── Phase 1 smoke test ───────────────────────────────────────────────────────
// The minimal end-to-end check from the plan: insert one vector, retrieve it.
TEST_CASE("Smoke test: insert one vector, query it back", "[smoke]") {
    fry::VectorStore store(3);

    const fry::Vector<float> v{1.0f, 0.0f, 0.0f};
    const fry::VectorId id = store.insert(v).value();

    // Querying with the exact same vector must return the same id.
    const auto result = store.search(v);
    REQUIRE(result.has_value());
    REQUIRE(*result == id);
}
