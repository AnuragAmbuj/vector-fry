#include "fry/flat_index.hpp"

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <vector>

// ── Helpers ───────────────────────────────────────────────────────────────────

// Returns true if `id` appears in `results`.
static bool contains(const std::vector<fry::VectorId>& results, fry::VectorId id) {
    return std::find(results.begin(), results.end(), id) != results.end();
}

// ═══════════════════════════════════════════════════════════════════════════════
// FlatIndex<L2> — insert
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("FlatIndex: rejects zero dimension", "[flat_index]") {
    REQUIRE_THROWS_AS((fry::FlatIndex<fry::Metric::L2>(0)), std::invalid_argument);
}

TEST_CASE("FlatIndex: insert returns sequential ids", "[flat_index]") {
    fry::FlatIndex<fry::Metric::L2> idx(3);
    REQUIRE(idx.size() == 0);

    auto r0 = idx.insert(fry::Vector<float>{1.0f, 0.0f, 0.0f});
    auto r1 = idx.insert(fry::Vector<float>{0.0f, 1.0f, 0.0f});

    REQUIRE(r0.has_value());
    REQUIRE(r1.has_value());
    REQUIRE(*r0 == 0);
    REQUIRE(*r1 == 1);
    REQUIRE(idx.size() == 2);
}

TEST_CASE("FlatIndex: insert rejects dimension mismatch", "[flat_index]") {
    fry::FlatIndex<fry::Metric::L2> idx(3);
    auto result = idx.insert(fry::Vector<float>{1.0f, 0.0f}); // dim=2, expects 3

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().code == fry::Error::Code::DimensionMismatch);
    REQUIRE(idx.size() == 0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FlatIndex<L2> — query top-K
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("FlatIndex: query on empty index returns error", "[flat_index]") {
    fry::FlatIndex<fry::Metric::L2> idx(2);
    auto result = idx.query(fry::Vector<float>{1.0f, 0.0f}, 1);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().code == fry::Error::Code::StoreEmpty);
}

TEST_CASE("FlatIndex: query k=0 returns error", "[flat_index]") {
    fry::FlatIndex<fry::Metric::L2> idx(2);
    idx.insert(fry::Vector<float>{1.0f, 0.0f});
    auto result = idx.query(fry::Vector<float>{1.0f, 0.0f}, 0);

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().code == fry::Error::Code::InvalidArgument);
}

TEST_CASE("FlatIndex: query rejects dimension mismatch", "[flat_index]") {
    fry::FlatIndex<fry::Metric::L2> idx(3);
    idx.insert(fry::Vector<float>{1.0f, 0.0f, 0.0f});
    auto result = idx.query(fry::Vector<float>{1.0f, 0.0f}, 1); // dim=2, expects 3

    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().code == fry::Error::Code::DimensionMismatch);
}

TEST_CASE("FlatIndex: k=1 returns nearest neighbour", "[flat_index]") {
    fry::FlatIndex<fry::Metric::L2> idx(2);
    const fry::VectorId id0 = idx.insert(fry::Vector<float>{1.0f, 0.0f}).value();
    const fry::VectorId id1 = idx.insert(fry::Vector<float>{0.0f, 1.0f}).value();

    auto r0 = idx.query(fry::Vector<float>{0.9f, 0.1f}, 1);
    REQUIRE(r0.has_value());
    REQUIRE(r0->size() == 1);
    REQUIRE((*r0)[0] == id0);

    auto r1 = idx.query(fry::Vector<float>{0.1f, 0.9f}, 1);
    REQUIRE(r1.has_value());
    REQUIRE(r1->size() == 1);
    REQUIRE((*r1)[0] == id1);
}

TEST_CASE("FlatIndex: k=2 returns two nearest in order", "[flat_index]") {
    // Three vectors along x-axis. Query near {1,0} — expect id0 first, id1 second.
    fry::FlatIndex<fry::Metric::L2> idx(2);
    const fry::VectorId id0 = idx.insert(fry::Vector<float>{1.0f, 0.0f}).value(); // closest
    const fry::VectorId id1 = idx.insert(fry::Vector<float>{0.5f, 0.0f}).value(); // second
    const fry::VectorId id2 = idx.insert(fry::Vector<float>{0.0f, 1.0f}).value(); // far

    auto result = idx.query(fry::Vector<float>{0.9f, 0.0f}, 2);
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 2);
    REQUIRE((*result)[0] == id0); // closest first
    REQUIRE((*result)[1] == id1); // second closest
    REQUIRE_FALSE(contains(*result, id2)); // far one excluded
}

TEST_CASE("FlatIndex: k > size returns all vectors", "[flat_index]") {
    // Asking for more neighbours than exist should return all, not error.
    fry::FlatIndex<fry::Metric::L2> idx(2);
    idx.insert(fry::Vector<float>{1.0f, 0.0f}).value();
    idx.insert(fry::Vector<float>{0.0f, 1.0f}).value();

    auto result = idx.query(fry::Vector<float>{0.5f, 0.5f}, 10);
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 2); // only 2 exist
}

TEST_CASE("FlatIndex: results are sorted closest-first", "[flat_index]") {
    fry::FlatIndex<fry::Metric::L2> idx(1);
    idx.insert(fry::Vector<float>{10.0f}).value(); // far
    idx.insert(fry::Vector<float>{1.0f}).value();  // closest
    idx.insert(fry::Vector<float>{5.0f}).value();  // middle

    auto result = idx.query(fry::Vector<float>{0.0f}, 3);
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 3);

    // Distances from 0: 1^2=1, 5^2=25, 10^2=100
    // Closest first → ids in insertion order: 1, 2, 0
    REQUIRE((*result)[0] == 1); // {1.0f} — closest
    REQUIRE((*result)[1] == 2); // {5.0f} — middle
    REQUIRE((*result)[2] == 0); // {10.0f} — farthest
}

// ═══════════════════════════════════════════════════════════════════════════════
// FlatIndex<Cosine> — metric selection
// ═══════════════════════════════════════════════════════════════════════════════

TEST_CASE("FlatIndex<Cosine>: nearest by angle not distance", "[flat_index][cosine]") {
    // {2,0} and {1,0} are in the same direction (cosine=1) but {2,0} is farther.
    // {0,1} is perpendicular (cosine=0).
    // Cosine index should rank {2,0} and {1,0} as equally close (both cosine=1),
    // and {0,1} as farthest.
    fry::FlatIndex<fry::Metric::Cosine> idx(2);
    const fry::VectorId id0 = idx.insert(fry::Vector<float>{2.0f, 0.0f}).value();
    const fry::VectorId id1 = idx.insert(fry::Vector<float>{1.0f, 0.0f}).value();
    idx.insert(fry::Vector<float>{0.0f, 1.0f}).value(); // perpendicular

    // Query along x-axis: both id0 and id1 should appear before the perpendicular one.
    auto result = idx.query(fry::Vector<float>{1.0f, 0.0f}, 2);
    REQUIRE(result.has_value());
    REQUIRE(result->size() == 2);
    REQUIRE(contains(*result, id0));
    REQUIRE(contains(*result, id1));
}
