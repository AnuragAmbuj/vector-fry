#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "fry/hnsw_index.hpp"
#include "fry/flat_index.hpp"   // ground-truth oracle for recall tests
#include "fry/types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>

using namespace fry;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Build a random unit vector of the given dimension using a seeded RNG.
// Using a fixed seed makes tests deterministic across runs.
static Vector<float> random_vector(std::size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(dim);
    for (auto& x : data) x = dist(rng);
    return Vector<float>(std::move(data));
}

// Fraction of HNSW results that appear in the flat ground-truth top-K.
// Recall@K = |HNSW_top_K ∩ Flat_top_K| / K
static float recall_at_k(const std::vector<VectorId>& hnsw_result,
                          const std::vector<VectorId>& flat_result) {
    if (flat_result.empty()) return 1.0f;
    std::size_t hits = 0;
    for (VectorId id : hnsw_result) {
        if (std::find(flat_result.begin(), flat_result.end(), id) != flat_result.end())
            ++hits;
    }
    return static_cast<float>(hits) / static_cast<float>(flat_result.size());
}

// ── Construction ──────────────────────────────────────────────────────────────

TEST_CASE("HNSW: construction with default parameters", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(3);
    REQUIRE(idx.size() == 0);
    REQUIRE(idx.dim()  == 3);
}

TEST_CASE("HNSW: construction with explicit M and ef_construction", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(128, /*M=*/16, /*ef_construction=*/200);
    REQUIRE(idx.size() == 0);
    REQUIRE(idx.dim()  == 128);
}

// ── Insert error cases ─────────────────────────────────────────────────────────

TEST_CASE("HNSW: insert dimension mismatch returns error", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(3);
    Vector<float> wrong_dim{1.0f, 2.0f};   // dim=2, index expects dim=3
    auto r = idx.insert(wrong_dim);
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().code == Error::Code::DimensionMismatch);
}

// ── Query error cases ──────────────────────────────────────────────────────────

TEST_CASE("HNSW: query on empty index returns StoreEmpty", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(3);
    auto r = idx.query(Vector<float>{1.0f, 0.0f, 0.0f}, 1);
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().code == Error::Code::StoreEmpty);
}

TEST_CASE("HNSW: query with k=0 returns InvalidArgument", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(3);
    idx.insert(Vector<float>{1.0f, 0.0f, 0.0f});
    auto r = idx.query(Vector<float>{1.0f, 0.0f, 0.0f}, 0);
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().code == Error::Code::InvalidArgument);
}

TEST_CASE("HNSW: query dimension mismatch returns error", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(3);
    idx.insert(Vector<float>{1.0f, 0.0f, 0.0f});
    auto r = idx.query(Vector<float>{1.0f, 0.0f}, 1);  // dim=2 vs dim=3
    REQUIRE_FALSE(r.has_value());
    REQUIRE(r.error().code == Error::Code::DimensionMismatch);
}

// ── Single vector ─────────────────────────────────────────────────────────────

TEST_CASE("HNSW: insert one vector and query it back", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(3);
    auto ins = idx.insert(Vector<float>{1.0f, 0.0f, 0.0f});
    REQUIRE(ins.has_value());

    auto r = idx.query(Vector<float>{1.0f, 0.0f, 0.0f}, 1);
    REQUIRE(r.has_value());
    REQUIRE(r->size() == 1);
    REQUIRE(r->at(0) == *ins);
}

// ── Basic correctness ─────────────────────────────────────────────────────────

TEST_CASE("HNSW: nearest of two distinct vectors", "[hnsw]") {
    // Insert two vectors that differ clearly in L2 distance from the query.
    HNSWIndex<Metric::L2> idx(2);
    auto id_near = idx.insert(Vector<float>{1.0f, 0.0f}).value();  // dist²=0 from query
    auto id_far  = idx.insert(Vector<float>{5.0f, 5.0f}).value();  // dist²=50 from query

    auto r = idx.query(Vector<float>{1.0f, 0.0f}, 1);
    REQUIRE(r.has_value());
    REQUIRE(r->at(0) == id_near);
    (void)id_far;
}

TEST_CASE("HNSW: top-2 results are the two nearest", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(2);
    auto id0 = idx.insert(Vector<float>{0.0f, 0.0f}).value();  // distance = 1
    auto id1 = idx.insert(Vector<float>{1.0f, 0.0f}).value();  // distance = 0 (exact)
    auto id2 = idx.insert(Vector<float>{9.0f, 9.0f}).value();  // distance = large

    auto r = idx.query(Vector<float>{1.0f, 0.0f}, 2);
    REQUIRE(r.has_value());
    REQUIRE(r->size() == 2);
    // Closest first: id1 then id0
    REQUIRE(r->at(0) == id1);
    REQUIRE(r->at(1) == id0);
    (void)id2;
}

TEST_CASE("HNSW: k larger than index size is clamped", "[hnsw]") {
    HNSWIndex<Metric::L2> idx(2);
    idx.insert(Vector<float>{1.0f, 0.0f});
    idx.insert(Vector<float>{2.0f, 0.0f});

    // Ask for 10 but only 2 exist — should return 2, not error
    auto r = idx.query(Vector<float>{1.0f, 0.0f}, 10);
    REQUIRE(r.has_value());
    REQUIRE(r->size() == 2);
}

// ── Multiple metrics ──────────────────────────────────────────────────────────

TEST_CASE("HNSW<Cosine>: nearest by angle", "[hnsw]") {
    HNSWIndex<Metric::Cosine> idx(2);

    // id_near points roughly in the same direction as the query [1, 0.1]
    auto id_near = idx.insert(Vector<float>{10.0f, 1.0f}).value(); // same direction, different magnitude
    auto id_far  = idx.insert(Vector<float>{-1.0f, 0.0f}).value(); // opposite direction

    auto r = idx.query(Vector<float>{1.0f, 0.1f}, 1);
    REQUIRE(r.has_value());
    REQUIRE(r->at(0) == id_near);
    (void)id_far;
}

TEST_CASE("HNSW<InnerProduct>: highest dot product returned first", "[hnsw]") {
    HNSWIndex<Metric::InnerProduct> idx(2);

    // query = [1, 0]. Inner product with [5,0] = 5, with [1,10] = 1.
    auto id_high = idx.insert(Vector<float>{5.0f, 0.0f}).value();
    auto id_low  = idx.insert(Vector<float>{1.0f, 10.0f}).value();

    auto r = idx.query(Vector<float>{1.0f, 0.0f}, 1);
    REQUIRE(r.has_value());
    REQUIRE(r->at(0) == id_high);
    (void)id_low;
}

// ── Recall vs FlatIndex (ground truth) ───────────────────────────────────────

TEST_CASE("HNSW<L2>: recall@10 >= 0.8 on 500 random vectors dim=16", "[hnsw][recall]") {
    // This is the key quality test: HNSW must find at least 80% of the true
    // nearest neighbours. With M=16, ef_construction=200, ef_search=50 this
    // should reach >95% recall in practice.

    const std::size_t DIM   = 16;
    const std::size_t N     = 500;
    const std::size_t K     = 10;
    const std::size_t EF    = 50;

    std::mt19937 rng(42);  // fixed seed for reproducibility

    HNSWIndex<Metric::L2>  hnsw(DIM, 16, 200);
    FlatIndex<Metric::L2>  flat(DIM);

    // Insert the same N vectors into both indexes
    for (std::size_t i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        hnsw.insert(vec);
        flat.insert(vec);
    }

    // Run 20 random queries and average recall
    float total_recall = 0.0f;
    const std::size_t NUM_QUERIES = 20;

    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        auto query = random_vector(DIM, rng);

        auto hnsw_r = hnsw.query(query, K, EF);
        auto flat_r = flat.query(query, K);

        REQUIRE(hnsw_r.has_value());
        REQUIRE(flat_r.has_value());

        total_recall += recall_at_k(*hnsw_r, *flat_r);
    }

    float avg_recall = total_recall / static_cast<float>(NUM_QUERIES);
    INFO("Average recall@10 = " << avg_recall);
    REQUIRE(avg_recall >= 0.80f);
}

TEST_CASE("HNSW<L2>: recall@1 == 1.0 on 50 vectors (small exact test)", "[hnsw][recall]") {
    // With a very small index and high ef_search, HNSW should be exact.
    const std::size_t DIM = 4;
    const std::size_t N   = 50;

    std::mt19937 rng(7);

    HNSWIndex<Metric::L2> hnsw(DIM, 8, 100);
    FlatIndex<Metric::L2> flat(DIM);

    for (std::size_t i = 0; i < N; ++i) {
        auto vec = random_vector(DIM, rng);
        hnsw.insert(vec);
        flat.insert(vec);
    }

    // With ef_search = N, HNSW degenerates to exact search → recall must be 1.0
    std::size_t perfect = 0;
    const std::size_t NUM_QUERIES = 10;

    for (std::size_t q = 0; q < NUM_QUERIES; ++q) {
        auto query = random_vector(DIM, rng);

        auto hnsw_r = hnsw.query(query, 1, /*ef_search=*/N);
        auto flat_r = flat.query(query, 1);

        REQUIRE(hnsw_r.has_value());
        REQUIRE(flat_r.has_value());

        if (hnsw_r->at(0) == flat_r->at(0)) ++perfect;
    }

    INFO("Exact matches: " << perfect << " / " << NUM_QUERIES);
    REQUIRE(perfect == NUM_QUERIES);
}
