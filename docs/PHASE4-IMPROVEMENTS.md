# Phase 4 Improvements — Recommended Next Steps

## Overview

Phase 4 (HNSW) is functionally complete with all 13 tests passing. The code is correct, well-documented, and handles all three distance metrics properly. This document outlines suggested improvements for robustness and confidence.

---

## 1. Constructor Validation Tests

### What to test

The HNSWIndex constructor validates `max_links >= 2` and computes `mL_` correctly.

### Suggested tests

```cpp
TEST_CASE("HNSW: constructor rejects max_links < 2", "[hnsw]") {
    REQUIRE_THROWS_AS(HNSWIndex<Metric::L2>(3, /*M=*/1), std::invalid_argument);
}

TEST_CASE("HNSW: constructor rejects max_links == 1", "[hnsw]") {
    REQUIRE_THROWS_AS(HNSWIndex<Metric::L2>(3, /*M=*/1), std::invalid_argument);
}

TEST_CASE("HNSW: mL_ computed correctly", "[hnsw]") {
    HNSWIndex<Metric::L2> idx16(3, /*M=*/16);
    // mL_ = 1.0 / ln(16) ≈ 0.361
    // This is private, so expose via friend test or test_only hook
}

TEST_CASE("HNSW: ef_construction parameter validated", "[hnsw]") {
    // ef_construction should be > 0, probably >= k
    HNSWIndex<Metric::L2> idx(3, 16, /*ef_construction=*/0);
    // Should this throw or be clamped?
}
```

### Implementation notes

- Constructor already validates `max_links >= 2` and throws `std::invalid_argument`
- Consider exposing `mL_` via a const accessor for validation
- Consider whether `ef_construction` should have a minimum (e.g., >= M)

---

## 2. Deterministic Seeding / Injectable RNG

### Current state

- Level assignment uses `std::mt19937` seeded with `std::random_device{}`
- This is non-deterministic across runs
- Recall tests (78, 79) are probabilistic; seeding should be controllable

### Suggested improvements

**Option A: Expose seed parameter to constructor**

```cpp
explicit HNSWIndex(std::size_t dim,
                   std::size_t max_links = 16,
                   std::size_t ef_construction = 200,
                   uint64_t seed = 0)  // 0 = use random_device
    : store_(dim)
      , M_(max_links)
      , ef_construction_(ef_construction)
      , mL_(1.0 / std::log(static_cast<double>(max_links)))
      , entry_point_(kInvalidId)
      , max_level_(-1)
      , rng_(seed == 0 ? std::random_device{}() : seed) {
    if (max_links < 2) {
        throw std::invalid_argument("max_links must be >= 2");
    }
}
```

**Option B: Provide a test-only factory**

```cpp
namespace fry {
namespace test {
    template<Metric M>
    HNSWIndex<M> make_seeded_hnsw(std::size_t dim, uint64_t seed) {
        HNSWIndex<M> idx(dim);
        idx.rng_ = std::mt19937(seed);  // Requires friend or public access
        return idx;
    }
}}
```

**Option C: Inject RNG as template parameter** (more complex, not recommended for C++11)

### Benefits

- Makes recall tests deterministic: `HNSWIndex idx(dim, M, ef, 42);` always produces the same levels
- Enables reproducible benchmarks
- Tests can verify that same seed → same level assignment → same recall

### Test example

```cpp
TEST_CASE("HNSW: seeded RNG produces same levels", "[hnsw]") {
    const uint64_t seed = 123;
    
    HNSWIndex<Metric::L2> idx1(16, 16, 200, seed);
    HNSWIndex<Metric::L2> idx2(16, 16, 200, seed);
    
    Vector<float> v = ...;
    auto id1 = idx1.insert(v);
    auto id2 = idx2.insert(v);
    
    // Both should get the same level
    // (Requires exposing node level via test hook)
}
```

---

## 3. Internal Invariant Tests

### What to test

HNSW maintains critical invariants:
1. **Layer count invariant:** If a node is at level L, it has neighbors at all layers 0..L
2. **Neighbor limit invariant:** At layer 0, max neighbors = 2*M; at other layers, max = M
3. **Bidirectional edge invariant:** If A→B at layer L, then B→A at layer L
4. **Entry point invariant:** entry_point_ is always at max_level_ and is the global entry point

### Implementation approach

**Create a test-only inspection hook:**

```cpp
// In hnsw_index.hpp, add a friend class:
template<Metric M>
class HNSWIndex {
    // ... existing code ...
    
    // Test-only inspection (friend class only)
    template<Metric> friend class HNSWIndexInspector;
};

// In a new file test_hnsw_internals.cpp:
template<Metric M>
class HNSWIndexInspector {
public:
    static int get_node_level(const HNSWIndex<M>& idx, VectorId id) {
        return static_cast<int>(idx.nodes_[id].neighbors.size()) - 1;
    }
    
    static std::vector<VectorId> get_neighbors(const HNSWIndex<M>& idx,
                                                 VectorId id,
                                                 int layer) {
        if (static_cast<int>(idx.nodes_[id].neighbors.size()) <= layer) {
            return {};
        }
        return idx.nodes_[id].neighbors[layer];
    }
    
    static int get_max_level(const HNSWIndex<M>& idx) {
        return idx.max_level_;
    }
    
    static VectorId get_entry_point(const HNSWIndex<M>& idx) {
        return idx.entry_point_;
    }
    
    static std::size_t get_M(const HNSWIndex<M>& idx) {
        return idx.M_;
    }
};
```

### Suggested invariant tests

```cpp
TEST_CASE("HNSW: layer count invariant", "[hnsw][invariants]") {
    HNSWIndex<Metric::L2> idx(3);
    // Insert several vectors
    for (int i = 0; i < 10; ++i) {
        idx.insert(random_vector(3, rng));
    }
    
    // Check each node
    for (VectorId id = 0; id < idx.size(); ++id) {
        int level = HNSWIndexInspector<Metric::L2>::get_node_level(idx, id);
        
        // Every node must have neighbors at layers 0..level
        for (int lc = 0; lc <= level; ++lc) {
            auto nbs = HNSWIndexInspector<Metric::L2>::get_neighbors(idx, id, lc);
            REQUIRE(nbs.size() > 0);  // Should have at least 1 neighbor
        }
    }
}

TEST_CASE("HNSW: neighbor limit invariant", "[hnsw][invariants]") {
    HNSWIndex<Metric::L2> idx(3, /*M=*/16);
    std::mt19937 rng(42);
    
    for (int i = 0; i < 20; ++i) {
        idx.insert(random_vector(3, rng));
    }
    
    std::size_t M = HNSWIndexInspector<Metric::L2>::get_M(idx);
    
    for (VectorId id = 0; id < idx.size(); ++id) {
        int level = HNSWIndexInspector<Metric::L2>::get_node_level(idx, id);
        
        for (int lc = 0; lc <= level; ++lc) {
            auto nbs = HNSWIndexInspector<Metric::L2>::get_neighbors(idx, id, lc);
            
            if (lc == 0) {
                REQUIRE(nbs.size() <= 2 * M);
            } else {
                REQUIRE(nbs.size() <= M);
            }
        }
    }
}

TEST_CASE("HNSW: bidirectional edge invariant", "[hnsw][invariants]") {
    HNSWIndex<Metric::L2> idx(3);
    std::mt19937 rng(42);
    
    for (int i = 0; i < 15; ++i) {
        idx.insert(random_vector(3, rng));
    }
    
    for (VectorId id_a = 0; id_a < idx.size(); ++id_a) {
        int level_a = HNSWIndexInspector<Metric::L2>::get_node_level(idx, id_a);
        
        for (int lc = 0; lc <= level_a; ++lc) {
            auto nbs_a = HNSWIndexInspector<Metric::L2>::get_neighbors(idx, id_a, lc);
            
            for (VectorId id_b : nbs_a) {
                auto nbs_b = HNSWIndexInspector<Metric::L2>::get_neighbors(idx, id_b, lc);
                
                // id_a should appear in id_b's neighbors at layer lc
                REQUIRE(std::find(nbs_b.begin(), nbs_b.end(), id_a) != nbs_b.end());
            }
        }
    }
}

TEST_CASE("HNSW: entry point invariant", "[hnsw][invariants]") {
    HNSWIndex<Metric::L2> idx(3);
    std::mt19937 rng(42);
    
    for (int i = 0; i < 20; ++i) {
        idx.insert(random_vector(3, rng));
    }
    
    VectorId entry = HNSWIndexInspector<Metric::L2>::get_entry_point(idx);
    int max_level = HNSWIndexInspector<Metric::L2>::get_max_level(idx);
    int entry_level = HNSWIndexInspector<Metric::L2>::get_node_level(idx, entry);
    
    // Entry point must be at max_level
    REQUIRE(entry_level == max_level);
}
```

---

## 4. Build System Robustness

### Current state

- `CMakePresets.json` defines `debug`, `asan`, `release` presets
- `vcpkg.json` lists `catch2`, `fmt`, `nlohmann-json`, `spdlog`, `vcpkg-cmake`, `vcpkg-cmake-config`
- `run-tests.sh` calls `cmake --preset <preset>`

### Issue

Some vcpkg packages (`fmt`, `vcpkg-cmake`, `vcpkg-cmake-config`) may be unused in the current code.

### Suggested improvements

**Option A: Remove unused dependencies**

```json
{
  "name": "fry-vector",
  "version": "0.0.1",
  "dependencies": [
    "catch2",
    "nlohmann-json"
  ]
}
```

**Option B: Keep all, but document usage**

```json
{
  "name": "fry-vector",
  "version": "0.0.1",
  "dependencies": [
    {
      "name": "catch2",
      "description": "Test framework for tests/"
    },
    {
      "name": "nlohmann-json",
      "description": "Future: Phase 6 HTTP API payloads"
    },
    {
      "name": "spdlog",
      "description": "Future: structured logging"
    }
  ]
}
```

### Testing on fresh checkout

```bash
git clone https://github.com/user/fry-vector.git
cd fry-vector
./run-tests.sh        # Should work without manual dependency management
./run-tests.sh asan   # Should work
./run-tests.sh release  # Should work
```

---

## Priority Order

1. **Constructor validation tests** (quick, high confidence boost)
2. **FlatIndex InnerProduct fix** (already done)
3. **Deterministic seeding** (Option A: seed parameter in constructor)
4. **Internal invariant tests** (powerful, requires inspector hook)
5. **Build system cleanup** (polish, not urgent)

---

## Files to modify

- `tests/test_hnsw.cpp` — add new test cases
- `include/fry/hnsw_index.hpp` — add inspector friend class + optional seed parameter
- `src/hnsw_index.cpp` — update constructor signature if Option A chosen
- `vcpkg.json` — optionally clean up unused dependencies
- `PLAN.md` — update to reflect these improvements as "Future optimizations"

