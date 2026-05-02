# fry-vector — Build Plan

A 7-phase plan to build a production-ready vector database in C++ from scratch.
The goal is to learn C++ and vector database internals simultaneously by implementing everything bottom-up.

**Standard:** C++11
**Compiler:** Clang (AppleClang on arm64-osx) + Clang (x86_64 on i7 MacBook)
**Build system:** CMake 3.25+ with Ninja, vcpkg for dependencies
**Testing:** Catch2 v3 via FetchContent
**Platforms:** ARM64/NEON (Apple M3) · x86_64/AVX2 (Intel i7)

---

## Progress Overview

| Phase | Title | Status |
|-------|-------|--------|
| 1 | Project Skeleton | ✅ Complete |
| 2 | Flat Index + Distance | ✅ Complete |
| 3 | SIMD Acceleration | ✅ Complete |
| 4 | HNSW Index | ✅ Complete |
| 5 | Persistence + WAL | ⬜ Upcoming |
| 6 | HTTP API + Metadata | ⬜ Upcoming |
| 7 | Concurrency | ⬜ Upcoming |

---

## Phase 1 — Project Skeleton ✅

Set up the repo, build system, and an end-to-end hello-world path.
**Goal:** compile and run a binary that inserts one vector and queries it back.

### Tooling

- [x] **Init CMake project with presets**
  `CMakeLists.txt`, `CMakePresets.json`, Debug + Release + ASan configs.
  *C++11 concept: build system structure, compile flags, preset inheritance.*

- [x] **Set up vcpkg manifest**
  `vcpkg.json` listing `nlohmann-json`, `spdlog`, Catch2.
  *Library: dependency management, manifest mode.*

- [x] **Configure clang-tidy and clang-format**
  `.clang-tidy`, `.clang-format` in repo root.
  *Tooling: static analysis and consistent code style.*

- [x] **Add AddressSanitizer and ThreadSanitizer presets**
  CMake `ENABLE_ASAN` flag; `asan` preset sets `-fsanitize=address,undefined`.
  *C++11 concept: runtime memory error detection.*

- [x] **`run-tests.sh`** — convenience script: `set -euo pipefail`, supports preset arg and extra ctest flags.

### Core Types

- [x] **Define `Vector<T>` wrapper over `std::vector<float>`**
  `include/fry/types.hpp` — owns data, knows its dimension.
  Constructed from `std::initializer_list<T>` or `std::vector<T>`.
  Exposes `dim()`, `data()`, `operator[]`, `operator==`.
  *C++11 concept: class templates, initializer lists, ownership semantics.*
  > **Note on CTAD:** In C++11 you must write `Vector<float>{1.0f, 0.0f}` explicitly.
  > Class Template Argument Deduction (`Vector{1.0f, 0.0f}`) is C++17.

- [x] **Define `VectorId` type alias (`uint64_t`)**
  Named alias for IDs — signals intent and avoids mixing IDs with sizes.
  `kInvalidId = std::numeric_limits<VectorId>::max()` as sentinel.
  *C++11 concept: `using` type aliases, `constexpr` at namespace scope.*

- [x] **Write `Result<T, E>` type (hand-rolled, not `std::expected`)**
  `include/fry/result.hpp` — `ResultImpl<T,E>` backed by `std::aligned_storage`.
  `Result<T>` is a C++11 alias template: `using Result = ResultImpl<T, Error>`.

  Key C++11 concepts demonstrated:
  - **`std::aligned_storage<Size, Align>`** — raw byte buffer with correct size/alignment.
  - **Placement new** — `new (&storage_) T(val)` constructs at a specific address without allocating.
  - **Explicit destructor call** — `ptr->~T()` is the only safe way to destroy a placement-new object.
  - **Rule of Five** — defining a custom destructor suppresses compiler-generated copy/move. All five must be written explicitly.

  `Error` struct carries a typed `Code` enum and a human-readable `message`.
  `make_ok<T>(val)` and `make_error<T>(err)` are the call-site factories.

### First Tests

- [x] **Add Catch2 test target to CMake**
  `tests/CMakeLists.txt` — FetchContent pulls Catch2 v3.7.1.

- [x] **Write smoke tests: 9 tests in `test_vector_store.cpp`**
  Round-trip identity, dimension mismatch, empty store, Result ok/error paths.

### Bugs Fixed

- `struct Result {}` + `using Result = ...` name conflict — removed empty struct.
- CTAD `Vector{1.0f}` → `Vector<float>{1.0f}` everywhere.
- Missing `operator->` on `ResultImpl` — added `T* operator->()`.

---

## Phase 2 — Flat Index + Distance ✅

A working brute-force vector search. Insert N vectors, query with a vector, get back K nearest neighbours. **Correct, not fast yet.**

### VectorStore (`src/vector_store.cpp`)

- [x] **Contiguous float32 slab** — `std::vector<float>` with stride = `dim_`. Vector `i` at `data_ + i * dim_`.
- [x] **Sequential VectorId assignment** — monotonic `next_id_` counter; ID == slab position, no separate map needed.
- [x] **`data()` accessor** — exposes raw slab pointer for FlatIndex and HNSW.

### Distance Functions (`include/fry/distance.hpp`)

- [x] **Scalar L2 squared** — `Σ(a[i]-b[i])²`, no `sqrt` (ordering preserved).
- [x] **Scalar cosine similarity** — `dot(a,b) / (|a| * |b|)`, zero-norm guard.
- [x] **Scalar inner product** — `Σ a[i]*b[i]`, used for MIPS.
- [x] **Template dispatch** — `template<Metric M> auto dispatch(...)` with `= delete` primary + three specialisations. Zero-cost: metric resolved at compile time.

### FlatIndex (`include/fry/flat_index.hpp`)

- [x] **`insert`** — delegates to VectorStore.
- [x] **`query(vec, k)`** — bounded max-heap O(N log K): heap_val negated for Cosine, extract + `std::reverse` for closest-first output.

### Tests

- [x] **31 tests in `test_distance.cpp`** — L2, cosine, inner product, dispatch specialisations.
- [x] **11 tests in `test_flat_index.cpp`** — insert, query, error cases, cosine metric correctness.

### Bugs Fixed

- `make_error<M>` used `Metric` as type param → `make_error<std::vector<VectorId>>`.
- Cosine heap: first branch used `dist` not `heap_val` → both branches must use `heap_val`.
- Cosine denominator `norm(a) + norm(b)` → `norm(a) * norm(b)`.
- `dispatch` specialisation had non-const `float*` → `const float*` to match primary.
- `heap.push(make_pair(...))` → `heap.emplace(...)` (clang-tidy).

---

## Phase 3 — SIMD Acceleration ✅

Make distance functions fast using platform SIMD. Dual-platform: ARM NEON (M3) + x86 AVX2 (i7).
**Target: 4–8× speedup on L2 distance vs scalar.**

### CMake Architecture Detection (`CMakeLists.txt`)

- [x] **`fry_simd` interface library** — detects `CMAKE_SYSTEM_PROCESSOR` at configure time.
  - `arm64|aarch64` → defines `FRY_HAVE_NEON=1` (NEON always present, no extra flags).
  - `x86_64|AMD64` → adds `-mavx2 -mfma`, defines `FRY_HAVE_AVX2=1`.
  - Unknown → scalar fallback only, no macros defined.

### NEON Kernels (`include/fry/distance_neon.hpp`) — ARM64/M3

- [x] **`l2_squared_neon`** — `float32x4_t` accumulator, `vld1q_f32` + `vsubq_f32` + `vfmaq_f32` loop (4 floats/iter), `vaddvq_f32` horizontal reduction, scalar tail for `dim % 4`.
- [x] **`inner_product_neon`** — same structure, `vfmaq_f32(acc, va, vb)` directly (no subtraction).
- [x] **`NOLINTBEGIN/END`** — suppresses `bugprone-easily-swappable-parameters` for raw `const float*` pairs.

### AVX2 Kernels (`include/fry/distance_avx2.hpp`) — x86_64/i7

- [x] **`hsum256`** — 5-step horizontal reduction: `hadd` × 2 → `extractf128` → `castps256_ps128` → `_mm_add_ps` → `_mm_cvtss_f32`.
- [x] **`l2_squared_avx2`** — `__m256` accumulator, `_mm256_loadu_ps` + `_mm256_sub_ps` + `_mm256_fmadd_ps` loop (8 floats/iter), `hsum256`, scalar tail for `dim % 8`.
- [x] **`inner_product_avx2`** — same structure, `_mm256_fmadd_ps(va, vb, acc)` directly.

### Unified Dispatcher (`include/fry/distance_simd.hpp`)

- [x] **`l2_squared_simd` / `inner_product_simd`** — compile-time `#if` chain: NEON → AVX2 → scalar. Zero runtime overhead.

### Tests

- [x] **`test_distance_simd.cpp`** — NEON tests gated on `FRY_HAVE_NEON`, AVX2 tests gated on `FRY_HAVE_AVX2`, dispatcher tests always compile on all platforms.

### Bugs Fixed

- `#endif FRY_HAVE_NEON` (missing `//`) → `#endif // FRY_HAVE_NEON`.
- `#if defined(...)` → `#ifdef` (user style preference, both valid).
- `0f` invalid literal → `0.0f`.
- `int i` loop variable vs `std::size_t dim` → changed to `std::size_t i`.

---

## Phase 4 — HNSW Index ✅

The real ANN index. Approximate search in O(log N) instead of O(N).

### Scaffolding — Done

- [x] **`include/fry/hnsw_index.hpp`** — full class definition with:
  - `HNSWNode` struct: `std::vector<std::vector<VectorId>> neighbors` (jagged, one entry per layer).
  - `HNSWIndex<M>` template: `M_`, `ef_construction_`, `mL_`, `entry_point_`, `max_level_`, `nodes_`, `store_`, `rng_`.
  - Private helpers declared: `node_dist`, `assign_level`, `search_layer`, `select_neighbors`.
  - Public API declared: `insert`, `query(vec, k, ef_search=50)`.
  - Comprehensive algorithm docs: Algorithms 1, 2, 3, 5 from the HNSW paper.
- [x] **`src/hnsw_index.cpp`** — full implementation with:
  - All 7 methods fully implemented: constructor, node_dist, assign_level, select_neighbors, search_layer, insert, query.
  - Explicit template instantiations: `L2`, `Cosine`, `InnerProduct`.
- [x] **`tests/test_hnsw.cpp`** — 13 tests covering:
  - Construction, insert/query error cases.
  - Single-vector round-trip, nearest of two, top-2 ordering, k clamping.
  - `Cosine` and `InnerProduct` metric correctness.
  - `recall@10 >= 0.80` on 500 random vectors (quality gate).
  - `recall@1 == 1.0` on 50 vectors with `ef_search=N` (exactness gate).
- [x] **`tests/CMakeLists.txt`** updated — `test_hnsw.cpp` added to `fry_tests`.
- [x] **`CMakeLists.txt`** updated — `hnsw_index.hpp` listed under `fry_vector_lib`.

### Implementation — Done

- [x] **`HNSWIndex` constructor** — member-init list + validates `max_links >= 2`.
  *C++11 concept: member initialiser order, `std::random_device` seeding.*

- [x] **`node_dist`** — slab pointer arithmetic + `dispatch<M>` + metric normalization.

- [x] **`assign_level`** — `floor(-log(uniform(0,1)) * mL_)`.
  *C++11 concept: `<random>`, `std::uniform_real_distribution`.*

- [x] **`select_neighbors`** — drain max-heap, take last `M` (the closest).
  *C++11 concept: priority_queue drain, vector slicing.*

- [x] **`search_layer`** — greedy beam search (Algorithm 2): min-heap candidates + max-heap found + `unordered_set` visited.
  *C++11 concept: multiple heap strategies, `unordered_set` for O(1) visited check.*

- [x] **`insert`** — orchestrate steps A–I: guard → store → level → node → first-node case → upper-layer descent (ef=1) → per-layer edge build + bidirectional shrink → update entry point.
  *C++11 concept: graph mutation, invariant maintenance, std::mt19937.*

- [x] **`query`** — guard → seed entry → descend layers max→1 (ef=1) → layer-0 beam search (ef=ef_search) → extract all, reverse, trim to top-K.

### Validation — Done

- [x] **Recall@10 >= 0.80** — passes on 500 random dim=16 vectors.
- [x] **Recall@1 == 1.0** — exact match on 50 vectors with `ef_search = N`.
- [x] **All metrics** — L2, Cosine, InnerProduct work correctly.
- [x] **Error handling** — dimension mismatch, empty index, invalid k all return proper errors.

### Documentation

- [x] **`docs/hnsw-insert-fgh.md`** — Detailed breakdown of insert steps F, G, H with pseudocode and visual examples.
- [x] **`docs/hnsw-query-detailed.md`** — Complete query algorithm walkthrough with examples and performance tuning guide.
- [x] **`docs/HNSW-IMPLEMENTATION-COMPLETE.md`** — Master reference covering all methods, how they work together, debugging tips.
- [x] **`docs/IMPLEMENTATION-SUMMARY.md`** — Quick summary of Phase 4 completion.

---

## Phase 5 — Persistence + WAL ⬜

Make data survive process restarts. Segment-based storage + Write-Ahead Log for crash safety.

### WAL

- [ ] **WAL file: append-only, length-prefixed records** — POSIX `write()`, `O_DSYNC` for durability.
  *C++11 concept: POSIX I/O, durability guarantees, fsync.*
- [ ] **WAL record types: Insert, Delete, Checkpoint** — enum + binary layout.
  *C++11 concept: binary serialisation, struct packing, endianness.*
- [ ] **WAL replay on startup** — read records, apply to in-memory index.
- [ ] **WAL rotation** — seal old log, open new after size threshold.

### Segments

- [ ] **Sealed segment** — write vector slab + HNSW graph to binary file.
  *C++11 concept: `#pragma pack`, endianness.*
- [ ] **`mmap` on load** — `mmap(PROT_READ, MAP_SHARED)`, zero-copy reads.
  *C++11 concept: virtual memory, page faults, `madvise`.*
- [ ] **Segment manifest** — `nlohmann/json` file listing segment id, count, path.
- [ ] **Boot sequence** — load manifest → mmap segments → replay WAL.

---

## Phase 6 — HTTP API + Metadata ⬜

Expose the database over HTTP and add payload/filter support.

### HTTP API (`cpp-httplib`)

- [ ] **`POST /vectors`** — insert with JSON payload.
- [ ] **`POST /query`** — ANN search, returns `[{id, score, payload}]`.
- [ ] **`DELETE /vectors/:id`** — soft delete (tombstone + WAL).
- [ ] **`GET /health` and `GET /stats`** — vector count, index type, segment count.

### Metadata + Filtering

- [ ] **`PayloadStore`** — `id → nlohmann::json` map, serialised to segment.
- [ ] **Filter expression AST** — `field == value`, `field > value`, AND. Recursive descent parser.
  *C++11 concept: AST design, recursive descent parsing.*
- [ ] **Post-filter ANN** — over-fetch `ef_search` candidates, apply filter, return top-K.

---

## Phase 7 — Concurrency ⬜

Make the DB safe and fast under concurrent reads and writes.

### Read-Write Safety

- [ ] **`std::shared_mutex` on VectorStore** — `shared_lock` for queries, `unique_lock` for inserts.
  *C++11 concept: reader-writer lock (`<shared_mutex>` is C++14).*
- [ ] **Lock-free segment reads** — immutable mmap requires no lock.
- [ ] **WAL writer on background thread** — `std::thread` + `std::condition_variable`.
  *C++11 concept: producer-consumer, spurious wakeups.*

### Lock-Free Hot Path

- [ ] **`std::atomic<VectorId>` entry point in HNSW** — `compare_exchange_weak` for safe updates.
  *C++11 concept: CAS loops, ABA problem, memory ordering.*
- [ ] **Epoch-based reclamation** — safe node deletion without locks.
  *C++11 concept: RCU pattern, hazard pointers.*
- [ ] **TSan clean run** — 8 writers + 8 readers, 100k ops each, zero data races.

---

## Developer Blogs (`docs/blogs/`)

Educational write-ups covering each phase for readers from layman to expert level.

| # | File | Status |
|---|------|--------|
| 1 | `01-what-is-a-vector-database.md` | ✅ Published |
| 2 | `02-core-types.md` | ✅ Published |
| 3 | `03-distance-metrics.md` | ✅ Published |
| 4 | `04-flat-index.md` | ✅ Published |
| 5 | `05-simd.md` | ✅ Published |
| 6 | HNSW deep dive | ⬜ Pending Phase 4 completion |
| 7 | Persistence + WAL | ⬜ Pending Phase 5 |

---

## C++11 Gotchas Log

| # | Feature | Available in | Symptom / Fix |
|---|---------|-------------|---------------|
| 1 | `std::expected<T,E>` | C++23 | Hand-roll `ResultImpl<T,E>` with `aligned_storage` + placement new |
| 2 | `std::span<T>` | C++20 | Expose `data()` + `dim()` as two-part view contract |
| 3 | `[[nodiscard]]` | C++17 | Remove attribute; compiler won't warn on ignored return values |
| 4 | CTAD (`Vector{1.0f}`) | C++17 | Must write `Vector<float>{1.0f}` explicitly in C++11 |
| 5 | `inline` variables | C++17 | Use `constexpr` (namespace-scope, internal linkage) or `static const` |
| 6 | `if constexpr` | C++17 | Use tag dispatch or explicit template specialisation |
| 7 | AVX2 / `<immintrin.h>` | x86 only | Must `#ifdef FRY_HAVE_AVX2` — `<immintrin.h>` absent on ARM |
| 8 | `alignas(32)` for AVX2 | x86 | Use `alignas(16)` for NEON (128-bit = 16 bytes) |
| 9 | Structured bindings `auto [a,b]` | C++17 | Write `float a = p.first; VectorId b = p.second;` in C++11 |
| 10 | `std::shared_mutex` | C++14 | Will need `#include <shared_mutex>` — technically post-C++11 |

---

## Key Concepts Reference

| Concept | Phase | File |
|---------|-------|------|
| Type aliases (`using`) | 1 | `types.hpp` |
| Class templates | 1 | `types.hpp` |
| Placement new | 1 | `result.hpp` |
| `std::aligned_storage` | 1 | `result.hpp` |
| Rule of Five | 1 | `result.hpp` |
| Contiguous slab layout | 1–2 | `vector_store.cpp` |
| `std::priority_queue` | 2 | `flat_index.hpp` |
| Template specialisation | 2 | `distance.hpp` |
| NEON SIMD intrinsics | 3 | `distance_neon.hpp` |
| AVX2 SIMD intrinsics | 3 | `distance_avx2.hpp` |
| CMake compile-time detection | 3 | `CMakeLists.txt` |
| `<random>` + exponential dist | 4 | `hnsw_index.cpp` |
| Jagged 2D vectors | 4 | `hnsw_index.hpp` |
| `std::unordered_set` | 4 | `hnsw_index.cpp` |
| `std::atomic` | 4, 7 | `hnsw_index.hpp` |
| POSIX I/O + `mmap` | 5 | `wal.cpp`, `segment.cpp` |
| `std::shared_mutex` | 7 | `vector_store.cpp` |
| `std::condition_variable` | 7 | `wal_writer.cpp` |
| Epoch-based reclamation | 7 | `hnsw_index.cpp` |
