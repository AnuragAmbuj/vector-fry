# fry-vector — Developer Blog Series

> Building a production-ready vector database from scratch in C++11.
> Each post covers one phase: the concept, the math, the code, and the C++11 gotchas.

---

## Posts

| # | Title | Topics | Level |
|---|---|---|---|
| [1](./01-what-is-a-vector-database.md) | What Is a Vector Database? | Embeddings, similarity search, use cases, 7-phase overview | Layman |
| [2](./02-core-types.md) | Core Types: Vector, Result, VectorStore | `Vector<T>`, `Result<T>` (placement new, Rule of Five), slab layout | Intermediate |
| [3](./03-distance-metrics.md) | Distance Metrics: L2, Cosine, Inner Product | Math, template dispatch, negation trick | Intermediate |
| [4](./04-flat-index.md) | FlatIndex: Brute-Force Top-K Search | Bounded max-heap, O(N log K), slab pointer arithmetic | Intermediate |
| [5](./05-simd.md) | SIMD: NEON and AVX2 Acceleration | ARM NEON, x86 AVX2, FMA, CMake detection, unified dispatcher | Advanced |

---

## Series Architecture

```mermaid
flowchart LR
    B1["Blog 1\nWhat & Why"] --> B2["Blog 2\nFoundation\nTypes"]
    B2 --> B3["Blog 3\nDistance\nMath"]
    B3 --> B4["Blog 4\nSearch\nAlgorithm"]
    B4 --> B5["Blog 5\nSIMD\nPerformance"]
    B5 --> B6["Blog 6\nHNSW\n(coming)"]
    B6 --> B7["Blog 7\nPersistence\n(coming)"]

    style B1 fill:#4ade80,color:#000
    style B2 fill:#4ade80,color:#000
    style B3 fill:#4ade80,color:#000
    style B4 fill:#4ade80,color:#000
    style B5 fill:#4ade80,color:#000
    style B6 fill:#94a3b8,color:#000
    style B7 fill:#94a3b8,color:#000
```

---

## File Map

```
fry-vector/
├── include/fry/
│   ├── types.hpp          ← Blog 2: Vector<T>, VectorId
│   ├── result.hpp         ← Blog 2: Result<T>, Error
│   ├── vector_store.h     ← Blog 2: VectorStore (slab layout)
│   ├── distance.hpp       ← Blog 3: l2_squared, cosine, inner_product, dispatch<M>
│   ├── flat_index.hpp     ← Blog 4: FlatIndex<M>, bounded max-heap
│   ├── distance_neon.hpp  ← Blog 5: ARM NEON kernels
│   ├── distance_avx2.hpp  ← Blog 5: x86 AVX2 kernels
│   └── distance_simd.hpp  ← Blog 5: unified dispatcher
├── src/
│   └── vector_store.cpp   ← Blog 2: insert, search implementation
└── tests/
    ├── test_vector_store.cpp
    ├── test_distance.cpp
    ├── test_flat_index.cpp
    └── test_distance_simd.cpp
```

---

*Source: [github.com/anuragambuj/fry-vector](https://github.com/anuragambuj/fry-vector)*
