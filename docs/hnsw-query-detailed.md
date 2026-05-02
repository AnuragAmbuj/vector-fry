# HNSW Query: Complete Breakdown

This guide explains how the query algorithm works — how you find the k nearest neighbors in the HNSW graph.

---

## Context

You have:
- A populated HNSW graph with nodes at various layers
- A query vector (new data point you want to find neighbors for)
- A target: find k approximate nearest neighbors

The graph looks like:
```
Layer 2:  A ─────────────── E          (sparse, few nodes)
Layer 1:  A ── C ─ D ─────── E
Layer 0:  A─B─C─D─E─F─G─H─I─J        (all nodes, dense)
           ↑
      entry_point_
```

---

## Algorithm Overview

Query has **5 steps**:

1. **A (Guards):** Check preconditions
2. **B (Seed):** Start at the global entry point
3. **C (Descend):** Navigate from the top layer down to layer 1
4. **D (Layer 0 Search):** Do the actual nearest-neighbor search at layer 0
5. **E (Extract & Reverse):** Return top k results in closest-first order

---

## Step A — Guards

```cpp
if (k == 0) {
    throw std::invalid_argument("query: k must be greater than zero");
}

auto actual_k = std::min(k, store_.size());
auto actual_ef = std::max(ef_search, actual_k);
```

**What it does:** Validates the input and computes working values.

**Why:**
- `k == 0` is invalid — you need to find at least 1 neighbor
- `actual_k` = how many results we actually want (can't return more than total nodes)
- `actual_ef` = beam width during layer 0 search. **Must be >= k** because you need to explore enough candidates to find the top k

**Example:**
```
User asks: query(vec, k=5, ef_search=20)
Nodes in index: 1000

actual_k  = min(5, 1000) = 5
actual_ef = max(20, 5) = 20

We'll search with 20 candidates to find the top 5.
```

---

## Step B — Seed

```cpp
std::vector<std::pair<float, VectorId>> ep = {
    { node_dist(entry_point_, vec.data()), entry_point_ }
};
```

**What it does:** Start with a single entry point — the global top-level node.

**Why:** Every query starts at the same place: `entry_point_` at `max_level_`.

**Data structure:** `ep` is a vector of `(distance, node_id)` pairs. We compute the distance from the entry point to the query vector.

---

## Step C — Descend

```cpp
for (int lc = max_level_; lc > 0; --lc) {
    auto result = search_layer(ep, vec.data(), 1, lc);
    ep = { result.top() };
}
```

**What it does:** Navigate from the top layer down to layer 1, keeping only the single closest node at each layer.

**Why:** 
- The upper layers are sparse (few nodes) with long-range links
- They let us quickly "jump" toward the query region
- Once we reach layer 0, we switch to a broader beam search

**Example:**
```
max_level_ = 3, query = "pizza"

Layer 3: search from entry point → found "food" (closest)
Layer 2: search from "food" → found "italian" (closer to pizza)
Layer 1: search from "italian" → found "restaurant" (even closer)
Layer 0: NOW do the full beam search from "restaurant" with ef=actual_ef
```

**Key point:** `ef=1` means keep only 1 candidate. We don't need to explore much at the upper layers because we just need to navigate to the right region. Once at layer 0, we explore broadly.

---

## Step D — Layer 0 Search

```cpp
auto found = search_layer(ep, vec.data(), actual_ef, 0);
```

**What it does:** Perform greedy beam search at layer 0 with `actual_ef` candidates.

**Why:** Layer 0 has ALL nodes, so this is where the actual nearest-neighbor search happens.

**What we get:** `found` is a max-heap (worst-first) containing the top `actual_ef` closest nodes to the query vector.

**Example:**
```
actual_ef = 20
After search_layer at layer 0:

found = max-heap with top 20 closest nodes
        {
            {10.5, node_42},  ← farthest (at .top())
            {8.2, node_15},
            {5.1, node_7},
            ...
            {0.3, node_99}    ← closest
        }
```

---

## Step E — Extract and Reverse

```cpp
std::vector<VectorId> results;
while (!found.empty() && results.size() < actual_k) {
    results.push_back(found.top().second);
    found.pop();
}
std::reverse(results.begin(), results.end());

return make_ok(results);
```

**What it does:**
1. Extract the top `actual_k` nodes from the max-heap (worst-first order)
2. Reverse them so closest nodes come first

**Why the reversal?**
- The max-heap gives us worst-first (worst node at `.top()`)
- We pop in that order, so `results` is: `[worst, ..., best]`
- Reversing gives us: `[best, ..., worst]` — closest-first order, as expected

**Example:**
```
Before reversal:
results = [node_42 (dist=10.5), node_15 (dist=8.2), node_7 (dist=5.1), node_99 (dist=0.3)]
                                                                        ↑ just popped

After reversal:
results = [node_99 (dist=0.3), node_7 (dist=5.1), node_15 (dist=8.2), node_42 (dist=10.5)]
           ↑ closest                                                    ↑ farthest in top-k
```

---

## Full Implementation

```cpp
template<Metric M>
auto HNSWIndex<M>::query(const Vector<float> &vec,
                         std::size_t k,
                         std::size_t ef_search) const -> Result<std::vector<VectorId> > {
    
    // A — guards
    if (k == 0) {
        throw std::invalid_argument("query: k must be greater than zero");
    }

    auto actual_k = std::min(k, store_.size());
    auto actual_ef = std::max(ef_search, actual_k);

    // B — seed: create entry point at max_level_
    std::vector<std::pair<float, VectorId>> ep = {
        { node_dist(entry_point_, vec.data()), entry_point_ }
    };

    // C — descend: from max_level_ down to layer 1 with ef=1
    for (int lc = max_level_; lc > 0; --lc) {
        auto result = search_layer(ep, vec.data(), 1, lc);
        ep = { result.top() };
    }

    // D — beam search at layer 0 with actual_ef candidates
    auto found = search_layer(ep, vec.data(), actual_ef, 0);

    // E — extract top k and reverse for closest-first order
    std::vector<VectorId> results;
    while (!found.empty() && results.size() < actual_k) {
        results.push_back(found.top().second);
        found.pop();
    }
    std::reverse(results.begin(), results.end());

    return make_ok(results);
}
```

---

## Visual Walkthrough

### Setup
```
Index has 5 nodes across 3 layers.
User queries: "pizza" (query vector)
Asks for: k=2 neighbors
ef_search=10

Computed:
  actual_k = min(2, 5) = 2
  actual_ef = max(10, 2) = 10
```

### Step B — Seed
```
Entry point is node A at layer 2.
ep = [{ dist(A, "pizza"), A }]
```

### Step C — Descend
```
Layer 2: search_layer with ef=1 from [A]
         → closest node is E
         → ep = [{ dist(E, "pizza"), E }]

Layer 1: search_layer with ef=1 from [E]
         → closest node is D
         → ep = [{ dist(D, "pizza"), D }]

Layer 0: ready to do full search
```

### Step D — Layer 0 Search
```
Layer 0: search_layer with ef=10 from [D]
         → explores and finds 10 closest nodes
         → found = max-heap:
           {
             {8.5, C},   ← worst
             {6.2, B},
             {3.1, E},
             {0.9, A}    ← best
           }
```

### Step E — Extract and Reverse
```
Extract top actual_k=2:

Pop from heap:
  results.push_back(C)  → results = [C]
  results.push_back(B)  → results = [C, B]
  (stop, size == actual_k)

Reverse:
  results = [B, C]  → closest first!
```

### Return
```
query() returns: [B, C]
Caller gets: "B is closest, then C"
```

---

## Key Differences: Query vs Insert

| Aspect | Insert | Query |
|--------|--------|-------|
| **Purpose** | Add a new node to the graph | Find neighbors for an external vector |
| **Graph changes** | Yes (adds node, creates edges) | No (read-only) |
| **Complexity** | O(M * log N) — builds many edges | O(log N * ef) — explores candidates |
| **Layer 0 beam width** | `ef_construction` (high, 200) | `ef_search` (lower, 50) |
| **Return value** | The ID of the inserted node | Top-k neighbor IDs |

---

## Common Mistakes

**C — Descend:**
- ❌ Using `ef > 1` at upper layers — wastes time exploring
- ✅ Use `ef=1` — just navigate to the right region

**D — Layer 0 Search:**
- ❌ Forgetting that `actual_ef` must be >= k
- ✅ Always enforce: `actual_ef = max(ef_search, actual_k)`

**E — Extract:**
- ❌ Forgetting to reverse — returns farthest-first instead of closest-first
- ✅ Always reverse the result before returning

**General:**
- ❌ Querying with k=0 — invalid
- ✅ Guard at the start: `if (k == 0) throw`

---

## Performance Tuning

**ef_search parameter:**
- `ef_search` = 50: fast, less accurate
- `ef_search` = 500: slow, more accurate
- General rule: `ef_search >= k`, ideally `2*k` to `4*k`

**Example:**
```
query(vec, k=10, ef_search=40)   // Fast, might miss some neighbors
query(vec, k=10, ef_search=100)  // Balanced
query(vec, k=10, ef_search=500)  // Thorough, slower
```

The actual search cost grows with `ef_search`, but recall (accuracy) improves.
