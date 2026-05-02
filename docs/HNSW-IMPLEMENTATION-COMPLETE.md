# HNSW Implementation — Complete Guide

This is your master reference for the three core methods of HNSW:
1. **insert()** — add a new vector to the index
2. **query()** — find k nearest neighbors for a query vector
3. **search_layer()** — helper used by both insert and query

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [The Three Methods](#the-three-methods)
3. [How They Work Together](#how-they-work-together)
4. [Implementation Checklist](#implementation-checklist)

---

## Core Concepts

### The Graph Structure

HNSW builds a **layered hierarchical graph**:

```
Layer 2:  A ─────────── E          (sparse, top layer)
Layer 1:  A ── C ─ D ── E
Layer 0:  A─B─C─D─E─F─G─H        (dense, all nodes)
```

**Key facts:**
- Layer 0 contains **all** nodes
- Each higher layer contains a **random subset** (exponential distribution)
- Each node can have up to M neighbors per layer (2M at layer 0)
- Edges are **bidirectional** — if A→B, then B→A

### Entry Point

The index tracks one global **entry_point_** — the node at the highest layer. All queries and inserts start here.

---

## The Three Methods

### 1. insert(vec) → VectorId

**Purpose:** Add a new vector to the index.

**Algorithm (5 main steps):**

#### Step A: Setup
```cpp
// Check dimension, store the vector, assign a random level
VectorId new_id = store_.insert(vec);
int new_level = assign_level();
nodes_.emplace_back(new_level);

// If this is the first node:
if (entry_point_ == kInvalidId) {
    entry_point_ = new_id;
    max_level_ = new_level;
    return new_id;  // Done!
}
```

#### Step F: Descend Upper Layers
Navigate from `max_level_` down to `new_level+1` with ef=1:

```cpp
std::vector<std::pair<float, VectorId>> ep = {
    { node_dist(entry_point_, vec.data()), entry_point_ }
};
for (int lc = max_level_; lc > new_level; --lc) {
    auto result = search_layer(ep, vec.data(), 1, lc);
    ep = { result.top() };
}
```

**Why:** Find the right region in upper layers to enter the layers where we'll build edges.

#### Step G: Build Edges at Each Layer
For each layer from `min(new_level, max_level_)` down to 0:

```cpp
for (int lc = std::min(new_level, max_level_); lc >= 0; --lc) {
    // Find candidates at this layer
    auto candidates = search_layer(ep, vec.data(), ef_construction_, lc);
    
    // How many neighbors for this layer?
    std::size_t M_lc = (lc == 0) ? 2 * M_ : M_;
    
    // Pick the best M_lc candidates
    auto nbs = select_neighbors(candidates, M_lc);
    
    // new_id → nbs (new node points to neighbors)
    nodes_[new_id].neighbors[lc] = nbs;
    
    // nbs → new_id (make edges bidirectional and shrink if needed)
    for (VectorId nb : nbs) {
        nodes_[nb].neighbors[lc].push_back(new_id);
        
        // If nb now has too many neighbors, shrink it
        if (nodes_[nb].neighbors[lc].size() > M_lc) {
            MaxHeap nb_cands;
            for (VectorId existing : nodes_[nb].neighbors[lc]) {
                float d = node_dist(existing, store_.data() + nb * store_.dim());
                nb_cands.emplace(d, existing);
            }
            nodes_[nb].neighbors[lc] = select_neighbors(nb_cands, M_lc);
        }
    }
    
    // Use these neighbors as entry point for next layer
    ep.clear();
    for (VectorId nb : nbs) {
        ep.emplace_back(node_dist(nb, vec.data()), nb);
    }
}
```

**Why:** Connect the new node at each layer, starting high and going low. At each layer, find candidates, pick the best M, and create bidirectional edges.

#### Step H: Update Entry Point
If the new node is higher than the current max level:

```cpp
if (new_level > max_level_) {
    entry_point_ = new_id;
    max_level_ = new_level;
}
```

---

### 2. query(vec, k, ef_search) → vector\<VectorId\>

**Purpose:** Find k approximate nearest neighbors for a query vector.

**Algorithm (5 main steps):**

#### Step A: Guards
```cpp
auto actual_k = std::min(k, store_.size());
auto actual_ef = std::max(ef_search, actual_k);
```

Ensure we don't ask for more results than exist, and that `ef >= k`.

#### Step B: Seed
Start at the global entry point:

```cpp
std::vector<std::pair<float, VectorId>> ep = {
    { node_dist(entry_point_, vec.data()), entry_point_ }
};
```

#### Step C: Descend
Navigate from `max_level_` down to layer 1 with ef=1:

```cpp
for (int lc = max_level_; lc > 0; --lc) {
    auto result = search_layer(ep, vec.data(), 1, lc);
    ep = { result.top() };
}
```

**Why:** Quickly jump through upper sparse layers to the right region.

#### Step D: Layer 0 Search
Perform full beam search at layer 0:

```cpp
auto found = search_layer(ep, vec.data(), actual_ef, 0);
```

**Why:** Layer 0 has all nodes. This is where the actual search happens.

#### Step E: Extract and Reverse
Get top k and reverse for closest-first order:

```cpp
std::vector<VectorId> results;
while (!found.empty() && results.size() < actual_k) {
    results.push_back(found.top().second);
    found.pop();
}
std::reverse(results.begin(), results.end());

return make_ok(results);
```

**Why:** The heap gives us worst-first; we reverse to get closest-first.

---

### 3. search_layer(entry_points, query, ef, layer) → max-heap

**Purpose:** Greedy beam search at one specific layer.

**Algorithm:**
```cpp
MinHeap candidates;          // Min-heap: best (closest) at top for expansion
MaxHeap found;               // Max-heap: worst (farthest) at top for eviction
std::unordered_set visited;  // Track visited nodes to avoid revisiting

// Initialize with entry points
for (const auto &ep : entry_points) {
    candidates.push(ep);
    found.push(ep);
    visited.insert(ep.second);
}

while (!candidates.empty()) {
    // Pop the best unexplored candidate
    const float c_dist = candidates.top().first;
    VectorId c_id = candidates.top().second;
    candidates.pop();
    
    // Stop condition: can't improve anymore
    if (c_dist > found.top().first && found.size() == ef) {
        break;
    }
    
    // Guard: only access neighbors[layer] if this node reaches that layer
    if (static_cast<int>(nodes_[c_id].neighbors.size()) > layer) {
        // Explore all neighbors of c at this layer
        for (VectorId nb : nodes_[c_id].neighbors[layer]) {
            if (visited.count(nb) == 0) {
                visited.insert(nb);
                float d = node_dist(nb, query);
                
                // Add to results if it's good enough
                if (d < found.top().first || found.size() < ef) {
                    candidates.push({d, nb});
                    found.push({d, nb});
                    if (found.size() > ef) {
                        found.pop();  // Evict the worst
                    }
                }
            }
        }
    }
}

return found;  // Return the max-heap of top ef nodes
```

**Key insight:** Keep two heaps in sync:
- `candidates` (min-heap) — unexplored nodes, best first (for expansion)
- `found` (max-heap) — results so far, worst first (for eviction)

Stop when the best candidate is worse than the worst result — nothing left to explore will be better.

---

## How They Work Together

### Insert Flow

```
insert(vec)
  ↓
  [A] Store vector, assign level, create node
  ↓
  [First node?] → Yes → Set entry point, done
  ↓
  [F] Descend upper layers using search_layer(ef=1)
  ↓
  [G] For each layer:
      ├─ search_layer(ef=ef_construction) to find candidates
      ├─ select_neighbors to pick best M
      ├─ Build edges: new_id → neighbors
      ├─ Build reverse edges: neighbors → new_id (with shrinking)
      └─ Move to next layer using neighbors as entry point
  ↓
  [H] If new_level > max_level_, update entry point
  ↓
  return new_id
```

### Query Flow

```
query(vec, k, ef_search)
  ↓
  [A] Guard: actual_k, actual_ef
  ↓
  [B] Seed: ep = entry_point_
  ↓
  [C] Descend upper layers using search_layer(ef=1)
  ↓
  [D] search_layer(ef=actual_ef) at layer 0
  ↓
  [E] Extract top k, reverse for closest-first
  ↓
  return results
```

### Relationship to search_layer

Both `insert` and `query` call `search_layer` multiple times:

| Method | ef value | Layer | Why |
|--------|----------|-------|-----|
| insert (step F) | 1 | max_level → new_level+1 | Quick navigation |
| insert (step G) | ef_construction | all (high to 0) | Find good neighbors |
| query (step C) | 1 | max_level → 1 | Quick navigation |
| query (step D) | actual_ef | 0 only | Thorough search |

---

## Implementation Checklist

### Required Methods
- [ ] `assign_level()` — sample random level from exponential distribution
- [ ] `node_dist(id, query)` — distance between a node and a query vector
- [ ] `select_neighbors(candidates, M)` — pick M closest from a heap
- [ ] `search_layer(entry_points, query, ef, layer)` — beam search one layer
- [ ] `insert(vec)` — add vector (steps A, F, G, H)
- [ ] `query(vec, k, ef_search)` — find neighbors (steps A, B, C, D, E)

### Edge Cases to Handle
- [ ] First insertion (empty index)
- [ ] Query with k=0 (guard against)
- [ ] Query with k > index size (use actual_k = min(k, size))
- [ ] Node accessing neighbors[layer] when layer doesn't exist (guard check)
- [ ] Neighbor list exceeding M limit (shrinking logic)
- [ ] ef_construction or ef_search being 0

### Testing
Run tests with:
```bash
./run-tests.sh
```

Expected test coverage:
- Single insert (first node)
- Multiple inserts (nodes at various levels)
- Queries on populated index
- Queries returning fewer results than k (small index)
- Distance metrics (L2, Cosine, InnerProduct)
- Edge shrinking during insert

---

## Code Patterns

### Using the Heaps

**Min-Heap** (best at top):
```cpp
MinHeap candidates;
candidates.push({distance, node_id});
auto best = candidates.top();  // smallest distance
```

**Max-Heap** (worst at top):
```cpp
MaxHeap found;
found.push({distance, node_id});
auto worst = found.top();  // largest distance
if (found.size() > ef) found.pop();  // evict worst
```

### Guarding Layer Access

Always check before accessing `neighbors[layer]`:
```cpp
if (static_cast<int>(nodes_[node_id].neighbors.size()) > layer) {
    // Safe to access neighbors[layer]
    for (VectorId nb : nodes_[node_id].neighbors[layer]) {
        // ...
    }
}
```

### Distance Calculation

```cpp
float d = node_dist(neighbor_id, query_vector);
// Returns distance between stored vector at neighbor_id and query_vector
```

---

## Common Debugging Tips

**Insert not building edges?**
- Check that step G loop goes from `min(new_level, max_level_)` down to 0
- Verify M_lc calculation (2*M at layer 0, M elsewhere)
- Ensure bidirectional edges are created (nb → new_id)

**Query returning wrong results?**
- Verify descend loop goes to layer 1, not layer 0
- Check that actual_ef >= actual_k
- Confirm reversal happens after extraction

**Shrinking not working?**
- Verify shrinking only happens when size > M_lc
- Check that candidate heap is rebuilt with correct distances
- Ensure select_neighbors picks the M_lc closest

**Segfault on layer access?**
- Add guard: `if (nodes_[id].neighbors.size() > layer)`
- Remember: neighbors.size() == level+1, not level

---

## Performance Characteristics

| Operation | Time | Space |
|-----------|------|-------|
| insert | O(ef_construction * log N) | O(M) |
| query | O(ef_search * log N) | O(ef_search) |
| Memory per node | O(M * avg_level) | ~(level+1)*M*sizeof(VectorId) |

Where N = total nodes in index.

---

## References

- Original paper: Malkov & Yashunin (2018) — "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
- Detailed step-by-step guides:
  - `docs/hnsw-insert-fgh.md` — Deep dive on insert steps F, G, H
  - `docs/hnsw-query-detailed.md` — Deep dive on query steps A-E

