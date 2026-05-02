# HNSW Insert: Steps F, G, H — Detailed Breakdown

This is a focused guide for the three most confusing steps of the insert algorithm.

---

## Context

You've just done:
- **Step A-E**: Stored the vector, assigned a level, created the node, handled the first-node case
- You have: `new_id`, `new_level`, `entry_point_`, `max_level_`

Now you need to connect the new node to the graph.

---

## Step F — Descend Upper Layers

**What it does:** Start at the top of the graph and navigate DOWN to the layer where we'll actually build edges.

**Why:** The new node enters the graph at layer 0. But if it got a high level (say, level 2), we need to find which layer-1 node is closest to it, so we can use that as an entry point for building layer-1 edges.

**Data we have:**
```
entry_point_      current global entry point (the top-level node)
max_level_        highest layer in graph right now
new_level         the level assigned to our new node
query vec         the vector we're inserting
```

**The descend:**

```
We're going from layer max_level_ down to layer new_level+1.
(If new_level >= max_level_, this loop doesn't run at all — skip to step G.)

Example: max_level_ = 3, new_level = 1

Before G:        After descend:
Layer 3: A       Layer 3: A  ← checked but didn't build edges
Layer 2: A-B     Layer 2: B  ← moved here
Layer 1: B-C-D   Layer 1: C  ← will build edges here
Layer 0: ...     Layer 0: ... ← will build edges here
```

**Pseudocode:**

```cpp
// Start from the current global entry point
std::vector<std::pair<float, VectorId>> ep = {
    { node_dist(entry_point_, vec.data()), entry_point_ }
};

// Descend each layer from max_level_ down to new_level+1
for (int lc = max_level_; lc > new_level; --lc) {
    // At layer lc, find the closest node using greedy search (ef=1)
    auto result = search_layer(ep, vec.data(), /*ef=*/1, lc);
    
    // The result is a max-heap with just 1 entry (ef=1)
    // Get that entry as the new entry point for the next layer down
    ep = { result.top() };
    //     ^^^^^^^^^^^^^ — single closest node we found at layer lc
}

// After loop: ep contains the closest node at layer new_level+1
// (or if new_level == max_level_, ep still has the original entry point)
```

**Key point:** We're NOT building edges during descent. Just navigating to find the right starting point.

---

## Step G — Build Edges at Each Layer

**What it does:** For each layer from `min(new_level, max_level_)` down to 0, find the best candidates and connect them.

**The loop structure:**

```
We go from layer min(new_level, max_level_) down to 0.

Layer min(new, max):  search + select + connect
Layer ...:            search + select + connect
Layer 0:              search + select + connect (layer 0 allows 2*M neighbors)
```

**For each layer, do this:**

### 3a. Beam search

```cpp
int lc = ...;  // current layer

auto candidates = search_layer(ep, vec.data(), ef_construction_, lc);
// candidates = max-heap of up to ef_construction_ closest nodes at layer lc
```

### 3b. How many neighbors for this layer?

```cpp
std::size_t M_lc;
if (lc == 0) {
    M_lc = 2 * M_;  // layer 0 is denser
} else {
    M_lc = M_;      // all other layers
}
```

### 3c. Pick the best M_lc from candidates

```cpp
auto nbs = select_neighbors(candidates, M_lc);
// nbs = [id1, id2, ..., id_M_lc]   — the M_lc closest nodes
```

### 3d. Connect new_id → neighbors

```cpp
nodes_[new_id].neighbors[lc] = nbs;
// Now new_id points to the M_lc best nodes at layer lc
```

### 3e. Bidirectional edge — neighbors → new_id

```cpp
for (VectorId nb : nbs) {
    nodes_[nb].neighbors[lc].push_back(new_id);
    // Now nb also knows about new_id
}
```

### 3f. Shrink any neighbor that exceeds its limit

This is the confusing part. **When we add new_id to nb's neighbor list, nb might exceed M_lc neighbors.**

Example:
```
Layer 1, M=16. Node X has 16 neighbors. We add new_id.
Now X has 17 neighbors. But layer 1 max is 16.
We need to shrink X back to 16.
```

**How to shrink:**

```cpp
for (VectorId nb : nbs) {
    nodes_[nb].neighbors[lc].push_back(new_id);
    
    // Check if nb now exceeds its limit
    if (nodes_[nb].neighbors[lc].size() > M_lc) {
        // We need to drop one of nb's neighbors
        // But which one? The farthest from nb.
        
        // Rebuild a candidate heap from all of nb's current neighbors
        std::priority_queue<std::pair<float, VectorId>> nb_cands;
        for (VectorId existing : nodes_[nb].neighbors[lc]) {
            float d = node_dist(existing, store_.data() + nb * store_.dim());
            nb_cands.emplace(d, existing);
        }
        
        // Re-run select_neighbors to pick the best M_lc
        nodes_[nb].neighbors[lc] = select_neighbors(nb_cands, M_lc);
        // The farthest neighbor got dropped
    }
}
```

**In English:**
1. Add new_id to nb's neighbor list
2. If nb now has too many neighbors, rebuild the candidate heap (distances from nb to all its neighbors)
3. Re-run select_neighbors to keep only the M_lc closest

### 3g. Update entry point for the next layer

After we've built edges at layer lc, we use the neighbors we picked as the starting point for the next (lower) layer:

```cpp
ep.clear();
for (VectorId nb : nbs) {
    ep.emplace_back(node_dist(nb, vec.data()), nb);
}
// ep now = [closest neighbors we just picked]
// Next iteration will use these as starting points
```

**Full layer loop:**

```cpp
for (int lc = std::min(new_level, max_level_); lc >= 0; --lc) {
    // 3a. Beam search
    auto candidates = search_layer(ep, vec.data(), ef_construction_, lc);
    
    // 3b. Size
    std::size_t M_lc = (lc == 0) ? 2 * M_ : M_;
    
    // 3c. Pick
    auto nbs = select_neighbors(candidates, M_lc);
    
    // 3d. new_id → nbs
    nodes_[new_id].neighbors[lc] = nbs;
    
    // 3e + 3f. nbs → new_id (bidirectional + shrink)
    for (VectorId nb : nbs) {
        nodes_[nb].neighbors[lc].push_back(new_id);
        if (nodes_[nb].neighbors[lc].size() > M_lc) {
            // rebuild and shrink nb
            std::priority_queue<std::pair<float, VectorId>> nb_cands;
            for (VectorId existing : nodes_[nb].neighbors[lc]) {
                nb_cands.emplace(
                    node_dist(existing, store_.data() + nb * store_.dim()),
                    existing
                );
            }
            nodes_[nb].neighbors[lc] = select_neighbors(nb_cands, M_lc);
        }
    }
    
    // 3g. Update ep for next layer
    ep.clear();
    for (VectorId nb : nbs)
        ep.emplace_back(node_dist(nb, vec.data()), nb);
}
```

---

## Step H — Update Entry Point

If the new node has a higher level than any node in the graph, it becomes the new global entry point.

```cpp
if (new_level > max_level_) {
    entry_point_ = new_id;
    max_level_   = new_level;
}
```

That's it. If we didn't insert a new max-level node, entry_point_ stays the same.

---

## Visual Example

Insert vector V with assigned level 1 into a graph with max_level=2.

```
Before insert:

Layer 2:  A ──────────── E
Layer 1:  A ── C ─ D ── E
Layer 0:  A─B─C─D─E─F─G

After F (descend upper layer):
- entry_point_ = A at layer 2
- search_layer(A, 1 ef, layer 2) → found E
- ep = [E]

G, iteration 1 (lc=1):
- search_layer([E], ef_construction, layer 1) → found [E, D, C]
- M_lc = 16 (not layer 0)
- nbs = [E, D, C] (let's say select_neighbors picked these 3)
- nodes_[V].neighbors[1] = [E, D, C]
- nodes_[E].neighbors[1].push_back(V) → E now links to V
- nodes_[D].neighbors[1].push_back(V) → D now links to V
- nodes_[C].neighbors[1].push_back(V) → C now links to V
- (assume no shrinking needed)
- ep = [E, D, C]

G, iteration 2 (lc=0):
- search_layer([E, D, C], ef_construction, layer 0) → found [E, D, C, B]
- M_lc = 32 (layer 0 is 2*M)
- nbs = [E, D, C, B]
- nodes_[V].neighbors[0] = [E, D, C, B]
- nodes_[E].neighbors[0].push_back(V)
- nodes_[D].neighbors[0].push_back(V)
- nodes_[C].neighbors[0].push_back(V)
- nodes_[B].neighbors[0].push_back(V)
- (assume no shrinking needed)

After G: V is connected at layers 0 and 1.

H (new_level=1, max_level=2):
- 1 > 2? No. entry_point_ stays as A.
```

---

## Common Mistakes

**F:**
- Forgetting that the loop runs from `max_level_` down to `new_level+1`. If `new_level >= max_level_`, the loop is empty.
- Not using `ep.top()` to get the single result from ef=1 search.

**G:**
- Confusing which direction the loop goes. It goes DOWN: `min(new_level, max_level_)` → 0.
- Forgetting that layer 0 is special: 2*M instead of M.
- Not shrinking when neighbors exceed the limit — this breaks the invariant.
- Rebuilding `ep` for the next layer using the neighbors you just added, not the candidates heap.

**H:**
- Updating entry_point_ even when `new_level <= max_level_`. Only update if strictly greater.
