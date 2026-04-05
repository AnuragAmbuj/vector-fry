#pragma once

// ── hnsw_index.hpp ────────────────────────────────────────────────────────────
//
// Hierarchical Navigable Small World (HNSW) approximate nearest-neighbour index.
//
// Reference: Malkov & Yashunin, 2018 — "Efficient and robust approximate
// nearest neighbor search using Hierarchical Navigable Small World graphs"
// https://arxiv.org/abs/1603.09320
//
// ── Algorithm overview ────────────────────────────────────────────────────────
//
// HNSW builds a layered graph. Layer 0 contains ALL nodes (dense). Each higher
// layer contains a random subset (sparser). Search starts at the top layer
// (few nodes, long jumps) and descends to layer 0 (precise, fine-grained).
//
//   Layer 2:  A ─────────── E             (sparse, long-range "express" links)
//   Layer 1:  A ── C ─ D ── E
//   Layer 0:  A─B─C─D─E─F─G─H─I─J        (all nodes, short-range links)
//
// ── Key parameters ────────────────────────────────────────────────────────────
//
//   M               — max number of bidirectional links per node per layer.
//                     Layer 0 allows 2*M links (denser bottom layer).
//                     Typical: 16. Higher M → better recall, more memory.
//
//   ef_construction — beam width during graph construction. The number of
//                     candidates kept during greedy search when inserting.
//                     Typical: 200. Higher → better graph quality, slower build.
//
//   ef_search       — beam width during query. The number of candidates kept
//                     during the layer-0 beam search.
//                     Typical: 50. Higher → better recall, slower query.
//
//   mL              — level normalisation factor = 1.0 / ln(M).
//                     Controls the probability distribution for level assignment.
//                     Most nodes land at level 0; very few reach level 2+.
//
// ── Level assignment ──────────────────────────────────────────────────────────
//
//   l = floor(-ln(uniform(0, 1)) * mL)
//
//   This is an exponential distribution. mL = 1/ln(M) is chosen so that the
//   expected number of nodes at each layer decreases geometrically by factor M.
//   With M=16: ~94% of nodes are at layer 0, ~6% at layer 1, ~0.4% at layer 2.
//
// ── Insert algorithm (Algorithm 1 from the paper) ────────────────────────────
//
//   1. Assign level l to the new node (exponential distribution).
//   2. If the index is empty: store node, set as entry point, done.
//   3. Starting from the current global entry point at max_level_:
//      a. For each layer from max_level_ down to l+1:
//         Greedy search (ef=1): find the single nearest neighbour.
//         This descends quickly through the upper sparse layers.
//      b. For each layer from min(l, max_level_) down to 0:
//         Greedy search (ef=ef_construction): find ef_construction candidates.
//         SELECT-NEIGHBORS: pick best M (or 2*M at layer 0) from candidates.
//         Add bidirectional edges. Shrink any neighbour list that exceeds M.
//   4. If l > max_level_: update entry point to new node, update max_level_.
//
// ── Query algorithm (Algorithm 5 from the paper) ─────────────────────────────
//
//   1. Start from the global entry point at max_level_.
//   2. For each layer from max_level_ down to 1:
//      Greedy search (ef=1): find the single nearest neighbour.
//   3. At layer 0: greedy search with ef=ef_search candidates.
//   4. Return top-K from the candidates.
//
// ── Greedy search (Algorithm 2 from the paper) ───────────────────────────────
//
//   Input:  entry candidates W (set of <dist,id> pairs), query q, layer lc, ef
//   Output: ef nearest neighbours found
//
//   candidates  = min-heap (best at top, for expansion)
//   found       = max-heap of size ef (worst at top, for eviction)
//   visited     = std::unordered_set<VectorId> (avoid revisiting nodes)
//
//   while candidates is not empty:
//       c = pop best from candidates            (closest unexplored node)
//       f = worst in found                      (farthest in current result set)
//       if dist(c, q) > dist(f, q): break       (can't improve anymore)
//
//       for each neighbour e of c at layer lc:
//           if e not in visited:
//               visited.insert(e)
//               d = dist(e, q)
//               if d < dist(f, q) OR found.size() < ef:
//                   candidates.push(e)
//                   found.push(e)
//                   if found.size() > ef: found.pop()  (evict worst)
//
//   return found
//
// ── SELECT-NEIGHBORS (simple version, Algorithm 3) ───────────────────────────
//
//   Given a set of candidates and a target count M:
//   Simply return the M closest candidates by distance.
//   (The heuristic version, Algorithm 4, additionally favours diversity —
//    implement that in a later optimisation pass.)
//
// ── Data structures ───────────────────────────────────────────────────────────
//
//   HNSWNode:
//     - neighbors: vector<vector<VectorId>>
//                  neighbors[layer] = list of neighbour IDs at that layer
//                  The outer vector has size (level + 1).
//
//   Vectors are stored in a VectorStore slab (same as FlatIndex).
//   node ID == VectorStore ID == index into nodes_ array.
//
// ── C++11 notes ───────────────────────────────────────────────────────────────
//
//   std::mt19937       — Mersenne Twister RNG (seeded with std::random_device)
//   std::uniform_real_distribution<double>  — generates U(0,1) for level assignment
//   std::priority_queue — used for both min-heap (negate scores) and max-heap
//   std::unordered_set — O(1) average visited-node tracking

#include "fry/distance.hpp"
#include "fry/result.hpp"
#include "fry/types.hpp"
#include "fry/vector_store.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

namespace fry {

// ── HNSWNode ──────────────────────────────────────────────────────────────────
//
// Stores the adjacency lists for one node across all its layers.
// `neighbors[lc]` is the list of neighbour IDs at layer lc.
// The size of `neighbors` equals (node_level + 1).
//
struct HNSWNode {
    // neighbors[layer] → list of neighbour VectorIds at that layer.
    // Layer 0 allows up to 2*M neighbours; higher layers allow up to M.
    std::vector<std::vector<VectorId>> neighbors;

    // Construct a node that participates in (level + 1) layers.
    // e.g. level=0 → neighbors has 1 entry (only layer 0).
    //      level=2 → neighbors has 3 entries (layers 0, 1, 2).
    explicit HNSWNode(int level)
        : neighbors(static_cast<std::size_t>(level + 1)) {}
};


// ── HNSWIndex<M> ─────────────────────────────────────────────────────────────
//
// Template parameter M selects the distance metric at compile time.
// All distance calls go through dispatch<M> — zero-cost, no virtual dispatch.
//
template<Metric M>
class HNSWIndex {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // dim            — dimension of every vector inserted into this index.
    // max_links      — max bidirectional links per node per layer (paper's M).
    //                  Layer 0 uses 2 * max_links. Default: 16.
    // ef_construction— beam width during build. Higher = better graph quality.
    //                  Default: 200.
    explicit HNSWIndex(std::size_t dim,
                       std::size_t max_links       = 16,
                       std::size_t ef_construction = 200);

    // ── Mutation ──────────────────────────────────────────────────────────────

    // Insert a vector into the index. Returns the assigned VectorId.
    // Error::DimensionMismatch if vec.dim() != this->dim().
    static auto insert(const Vector<float>& vec) -> Result<VectorId>;

    // ── Query ─────────────────────────────────────────────────────────────────

    // Find the k approximate nearest neighbours to vec.
    // ef_search — beam width for the layer-0 search. Must be >= k.
    //             Higher ef_search → better recall, slower query.
    // Returns VectorIds ordered closest-first.
    // Error::StoreEmpty       if no vectors have been inserted.
    // Error::DimensionMismatch if vec.dim() != this->dim().
    // Error::InvalidArgument  if k == 0.
    auto query(const Vector<float>& vec,
               std::size_t k,
               std::size_t ef_search = 50) const -> Result<std::vector<VectorId>>;

    // ── Accessors ─────────────────────────────────────────────────────────────

    auto size() const -> std::size_t { return store_.size(); }
    auto dim()  const -> std::size_t { return store_.dim(); }

private:
    // ── State ─────────────────────────────────────────────────────────────────

    VectorStore store_;           // flat float slab — same as FlatIndex
    std::vector<HNSWNode> nodes_; // one per inserted vector; nodes_[id].neighbors

    std::size_t M_;               // max links per layer (layer 0 gets 2*M_)
    std::size_t ef_construction_; // beam width during insert
    double      mL_;              // level normalisation = 1.0 / ln(M_)

    VectorId    entry_point_;     // ID of the current top-level entry node
    int         max_level_;       // highest layer currently in the graph

    mutable std::mt19937 rng_;    // RNG for level assignment (mutable: used in insert)

    // ── Private helpers ───────────────────────────────────────────────────────

    // Returns the distance between node `id` and raw pointer `query`.
    // Looks up the node's vector in the store slab and calls dispatch<M>.
    auto node_dist(VectorId id, const float* query) const -> float;

    // Assign a random level to a new node using the exponential distribution:
    //   level = floor(-ln(U(0,1)) * mL_)
    // Returns an int >= 0. Most results are 0; rare results reach 2, 3, ...
    auto assign_level() const -> int;

    // Greedy beam search at layer `layer`.
    //
    // entry_points — initial set of candidates (dist, id) to start from.
    // query        — raw float pointer for the query vector.
    // ef           — maximum number of candidates to keep.
    //
    // Returns up to `ef` nearest neighbours found, as a max-heap
    // (pair<dist, id>) so the worst is at .top().
    //
    // Algorithm: keep two structures in sync:
    //   candidates — min-heap (best unexplored node at top)
    //   found      — max-heap of size ef (worst result at top, for eviction)
    // Expand the best candidate; for each unvisited neighbour, compute distance.
    // If the neighbour is better than the worst in found (or found isn't full),
    // add it to both. Stop when the best candidate is worse than found.top().
    auto search_layer(
        const std::vector<std::pair<float, VectorId>>& entry_points,
        const float* query,
        std::size_t ef,
        int layer
    ) const -> std::priority_queue<std::pair<float, VectorId>>;  // max-heap

    // SELECT-NEIGHBORS (simple version).
    //
    // Given a max-heap of candidates (size >= M), extract the M closest.
    // The heap is consumed (passed by value intentionally).
    // Returns a vector of the M nearest VectorIds.
    static auto select_neighbors(
        std::priority_queue<std::pair<float, VectorId>> candidates,
        std::size_t num_neighbors
    ) -> std::vector<VectorId>;
};

} // namespace fry
