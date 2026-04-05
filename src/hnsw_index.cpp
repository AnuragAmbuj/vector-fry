#include "fry/hnsw_index.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace fry {
    // C++11 Min-Heap for Vector Id
    using MinHeap = std::priority_queue<
        std::pair<float, VectorId>,
        std::vector<std::pair<float, VectorId> >,
        std::greater<std::pair<float, VectorId> > >;

    // C++11 Max-Heap for Vector Id
    using MaxHeap = std::priority_queue<
        std::pair<float, VectorId> >;

    // Constructor
    // See hnsw_index.hpp for what each member does.
    // mL_ must be 1/log(M) — blow up early if M < 2 since log(1) == 0.
    template<Metric M>
    HNSWIndex<M>::HNSWIndex(std::size_t dim,
                            std::size_t max_links,
                            std::size_t ef_construction)
        : store_(dim)
          , nodes_()
          , M_(max_links)
          , ef_construction_(ef_construction)
          , mL_(1.0 / std::log(static_cast<double>(max_links)))
          , entry_point_(kInvalidId)
          , max_level_(-1)
          , rng_(std::random_device{}()) {
        if (max_links < 2) {
            throw std::invalid_argument("max_links must be greater than 2");
        }
    }


    // Returns dist(node `id`, query). Looks up the node's floats in the slab.
    // Vector id lives at: store_.data() + id * store_.dim()
    template<Metric M>
    auto HNSWIndex<M>::node_dist(VectorId id, const float *query) const -> float {
        return dispatch<M>(query, store_.data() + id * store_.dim(), store_.dim());
    }


    // Samples a random level using the exponential distribution from the paper.
    // formula: floor(-ln(U(0,1)) * mL_)
    // Most draws return 0. Roughly 1/M reach level 1, 1/M^2 reach level 2, etc.
    template<Metric M>
    auto HNSWIndex<M>::assign_level() const -> int {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return static_cast<int>(std::floor(-std::log(dist(rng_)) * mL_));
    }


    // Picks the `num_neighbors` closest from a max-heap of candidates.
    // The heap is consumed (passed by value). Returns only the IDs, not distances.
    //
    // How: drain into a vector (worst-first), then take the last `num_neighbors`
    // entries — those are the closest since the heap gave us worst at top.
    template<Metric M>
    auto HNSWIndex<M>::select_neighbors(
        std::priority_queue<std::pair<float, VectorId> > candidates,
        std::size_t num_neighbors
    ) -> std::vector<VectorId> {
        std::vector<std::pair<float, VectorId> > all;
        while (!candidates.empty()) {
            all.push_back(candidates.top());
            candidates.pop();
        }
        std::size_t const take = std::min(num_neighbors, all.size());
        std::vector<VectorId> result;
        for (std::size_t i = all.size() - take; i < all.size(); ++i) {
            result.push_back(all[i].second);
        }
        return result;
    }


    // Greedy beam search on one layer of the graph. Algorithm 2 from the paper.
    //
    // Keeps two structures in sync:
    //   found      — max-heap, bounded to `ef`. Worst match at top. This is returned.
    //   candidates — min-heap. Best unexplored node at top. We expand these.
    //   visited    — unordered_set so we never process the same node twice.
    //
    // Stop condition: the best remaining candidate is already worse than the worst
    // result in `found` (and found is full). Nothing we expand can improve things.


    // Insert. Algorithm 1 from the paper.
    //
    // A. Guard — dimension check.
    // B. Store the vector: auto id_result = store_.insert(vec); propagate on error.
    // C. Level: int new_level = assign_level();
    // D. Add node: nodes_.emplace_back(new_level);
    // E. First node — set entry_point_ and max_level_, return.
    // F. Descend upper layers (max_level_ → new_level+1) with ef=1 to get close entry.
    //      std::vector<std::pair<float,VectorId>> ep = {{ node_dist(entry_point_, vec.data()), entry_point_ }};
    //      for (int lc = max_level_; lc > new_level; --lc)
    //          ep = { search_layer(ep, vec.data(), 1, lc).top() };
    // G. For each layer min(new_level, max_level_) → 0:
    //      auto W   = search_layer(ep, vec.data(), ef_construction_, lc);
    //      M_lc     = (lc == 0) ? 2 * M_ : M_;
    //      auto nbs = select_neighbors(W, M_lc);
    //      nodes_[new_id].neighbors[lc] = nbs;
    //      for each nb in nbs:
    //          nodes_[nb].neighbors[lc].push_back(new_id);
    //          if (nodes_[nb].neighbors[lc].size() > M_lc) shrink it with select_neighbors.
    //      rebuild ep from nbs for the next (lower) layer.
    // H. If new_level > max_level_: update entry_point_ and max_level_.
    // I. return make_ok(new_id).
    //
    // Shrinking a neighbour (step G):
    //   build a candidate heap from the neighbour's current links, then
    //   nodes_[nb].neighbors[lc] = select_neighbors(that heap, M_lc);
    template<Metric M>
    auto HNSWIndex<M>::insert(const Vector<float> &vec) -> Result<VectorId> {
        // A — dimension guard
        // TODO

        // B — store
        // TODO: auto id_result = store_.insert(vec); if (!id_result) return id_result;
        //       VectorId new_id = *id_result;

        // C — level
        // TODO: int new_level = assign_level();

        // D — node
        // TODO: nodes_.emplace_back(new_level);

        // E — first node
        // TODO: if (entry_point_ == kInvalidId) { entry_point_ = new_id; max_level_ = new_level; return make_ok(new_id); }

        // F — descend upper layers
        // TODO

        // G — build edges per layer
        // TODO

        // H — update entry point
        // TODO

        // I — return
        (void) vec;
        return make_error<VectorId>(Error::invalid_argument("insert: not yet implemented"));
    }


    // Query. Algorithm 5 from the paper.
    //
    // A. Guards: StoreEmpty, InvalidArgument (k==0), DimensionMismatch.
    //    actual_k  = min(k, store_.size())
    //    actual_ef = max(ef_search, actual_k)   // ef must be >= k
    // B. Seed: ep = {{ node_dist(entry_point_, vec.data()), entry_point_ }}
    // C. Descend layers max_level_ → 1 with ef=1 each.
    // D. Beam search layer 0 with actual_ef.
    // E. Extract top actual_k from result heap, reverse for closest-first, return.
    template<Metric M>
    auto HNSWIndex<M>::query(const Vector<float> &vec,
                             std::size_t k,
                             std::size_t ef_search) const -> Result<std::vector<VectorId> > {
        // A — guards
        // TODO
        if (k == 0) {
            throw std::invalid_argument("query: k must be greater than zero");
        }

        auto actual_k = std::min(k,store_.size());
        auto actual_ef = std::max(ef_search, actual_k);

        if (actual_ef < actual_k) {
            throw std::invalid_argument("query: dimension mismatch");
        }

        // B — seed
        // TODO

        // C — descend
        // TODO

        // D — layer 0 search
        // TODO

        // E — extract + reverse
        // TODO

        (void) vec;
        (void) k;
        (void) ef_search;
        return make_error<std::vector<VectorId> >(
            Error::invalid_argument("query: not yet implemented"));
    }


    //
    // No structured bindings in C++11 — use .first / .second explicitly.
    //
    // Guard before accessing nodes_[c_id].neighbors[layer]:
    //   only do it if static_cast<int>(nodes_[c_id].neighbors.size()) > layer
    template<Metric M>
    std::priority_queue<std::pair<float, VectorId> > HNSWIndex<M>::search_layer(
        const std::vector<std::pair<float, VectorId> > &entry_points,
        const float *query,
        const std::size_t ef,
        int layer
    ) const {
        MinHeap candidates;
        MaxHeap found;
        std::unordered_set<VectorId> visited;

        for (const auto &ep: entry_points) {
            candidates.push(ep);
            found.push(ep);
            visited.insert(ep.second);
        }

        while (!candidates.empty()) {
            const float c_dist = candidates.top().first;
            VectorId const c_id = candidates.top().second;
            candidates.pop();

            if (c_dist > found.top().first && found.size() == ef) {
                break;
            }

            if (static_cast<int>(nodes_[c_id].neighbors.size()) <= layer) {
                for (VectorId const nb: nodes_[c_id].neighbors[layer]) {
                    if (visited.count(nb) == 0) {
                        visited.insert(nb);
                        float d = node_dist(nb, query);
                        if (d < found.top().first || found.size() < ef) {
                            candidates.push({d, nb});
                            found.push({d, nb});
                            if (found.size() > ef) found.pop();
                        }
                    }
                }
            }
        }

        return found;
    }


    // Explicit instantiations — needed because the implementation is in a .cpp.
    // Add a line here if you need a new metric.
    template class HNSWIndex<Metric::L2>;
    template class HNSWIndex<Metric::Cosine>;
    template class HNSWIndex<Metric::InnerProduct>;
} // namespace fry
