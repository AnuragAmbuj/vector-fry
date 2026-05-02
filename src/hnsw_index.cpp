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
    // For Cosine and InnerProduct (similarity metrics), negate so "smaller is better" for HNSW
    template<Metric M>
    auto HNSWIndex<M>::node_dist(VectorId id, const float *query) const -> float {
        float raw_distance = dispatch<M>(query, store_.data() + id * store_.dim(), store_.dim());
        return as_hnsw_distance<M>(raw_distance);
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
    ) const -> std::vector<VectorId> {
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
        // A — guard: dimension check
        if (vec.dim() != store_.dim()) {
            return make_error<VectorId>(
                Error::dimension_mismatch(vec.dim(), store_.dim()));
        }

        // B — store the vector
        auto id_result = store_.insert(vec);
        if (!id_result) {
            return id_result;
        }

        // C — assign level and create node
        VectorId const new_id = *id_result;
        int const new_level = assign_level();
        nodes_.emplace_back(new_level);

        // D — first node case: set as entry point and return
        if (entry_point_ == kInvalidId) {
            entry_point_ = new_id;
            max_level_ = new_level;
            return make_ok(new_id);
        }

        // F — descend upper layers
        std::vector<std::pair<float, VectorId>> ep = {
            { node_dist(entry_point_, vec.data()), entry_point_ }
        };
        for (int lc = max_level_; lc > new_level; --lc) {
            auto result = search_layer(ep, vec.data(), 1, lc);
            ep = { result.top() };
        }

        // G — build edges per layer
        for (int lc = std::min(new_level, max_level_); lc >= 0; --lc) {
            auto candidates = search_layer(ep, vec.data(), ef_construction_, lc);
            std::size_t const M_lc = (lc == 0) ? 2 * M_ : M_;
            auto nbs = select_neighbors(candidates, M_lc);
            nodes_[new_id].neighbors[static_cast<std::size_t>(lc)] = nbs;
            for (VectorId const nb : nbs) {
                nodes_[nb].neighbors[static_cast<std::size_t>(lc)].push_back(new_id);
                if (nodes_[nb].neighbors[static_cast<std::size_t>(lc)].size() > M_lc) {
                    MaxHeap nb_cands;
                    for (VectorId const existing : nodes_[nb].neighbors[static_cast<std::size_t>(lc)]) {
                        float const d = node_dist(existing, store_.data() + nb * store_.dim());
                        nb_cands.emplace(d, existing);
                    }
                    nodes_[nb].neighbors[static_cast<std::size_t>(lc)] = select_neighbors(nb_cands, M_lc);
                }
            }
            ep.clear();
            for (VectorId const nb : nbs) {
                ep.emplace_back(node_dist(nb, vec.data()), nb);
            }
        }

        // H — update entry point if new_level is higher
        if (new_level > max_level_) {
            entry_point_ = new_id;
            max_level_ = new_level;
        }

        // I — return the new node ID
        return make_ok(new_id);
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
        // Check 1: Store is not empty
        if (store_.size() == 0) {
            return make_error<std::vector<VectorId> >(
                Error::store_empty());
        }

        // Check 2: k is valid
        if (k == 0) {
            return make_error<std::vector<VectorId> >(
                Error::invalid_argument("query: k must be greater than zero"));
        }

        // Check 3: Dimension matches
        if (vec.dim() != store_.dim()) {
            return make_error<std::vector<VectorId> >(
                Error::dimension_mismatch(vec.dim(), store_.dim()));
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

        // E — extract ALL from max-heap (worst-first), reverse to get best-first, take top k
        std::vector<VectorId> results;
        while (!found.empty()) {
            results.push_back(found.top().second);
            found.pop();
        }
        // Now results is in worst-first order (largest distance first)
        // Reverse to get best-first order (smallest distance first)
        std::reverse(results.begin(), results.end());
        // Keep only the first k (the best k)
        if (results.size() > actual_k) {
            results.resize(actual_k);
        }

        return make_ok(results);
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

            if (static_cast<int>(nodes_[c_id].neighbors.size()) > layer) {
                for (VectorId const nb: nodes_[c_id].neighbors[static_cast<std::size_t>(layer)]) {
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
