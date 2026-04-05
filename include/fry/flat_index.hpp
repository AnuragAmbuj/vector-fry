//
// Created by Anurag Ambuj on 26/03/26.
//

#ifndef FRY_VECTOR_FLAT_INDEX_HPP
#define FRY_VECTOR_FLAT_INDEX_HPP
#include <queue>
#include <utility>
#include <algorithm>

#include "distance.hpp"
#include "types.hpp"
#include "vector_store.h"
#include "result.hpp"

namespace fry {
    template<Metric M>
    class FlatIndex {
    private:
        VectorStore store_;

    public:
        explicit FlatIndex(const std::size_t dim) : store_(dim) {
        }

        auto insert(const Vector<> &vec) -> Result<VectorId> {
            return store_.insert(vec);
        }

        auto query(const Vector<float> &vec, std::size_t k) const -> Result<std::vector<VectorId> > {
            if (store_.size() == 0) {
                return make_error<std::vector<VectorId>>(Error::store_empty());
            }
            if (k == 0) {
                return make_error<std::vector<VectorId>>(Error::invalid_argument("k must be > 0"));
            }
            if (vec.dim() != dim()) {
                return make_error<std::vector<VectorId>>(Error::dimension_mismatch(vec.dim(), dim()));
            }

            const float *slab = store_.data(); // need to add data() to VectorStore
            const float *query_ptr = vec.data();
            const std::size_t dimension = store_.dim();
            const std::size_t actual_k = (k < store_.size()) ? k : store_.size();

            std::priority_queue<std::pair<float, VectorId> > heap;

            for (VectorId i = 0; i < store_.size(); ++i) {
                const float *vec_i = slab + (i * dimension);
                float dist = dispatch<M>(query_ptr, vec_i, dimension);
                float heap_val = (M == Metric::Cosine) ? -dist : dist;

                if (heap.size() < actual_k) {
                    heap.emplace(heap_val, i);
                } else if (heap_val < heap.top().first) {
                    heap.pop();
                    heap.emplace(heap_val, i);
                }
            }

            std::vector<VectorId> result;
            while (!heap.empty()) {
                result.push_back(heap.top().second);
                heap.pop();
            }
            std::reverse(result.begin(), result.end());
            return make_ok(result);
        }


        auto size() const -> std::size_t {
            return store_.size();
        }

        auto dim() const -> std::size_t {
            return store_.dim();
        }
    };
}

#endif //FRY_VECTOR_FLAT_INDEX_HPP
