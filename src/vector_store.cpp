#include "fry/vector_store.h"

#include <limits>
#include <stdexcept>

namespace fry {
    VectorStore::VectorStore(std::size_t dim)
        : dim_(dim), next_id_(0) {
        if (dim == 0) {
            throw std::invalid_argument("fry::VectorStore: dim must be > 0");
        }
    }

    auto VectorStore::insert(const Vector<float> &vec) -> Result<VectorId> {
        if (vec.dim() != dim_) {
            return make_error<VectorId>(Error::dimension_mismatch(vec.dim(), dim_));
        }
        data_.insert(data_.end(), vec.data(), vec.data() + dim_);
        return make_ok(next_id_++);
    }

    auto VectorStore::search(const Vector<float> &query) const -> Result<VectorId> {
        if (next_id_ == 0) {
            return make_error<VectorId>(Error::store_empty());
        }

        if (query.dim() != dim_) {
            return make_error<VectorId>(Error::dimension_mismatch(query.dim(), dim_));
        }

        VectorId best_id = 0;
        float best_dist = std::numeric_limits<float>::max();

        // Linear scan over the slab.
        //
        // Why not use iterators or range-based for here?
        //   We need pointer arithmetic to walk the slab by stride (dim_).
        //   An index loop is the clearest way to express that.
        //
        // In Phase 3 this inner loop becomes an AVX2 kernel — keeping it as a
        // plain loop now makes the before/after comparison educational.
        const float *query_ptr = query.data();

        for (VectorId i = 0; i < next_id_; ++i) {
            const float *base = data_.data() + (i * dim_);

            float dist = 0.0F;
            for (std::size_t d = 0; d < dim_; ++d) {
                const float diff = query_ptr[d] - base[d];
                dist += diff * diff; // Squared L2 — no sqrt needed for comparison.
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_id = i;
            }
        }

        return make_ok(best_id);
    }
} // namespace fry
