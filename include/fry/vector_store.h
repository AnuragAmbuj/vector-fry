#pragma once

#include "fry/result.hpp"
#include "fry/types.hpp"

#include <cstddef>
#include <vector>

namespace fry {


// A flat, contiguous in-memory store for fixed-dimension float vectors.
//
// Storage layout:
//   data_  is a single std::vector<float> of length (count * dim_).
//   Vector i occupies positions [i*dim_, (i+1)*dim_).
//   This "slab" layout is cache-friendly: sequential search reads one cache
//   line per 16 floats (64 bytes / 4 bytes), with no pointer chasing.
//
// ID assignment:
//   IDs are assigned sequentially starting from 0. After insertion, the
//   ID is stable for the lifetime of the store.
//
// Error handling:
//   Functions that can fail return Result<T> instead of throwing.
//   The constructor still throws for programmer errors (zero dimension) since
//   those represent a broken program, not a recoverable runtime condition.
//
class VectorStore {
public:
    // Construct a store for vectors of the given dimension.
    // Throws std::invalid_argument if dim == 0 — programmer error, not
    // recoverable, so an exception is appropriate here.
    explicit VectorStore(std::size_t dim);

    // ── Mutation ──────────────────────────────────────────────────────────────

    // Insert a vector; returns the assigned VectorId on success.
    // Returns Error::DimensionMismatch if vec.dim() != this->dim().
    //
    // Why Result<VectorId> instead of throwing?
    //   In Phase 6 this function will be called from an HTTP handler processing
    //   user-supplied data. We never want user input to trigger an exception —
    //   catching exceptions across async/thread boundaries is fragile.
    //   Result forces the call site to handle the error explicitly.
    auto insert(const Vector<float>& vec) -> Result<VectorId>;

    // ── Query ─────────────────────────────────────────────────────────────────

    // Brute-force nearest-neighbour search (L2 distance).
    // Returns the VectorId of the closest stored vector.
    // Returns Error::StoreEmpty if no vectors have been inserted.
    // Returns Error::DimensionMismatch if query.dim() != this->dim().
    auto search(const Vector<float>& query) const -> Result<VectorId>;

    // ── Accessors ─────────────────────────────────────────────────────────────

    auto size() const -> std::size_t { return next_id_; }
    auto dim()  const -> std::size_t { return dim_; }
    auto data() const -> const float* { return  data_.data(); }

private:
    std::size_t        dim_;      // Expected dimension of every stored vector.
    std::vector<float> data_;     // Flat slab: stride = dim_.
    VectorId           next_id_;  // Monotonically increasing ID counter.
};

} // namespace fry
