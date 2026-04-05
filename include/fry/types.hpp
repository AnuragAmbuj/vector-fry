#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fry {

// ── VectorId ─────────────────────────────────────────────────────────────────
//
// A named type alias for vector IDs.
//
// Why uint64_t and not size_t?
//   std::size_t is platform-defined (32-bit on 32-bit systems). uint64_t is
//   always 64-bit, giving us ~18 quintillion IDs regardless of platform.
//   It also signals intent: this is an identifier, not a size or an index.
//
using VectorId = std::uint64_t;

// Sentinel meaning "no valid ID" (analogous to std::string::npos).
// constexpr at namespace scope is valid in C++11 and has internal linkage,
// so including this header in many TUs does not cause ODR violations.
constexpr VectorId kInvalidId = std::numeric_limits<VectorId>::max();


// ── Vector<T> ─────────────────────────────────────────────────────────────────
//
// A fixed-dimension dense vector of element type T (almost always float).
//
// Design decisions:
//
//   1. Owns its data via std::vector<T>.
//      The alternative is a non-owning raw pointer + size, but then lifetime
//      management falls on every call site. Ownership here keeps things safe.
//
//   2. Dimension is checked at construction time.
//      A raw std::vector<float> carries no dimension invariant — mismatches
//      are only caught deep inside a distance loop. By encoding dim_ at
//      construction we move the error to the earliest possible moment.
//
//   3. Provides data() for zero-copy read access.
//      Callers (distance functions, SIMD kernels in Phase 3) get a const
//      raw pointer without any allocation or copy.
//      Note: std::span would be ideal here, but it's C++20. For C++11 we
//      expose data() + dim() as the two-part "view" contract.
//
//   4. Templated on T so the same class works for float, double, or uint8_t
//      (quantised vectors) without any code duplication.
//
template <typename T = float>
class Vector {
public:
    // Construct from an initialiser list — convenient in tests and call sites.
    // e.g.  Vector v{1.0f, 0.0f, 0.0f};
    Vector(std::initializer_list<T> elems)
        : data_(elems), dim_(data_.size()) {
        if (dim_ == 0)
            throw std::invalid_argument("fry::Vector: dimension must be > 0");
    }

    // Construct from a std::vector<T> (takes ownership via move).
    // explicit prevents accidental implicit conversions.
    explicit Vector(std::vector<T> data)
        : data_(std::move(data)), dim_(data_.size()) {
        // Note: member init list runs in declaration order.
        // data_ is declared before dim_, so data_ is populated first,
        // then dim_ = data_.size() reads the already-moved-in size. Correct.
        if (dim_ == 0)
            throw std::invalid_argument("fry::Vector: dimension must be > 0");
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    // Number of elements (the vector's dimension).
    std::size_t dim() const { return dim_; }

    // Raw const pointer to the first element. Used by distance kernels.
    // Callers must not mutate the data through this pointer — it's const.
    const T* data() const { return data_.data(); }

    // Element access (read-only).
    const T& operator[](std::size_t i) const { return data_[i]; }

    // ── Comparison ────────────────────────────────────────────────────────────

    // Two vectors are equal if they have identical data (same dim implied).
    // Useful in smoke tests: "insert then retrieve gives back the same data".
    bool operator==(const Vector& other) const { return data_ == other.data_; }
    bool operator!=(const Vector& other) const { return !(*this == other); }

private:
    std::vector<T> data_;  // Owned storage.
    std::size_t    dim_;   // Cached size (== data_.size() always).
};

} // namespace fry
