//
// Created by Anurag Ambuj on 24/03/26.
//

#ifndef FRY_VECTOR_DISTANCE_HPP
#define FRY_VECTOR_DISTANCE_HPP
#include <cstddef>
#include <cmath>

namespace fry {
    inline auto l2_squared(const float* a, const float* b, const std::size_t dim) -> float {
        float squared = 0.0f;
        for (std::size_t i = 0;i < dim;++i) {
            squared = squared + ((a[i]-b[i])*(a[i]-b[i]));
        }
        return squared;
    }

    inline auto norm (const float* arr, const std::size_t dim) -> float {
        float norm = 0.0f;
        for (std::size_t i = 0;i < dim;++i) {
            norm = norm + (arr[i]*arr[i]);
        }
        return std::sqrt(norm);
    }

    inline auto inner_product(const float* a, const float* b, const std::size_t dim) -> float {
        float dot = 0.0F;
        for (std::size_t i = 0;i < dim;++i) {
            dot = dot + (a[i]*b[i]);
        }
        return dot;
    }

    inline auto cosine_similarity(const float* a, const float* b, std::size_t dim) -> float {
       float const norm_product = norm(a, dim) * norm(b, dim);
        if (norm_product == 0.0F) {
            return 0.0F;
        }
       return inner_product(a, b, dim) / norm_product;
    }

    enum class Metric { L2, Cosine, InnerProduct };

    template<Metric M>
    auto dispatch(const float* a, const float* b, std::size_t dim) -> float = delete;

    template<>
    inline auto dispatch<Metric::L2>(const float* a, const float* b, std::size_t dim) -> float {
        return l2_squared(a, b, dim);
    }

    template<>
    inline auto dispatch<Metric::Cosine>(const float* a, const float* b, std::size_t dim) -> float {
        return cosine_similarity(a, b, dim);
    }

    template<>
    inline auto dispatch<Metric::InnerProduct>(const float* a, const float* b, std::size_t dim) -> float {
        return inner_product(a, b, dim);
    }

}
#endif //FRY_VECTOR_DISTANCE_HPP