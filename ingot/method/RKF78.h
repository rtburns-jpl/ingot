#pragma once

#include "../ingot.h"

namespace ingot {
namespace method {

#include "Integrator.h"

struct RKF78 {
    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func const& f, double const t, double const h,
                                 Eigen::Array<T, N, 1> const& u,
                                 Eigen::Array<T, N, 1>& err) const {

        Eigen::Array<T, N, 1> up;

        method(t, h, u, up, err, f);

        return up;
    }

    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func&& f, const double t, const double h,
                                 Eigen::Array<T, N, 1> const& u) const {
        Eigen::Array<T, N, 1> err;
        return (*this)(std::forward<Func>(f), t, h, u, err);
    }

    CUDA_HOSTDEV
    static constexpr auto error_exponent() { return Frac{1, 8}; }
};

} // namespace method
} // namespace ingot
