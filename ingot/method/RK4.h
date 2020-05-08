#pragma once

#include "../ingot.h"

namespace ingot {
namespace method {

struct RK4 {
    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func const& f, const double t, const double h,
                                 Eigen::Array<T, N, 1> const& u) const {

        using Arr = Eigen::Array<T, N, 1>;

        const Arr k1 = f(t, u);
        const Arr k2 = f(t + h / 2, (u + h * k1 / 2).eval());
        const Arr k3 = f(t + h / 2, (u + h * k2 / 2).eval());
        const Arr k4 = f(t + h, (u + h * k3).eval());

        Eigen::Array<T, N, 1> ret = u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        return ret;
    }
};

} // namespace method
} // namespace ingot
