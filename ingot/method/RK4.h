#pragma once

#include "../ingot.h"

namespace ingot {
namespace method {

struct RK4 {
    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func const& f, const double t, const double h,
                                 Eigen::Array<T, N, 1> const& ue) const {

        using Arr = StackArray<T, N>;

        Arr u = ue.stackarray();

        const Arr k1 = f(t, u);
        const Arr k2 = f(t + h / 2, u + h * k1 / 2);
        const Arr k3 = f(t + h / 2, u + h * k2 / 2);
        const Arr k4 = f(t + h, u + h * k3);

        return u + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }
};

} // namespace method
} // namespace ingot
