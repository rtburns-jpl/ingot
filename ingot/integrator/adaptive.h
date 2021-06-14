#pragma once

#include "fixed.h"

namespace ingot {
namespace integrator {

template<typename Method>
struct Adaptive : public Fixed<Method> {

    double tol = 1e-12;
    double scale_min = 0;
    double scale_max = std::numeric_limits<double>::max();
    double h_max = std::numeric_limits<double>::max();

    using super_t = Fixed<Method>;
    using super_t::method;
    using super_t::tf;

    template<typename ODE, typename T, int N>
    CUDA_DEV void operator()(ODE ode, double& t, double& h,
                             ColRef<T, N> y) const {

        if (t + h > tf) {
            h = tf - t;
        }
        if (h <= 0) {
            return;
        }

        Eigen::Array<T, N, 1> ytmp, yerr;
        ytmp = y;
        ytmp = method(ode, t, h, ytmp, yerr);

        // Compute | Delta_0 / Delta_1 |
        // TODO tweak scaling of yerr for local+relative tolerance
        const double err_frac = (yerr / 1).abs().maxCoeff() / tol;

        if (err_frac < 1) {
            y = ytmp;
            t += h;
        }

        // "Safety factor" so that the following step is likely to succeed
        static constexpr const double S = 0.9;

        // Numerical Recipes 16.2.7
        h *= S * pow(err_frac, -Method::error_exponent());

        // scale = min(max(scale, scale_min), scale_max);
        // h = idir * min(idir * h_max, idir * scale * h);
        // h = min(h_max, scale * h);

        h = min(h, h_max);
    }
};
template<typename Method>
auto make_adaptive(Method m, double tol) {
    auto ret = Adaptive<Method>{};
    ret.method = m;
    ret.tol = tol;
    return ret;
}

} // namespace integrator
} // namespace ingot
