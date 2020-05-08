#pragma once

namespace ingot {
namespace integrator {

template<typename Method>
struct Fixed {

    Method method;
    double tf = std::numeric_limits<double>::max();

    template<typename ODE, typename T, int N>
    CUDA_DEV void operator()(ODE ode, double& t, double& h,
                             ColRef<T, N> y) const {
        if (t + h > tf) {
            h = tf - t;
        }
        if (h <= 0) {
            return;
        }

        Eigen::Array<T, N, 1> ytmp = y;
        y = method(ode, t, h, ytmp);
        t += h;
    }
};
template<typename Method>
auto make_fixed(Method m) {
    return Fixed<Method>{m};
}

} // namespace integrator
} // namespace ingot
