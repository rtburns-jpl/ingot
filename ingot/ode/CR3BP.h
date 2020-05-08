#pragma once

#include "../ingot.h"

namespace ingot {
namespace ode {

class CR3BP : public HostDevTimeInvariantODE<CR3BP> {

    using base_t = HostDevTimeInvariantODE<CR3BP>;

    double mu;

public:
    CUDA_HOSTDEV
    constexpr CR3BP(double mu) : mu{mu} {}

    using base_t::operator();

    template<typename T>
    CUDA_HOSTDEV constexpr void operator()(Eigen::Array<T, 6, 1>& up, // u-prime
                                           Eigen::Array<T, 6, 1> const& u) const {

        const double m1 = 1 - mu;
        const double m2 = -mu;
        const double r1 = -mu;
        const double r2 = 1 - mu;

        const auto dx1 = u[0] - r1;
        const auto dx2 = u[0] - r2;

        const auto r1r = 1 / sqrt(dx1 * dx1 + u[1] * u[1] + u[2] * u[2]);
        const auto r2r = 1 / sqrt(dx2 * dx2 + u[1] * u[1] + u[2] * u[2]);

        const auto r1rcube = r1r * r1r * r1r;
        const auto r2rcube = r2r * r2r * r2r;

        up[0] = u[3];
        up[1] = u[4];
        up[2] = u[5];
        up[3] = -m1 * dx1 * r1rcube - m2 * dx2 * r2rcube + 2 * u[4] + u[0];
        up[4] = u[1] * (m1 * r1rcube + m2 * r2rcube - 1) - 2 * u[3];
        up[5] = u[2] * (m1 * r1rcube + m2 * r2rcube); // no centrip force in z
    }
};

} // namespace ode
} // namespace ingot
