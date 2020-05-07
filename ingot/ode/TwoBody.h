#pragma once

#include "../ingot.h"

namespace ingot {
namespace ode {

class TwoBody : public HostDevTimeInvariantODE<TwoBody> {

    using base_t = HostDevTimeInvariantODE<TwoBody>;

    double m = 1;

public:
    TwoBody(double m = 1) : m{m} {}

    using base_t::operator();

    template<typename T>
    CUDA_HOSTDEV constexpr void operator()(StackArray<T, 6>& up, // u-prime
                                           StackArray<T, 6> const& u) const {

        const auto rr = 1 / sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        const auto rrcube = rr * rr * rr;

        up[0] = u[3];
        up[1] = u[4];
        up[2] = u[5];

        up[3] = -u[0] * m * rrcube;
        up[4] = -u[1] * m * rrcube;
        up[5] = -u[2] * m * rrcube;
    }
};

} // namespace ode
} // namespace ingot
