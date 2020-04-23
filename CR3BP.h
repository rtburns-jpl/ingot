#pragma once

#include "StackArray.h"

class CR3BP {

    double mu;

public:
    template<typename T>
    __host__ __device__
    constexpr auto operator()(
            StackArray<T, 6>& yp,
            StackArray<T, 6> const& y) const {

        const double m1 = 1-mu;
        const double m2 =  -mu;
        const double r1 =  -mu;
        const double r2 = 1-mu;

        const auto dx1 = y[0] - r1;
        const auto dx2 = y[0] - r2;

        const auto r1r = 1/sqrt(dx1*dx1 + y[1]*y[1] + y[2]*y[2]);
        const auto r2r = 1/sqrt(dx2*dx2 + y[1]*y[1] + y[2]*y[2]);

        const auto r1rcube = r1r*r1r*r1r;
        const auto r2rcube = r2r*r2r*r2r;

        yp[0] = y[3];
        yp[1] = y[4];
        yp[2] = y[5];
        yp[3] = -m1*dx1*r1rcube - m2*dx2*r2rcube + 2 * y[4] + y[0];
        yp[4] = y[1]*(m1*r1rcube + m2*r2rcube - 1) - 2*y[3];
        yp[5] = y[2]*(m1*r1rcube + m2*r2rcube); // no centrip force in z
    }

    __host__ __device__
    constexpr CR3BP(double mu) : mu{mu} {}

    template<typename T>
    __host__ __device__
    constexpr auto operator()(StackArray<T, 6> const& y) const {
        StackArray<T, 6> yp = y;
        (*this)(yp, y);
        return yp;
    }

    template<typename T>
    __host__ __device__
    constexpr auto operator()(const double t, StackArray<T, 6> const& y) const {
        return (*this)(y);
    }
};