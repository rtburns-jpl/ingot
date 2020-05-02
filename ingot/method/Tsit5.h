#pragma once

#include "../ingot.h"

namespace ingot {
namespace method {

struct Tsit5 {
    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func const& f, const double t, const double h,
                                 StackArray<T, N> const& y,
                                 Eigen::Array<T, N, 1>& yerr
                                 ) const {

        using Arr = StackArray<T, N>;
        using Coeff = T; // TODO use scalar type for complex

        // https://en.wikipedia.org/wiki/Dormand–Prince_method

        static constexpr Coeff a21 = 0.161;

        static constexpr Coeff a31 = -0.008480655492357;
        static constexpr Coeff a32 = 0.3354806554923570;

        static constexpr Coeff a41 = 2.897153057105494;
        static constexpr Coeff a42 = -6.359448489975075;
        static constexpr Coeff a43 = 4.362295432869581;

        static constexpr Coeff a51 = 5.32586482843925895;
        static constexpr Coeff a52 = -11.74888356406283;
        static constexpr Coeff a53 = 7.495539342889836;
        static constexpr Coeff a54 = -0.09249506636175525;

        static constexpr Coeff a61 = 5.86145544294642038;
        static constexpr Coeff a62 = -12.92096931784711;
        static constexpr Coeff a63 = 8.159367898576159;
        static constexpr Coeff a64 = -0.071584973281401006;
        static constexpr Coeff a65 = -0.02826905039406838;

        static constexpr Coeff a71 = 0.09646076681806523;
        static constexpr Coeff a72 = 0.01;
        static constexpr Coeff a73 = 0.4798896504144996;
        static constexpr Coeff a74 = 1.379008574103742;
        static constexpr Coeff a75 = -3.290069515436081;
        static constexpr Coeff a76 = 2.324710524099774;

        //static constexpr Coeff c1 = 0;
        static constexpr Coeff c2 = 0.161;
        static constexpr Coeff c3 = 0.327;
        static constexpr Coeff c4 = 0.9;
        static constexpr Coeff c5 = 0.9800255409045097;
        //static constexpr Coeff c6 = 1;
        //static constexpr Coeff c7 = 1;

        static constexpr Coeff e1 = 0.09646076681806523 - 0.001780011052226;
        static constexpr Coeff e2 = 0.01 - 0.000816434459657;
        static constexpr Coeff e3 = 0.4798896504144996 - -0.007880878010262;
        static constexpr Coeff e4 = 1.379008574103742 - 0.144711007173263;
        static constexpr Coeff e5 = -3.290069515436081 - -0.582357165452555;
        static constexpr Coeff e6 = 2.324710524099774 - 0.458082105929187;
        static constexpr Coeff e7 = -1 / 66;

        const Arr k1 = f(t, y);
        const Arr k2 = f(t + c2 * h, y + a21 * k1 * h);
        const Arr k3 = f(t + c3 * h, y + (a31 * k1 + a32 * k2) * h);
        const Arr k4 = f(t + c4 * h, y + (a41 * k1 + a42 * k2 + a43 * k3) * h);
        const Arr k5 = f(t + c5 * h, y + (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4) * h);
        const Arr k6 = f(t + h, y + (a61 * k1 + a62 * k2 + a63 * k3 +
                                     a64 * k4 + a65 * k5) * h);

        const Arr b1 = y + (a71 * k1 + a72 * k2 + a73 * k3 +
                            a74 * k4 + a75 * k5 + a76 * k6) * h;

        const Arr k7 = f(t + h, b1);

        yerr = (e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4 +
                e5 * k5 + e6 * k6 + e7 * k7) * h;

        return b1;
    }

    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func&& f, const double t, const double h,
                                 StackArray<T, N> const& y) const {
        Eigen::Array<T, N, 1> err;
        return (*this)(std::forward<Func>(f), t, h, y, err);
    }

    CUDA_HOSTDEV
    static constexpr auto error_exponent() { return Frac{1, 5}; }
};

} // namespace method
} // namespace ingot
