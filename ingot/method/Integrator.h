// Embedded Fehlberg 7(8) integration step

template<typename ODEFunc, typename T, int N>
CUDA_HOSTDEV
void method(const double t, const double h,
            Eigen::Array<T, N, 1> const& y,
            Eigen::Array<T, N, 1>& yp,
            Eigen::Array<T, N, 1>& err,
            ODEFunc const& ODE) {

    static const double c_12_13 = 41.0 / 840.0,
                        c6 = 34.0 / 105.0,
                        c_7_8= 9.0 / 35.0,
                        c_9_10 = 9.0 / 280.0,

                        a2 = 2.0 / 27.0,
                        a3 = 1.0 / 9.0,
                        a4 = 1.0 / 6.0,
                        a5 = 5.0 / 12.0,
                        a6 = 1.0 / 2.0,
                        a7 = 5.0 / 6.0,
                        a8 = 1.0 / 6.0,
                        a9 = 2.0 / 3.0,
                        a10 = 1.0 / 3.0,

                        b31 = 1.0 / 36.0,
                        b32 = 3.0 / 36.0,
                        b41 = 1.0 / 24.0,
                        b43 = 3.0 / 24.0,
                        b51 = 20.0 / 48.0,
                        b53 = -75.0 / 48.0,
                        b54 = 75.0 / 48.0,
                        b61 = 1.0 / 20.0,
                        b64 = 5.0 / 20.0,
                        b65 = 4.0 / 20.0,
                        b71 = -25.0 / 108.0,
                        b74 =  125.0 / 108.0,
                        b75 = -260.0 / 108.0,
                        b76 =  250.0 / 108.0,
                        b81 = 31.0/300.0,
                        b85 = 61.0/225.0,
                        b86 = -2.0/9.0,
                        b87 = 13.0/900.0,
                        b91 = 2.0,
                        b94 = -53.0/6.0,
                        b95 = 704.0 / 45.0,
                        b96 = -107.0 / 9.0,
                        b97 = 67.0 / 90.0,
                        b98 = 3.0,
                        b10_1 = -91.0 / 108.0,
                        b10_4 = 23.0 / 108.0,
                        b10_5 = -976.0 / 135.0,
                        b10_6 = 311.0 / 54.0,
                        b10_7 = -19.0 / 60.0,
                        b10_8 = 17.0 / 6.0,
                        b10_9 = -1.0 / 12.0,
                        b11_1 = 2383.0 / 4100.0,
                        b11_4 = -341.0 / 164.0,
                        b11_5 = 4496.0 / 1025.0,
                        b11_6 = -301.0 / 82.0,
                        b11_7 = 2133.0 / 4100.0,
                        b11_8 = 45.0 / 82.0,
                        b11_9 = 45.0 / 164.0,
                        b11_10 = 18.0 / 41.0,
                        b12_1 = 3.0 / 205.0,
                        b12_6 = - 6.0 / 41.0,
                        b12_7 = - 3.0 / 205.0,
                        b12_8 = - 3.0 / 41.0,
                        b12_9 = 3.0 / 41.0,
                        b12_10 = 6.0 / 41.0,
                        b13_1 = -1777.0 / 4100.0,
                        b13_4 = -341.0 / 164.0,
                        b13_5 = 4496.0 / 1025.0,
                        b13_6 = -289.0 / 82.0,
                        b13_7 = 2193.0 / 4100.0,
                        b13_8 = 51.0 / 82.0,
                        b13_9 = 33.0 / 164.0,
                        b13_10 = 12.0 / 41.0,

                        err_factor  = -41.0 / 840.0;

    const double h2_7 = a2 * h;

    Eigen::Array<T, N, 1> yi = y;

    const auto k1 = ODE(t, yi);

    const auto k2 = ODE(t+ h2_7, (y + k1*h2_7).eval());

    const auto k3 = ODE(t+ a3*h, (y + (b31*k1 + b32*k2)*h).eval());

    yi = y + (b41*k1 + b43*k3)*h;
    const auto k4 = ODE(t+ a4*h, yi);

    yi = y + (b51*k1 + b53*k3 + b54*k4)*h;
    const auto k5 = ODE(t+ a5*h, yi);

    yi = y + (b61*k1 + b64*k4 + b65*k5)*h;
    const auto k6 = ODE(t+ a6*h, yi);

    yi = y + (b71*k1 + b74*k4 + b75*k5 + b76*k6)*h;
    const auto k7 = ODE(t+ a7*h, yi);

    yi = y + (b81*k1 + b85*k5 + b86*k6 + b87*k7)*h;
    const auto k8 = ODE(t+ a8*h, yi);

    yi = y + (b91*k1 + b94*k4 + b95*k5 + b96*k6 + b97*k7 + b98*k8)*h;
    const auto k9 = ODE(t+ a9*h, yi);

    yi = y + (b10_1*k1 + b10_4*k4 + b10_5*k5
             + b10_6*k6 + b10_7*k7 + b10_8*k8 + b10_9*k9)*h;
    const auto k10 = ODE(t+a10*h, yi);

    yi = y + (b11_1*k1 + b11_4*k4 + b11_5*k5 + b11_6*k6 + b11_7*k7
             + b11_8*k8 + b11_9*k9 + b11_10*k10)*h;
    const auto k11 = ODE(t+h, yi);

    yi = y + (b12_1*k1 + b12_6*k6 + b12_7*k7
            + b12_8*k8 + b12_9*k9 + b12_10*k10)*h;
    const auto k12 = ODE(t, yi);

    yi = y + (b13_1*k1 + b13_4*k4 + b13_5*k5 + b13_6 *k6
            + b13_7*k7 + b13_8*k8 + b13_9*k9 + b13_10*k10 + k12)*h;
    const auto k13 = ODE(t+h, yi);

    // compute final weighted sum

    // 7th order update
    // static const double c_1_11 = 41.0 / 840.0;
    // yp = y + (c_1_11*(k1 + k11) + c6*k6
    //         + c_7_8 *(k7 + k8) + c_9_10*(k9 + k10))*h;

    // 8th order update
    yp = y + (c_12_13*(k12+k13) + c6*k6 + c_7_8*(k7+k8) + c_9_10*(k9+k10))*h;

    // compute error value
    err = -h * err_factor*(k1 + k11 - k12 - k13);
}
