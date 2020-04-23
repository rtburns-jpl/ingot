#pragma once

/*
 * Simple RK4 example for testing
 */
struct RK4 {
    template<typename Func, typename T, int N>
    __host__ __device__
    auto operator()(Func const& f, const double t, const double h,
                    StackArray<T, N> const& y) const {

        using Arr = StackArray<T, N>;

        const Arr k1 = f(t,       y);
        const Arr k2 = f(t + h/2, y + h*k1/2);
        const Arr k3 = f(t + h/2, y + h*k2/2);
        const Arr k4 = f(t + h,   y + h*k3);

        return y + h * (k1 + 2*k2 + 2*k3 + k4) / 6;
    }
};

// Embedded Fehlberg 7(8) integration step
//#define ERR_EXPONENT (1./7)
#define ERR_EXPONENT (1./8)

struct Frac {
    int numer;
    int denom;

    __host__ __device__
    constexpr operator double() const {
        return double(numer) / denom;
    }
};

template<typename ODEFunc, typename T, int N>
__host__ __device__
void method(const double t, const double h,
            StackArray<T, N> const& y,
            StackArray<T, N>& yp,
            StackArray<T, N>& err,
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

    auto yi = y;

    const auto k1 = ODE(t, yi);

    const auto k2 = ODE(t+ h2_7, k1*h2_7);

    const auto k3 = ODE(t+ a3*h, y + (b31*k1 + b32*k2)*h);

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

struct RKF78 {
    template<typename Func, typename T, int N>
    __host__ __device__
    auto operator()(Func const& f, double const t, double const h,
                    StackArray<T, N> const& y) const {

        StackArray<T, N> yp, err;

        method(t, h, y, yp, err, f);

        return yp;
    }
};

template<class ODE, typename T1, typename T2>
__device__
void doStep(
        // input/output integration parameters (time + state vector)
        const double t_in, const T1& sv,
        double& t_out, T2& sv_out,
        double& h) {
#define tol 1e-6

    Eigen::Matrix<double, ODE::N, 1> yp, err, y = sv;
    double t = t_in;
    //double h = h_in;

    //decltype(y) yp, err;

    method<ODE>(t, h, y, yp, err);

    // Generate ideal scaling factors for each component
    const double maxy   =   y.maxabs();
    const double maxerr = err.maxabs();

    // Calculate min scale factor, assuming worst case precision
    // so that none of the components lose accuracy
    const double tau = tol * max(maxy, 1.);

    // Only store the value if all results were within tolerance
    if (maxerr < tau) {
        // Update particle time/position
        y = yp;
        t += h;
    }

    // Update timestep adaptively:
    // Enforce scaling bounds
#define SCALE_MIN .2
#define SCALE_MAX 5.
    double scale = 0.8 * pow(tau / maxerr, ERR_EXPONENT);
    scale = min(max(scale, SCALE_MIN), SCALE_MAX);
    //h = idir * min(idir * h_max, idir * scale * h);
    //h = min(h_max, scale * h);
    h *= scale;

    sv_out = y;
    t_out  = t;
}

/*
template<class ODE>
void do_propagate(const ODEInput& oi) {

    assert(oi.v.size() == ODE::N);

    const size_t size = oi.t.size();
    for (int i = 0; i < oi.v.size(); i++)
        assert(oi.v[i].size() == size);

    // iteration vectors
    thrust::device_vector<double> t1 = oi.t;
    thrust::device_vector<double> t2(size); // time
    thrust::device_vector<double> dt(size);

    Matrix<ODE::N> y1(oi.v);
    Matrix<ODE::N> y2(size);

    thrust::fill(dt.begin(), dt.end(), std::numeric_limits<double>::epsilon());

    // output vectors
    thrust::device_vector<bool> do_output(size);
    thrust::device_vector<double> to(size);
    Matrix<ODE::N> yo(50 * size);

    auto outpos = zip_tuple_iters(to, yo);

    //while (oi.width() > 0 or not ip.isDone())
    for (int i = 0; i < 50; i++) {
        //printf("i = %d\n", i);

        // pass input/output to assign output by reference
        {
            auto iter = zip_tuple_iters(t1, y1, t2, y2, dt);
            thrust::for_each(iter, iter + size,
                             AdaptiveStepTransform<ODE>());
        }

        cudaErrorCheck(cudaDeviceSynchronize());

        // compare before/after for events;
        // output state vectors where events occurred
        {
            auto compare_iter = zip_tuple_iters(t1, y1, t2, y2, do_output);
            thrust::for_each(compare_iter, compare_iter + size,
                             CheckEventTransform<ODE>());

            auto  input = zip_tuple_iters(t1, y1);

            outpos = thrust::copy_if(input, input + size,
                                     do_output.begin(), // stencil
                                     outpos,
                                     is_true());        // predicate
        }

        cudaErrorCheck(cudaDeviceSynchronize());

        std::swap(t1, t2);
        std::swap(y1, y2);
    }
}
*/
