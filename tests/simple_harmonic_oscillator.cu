#include <doctest/doctest.h>
#include <ingot/ingot.h>
#include <ingot/integrate.h>
#include <ingot/integrator/adaptive.h>
#include <ingot/method/RKF78.h>

namespace ingot {
namespace ode {

class SHO : public HostDevTimeInvariantODE<SHO> {

    using base_t = HostDevTimeInvariantODE<SHO>;

    double k;

public:
    CUDA_HOSTDEV
    SHO(double k) : k{k} {}

    using base_t::operator();

    template<typename T>
    CUDA_HOSTDEV void operator()(Eigen::Array<T, 2, 1>& up, // u-prime
                                 Eigen::Array<T, 2, 1> const& u) const {

        up[0] = u[1];
        up[1] = -k * u[0];
    }
};

} // namespace ode
} // namespace ingot

using namespace ingot;

// Did the sign of the x-component change?
template<typename T, int N>
struct XVal {
    CUDA_HOSTDEV
    T operator()(double t, double h, ColVal<T, N> const& u) const {
        return u[0];
    }
};

CUDA_HOST static double rand_gen() { return 0.1 + double(rand()) / RAND_MAX / 2; }

TEST_CASE("SimpleHarmonicOscillator, EnsembleAdaptive") {

    /*
     * Allocate and initialize particles
     */
    int size = 4;
    Ensemble<double, 2> ensemble{size}; // best keep size a power of 2 for now
    thrust::fill(ensemble.t.begin(), ensemble.t.end(), 0);
    thrust::fill(ensemble.h.begin(), ensemble.h.end(), .1);
    {
        srand(392);
        thrust::host_vector<double> hv(2 * size);
        auto u0 = hv.begin();
        thrust::generate_n(u0, size, rand_gen);
        thrust::fill(u0 + size, hv.end(), 0);
        ensemble.u.data = hv;
    }

    double tmax = 10;
    auto i = integrator::make_adaptive(method::RKF78{}, 1e-8);
    i.h_max = M_PI / 2;

#if 1
    const auto sols = integrate_steps(i, ode::SHO{1}, ensemble, tmax / 0.1,
                                      XVal<double, 2>{});
    for (const auto& s : sols) {
        printf("%f: %f, %f\n", s.t, s.u[0], s.u[1]);
    }
#else
    const auto sols = integrate_dense(i, ode::SHO{1}, ensemble, tmax);
    int idx = 0;
    for (const auto& isols : sols) {
        for (const auto& s : isols) {
            printf("%d : %g: %g, %g\n", idx, s.t, s.u[0], s.u[1]);
        }
        idx++;
    }
#endif

    printf("\nNumber of solutions: %d\n", sols.size());
}

