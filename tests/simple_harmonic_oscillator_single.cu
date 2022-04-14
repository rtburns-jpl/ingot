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
    Ensemble<double, 2> ensemble{1};
    thrust::fill(ensemble.t.begin(), ensemble.t.end(), 0);
    thrust::fill(ensemble.h.begin(), ensemble.h.end(), .1);
    {
        thrust::host_vector<double> hv{2 * size};
        hv[0] = 1;
        hv[1] = 0;
        ensemble.u.data = hv;
    }

    double tmax = 10;
    auto i = integrator::make_adaptive(method::RKF78{}, 1e-8);
    i.h_max = M_PI / 2;

    const auto sols = integrate_steps(i, ode::SHO{1}, ensemble, tmax / 0.1,
                                      XVal<double, 2>{});

    double tol = 1e-6;

    double current_sol = M_PI / 2;
    for (const auto& s : sols) {

        // check current time
        CHECK(s.t - current_sol < tol);
        current_sol += M_PI;

        // check position
        CHECK(s.u[0] < tol);

        // check velocity
        CHECK(fabs(s.u[1]) - 1 < tol);
    }

    // make sure we actually checked some solutions :)
    CHECK(sols.size() > 10);
}
