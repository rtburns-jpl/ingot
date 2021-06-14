#include <ingot/ingot.h>
#include <ingot/integrate.h>
#include <ingot/integrator/adaptive.h>
#include <ingot/method/RKF78.h>
#include <ingot/ode/CR3BP.h>
using namespace ingot;

// Did the sign of the y-component change?
template<typename T, int N>
struct YVal {
    CUDA_HOSTDEV
    T operator()(double t, double h, ColVal<T, N> const& u) const {
        return u[1];
    }
};

CUDA_HOST double rand_gen() { return 0.1 + double(rand()) / RAND_MAX / 2; }

int main() {
    /*
     * Allocate and initialize particles
     */
    int size = 256;
    Ensemble<double, 6> ensemble{size}; // best keep this a power of 2 for now
    thrust::fill(ensemble.t.begin(), ensemble.t.end(), 0);
    thrust::fill(ensemble.h.begin(), ensemble.h.end(), .1);
    {
        srand(392);
        thrust::host_vector<double> hv{6 * size};
        auto u0 = hv.begin();
        thrust::generate_n(u0, 3 * size, rand_gen);
        thrust::fill(u0 + 3 * size, hv.end(), 0);
        ensemble.u.data = hv;
    }

    /*
     * Integrate with output function for fixed number of steps
     */
    const auto i = integrator::make_adaptive(method::RKF78{}, 1e-8);
    const auto sols = integrate_steps(i, ode::CR3BP{0.04}, ensemble, 10000,
                                      YVal<double, 6>{});

    /*
     * Print output
     */
    for (const auto& s : sols) {
        printf("%g, %g, %g, %g, %g, %g\n",
               s.u[0], s.u[1], s.u[2],
               s.u[3], s.u[4], s.u[5]);
    }
}
