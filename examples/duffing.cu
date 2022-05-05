#include <ingot/ingot.h>
#include <ingot/integrate.h>
#include <ingot/integrator/adaptive.h>
#include <ingot/method/RKF78.h>

namespace ingot {
namespace ode {

// x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t)

class Duffing {

public:
    double delta;
    double alpha;
    double beta;
    double gamma;
    double omega;

    template<typename T>
    CUDA_HOSTDEV constexpr auto operator()(const double t, Eigen::Array<T, 2, 1> const& u) const {

        Eigen::Array<T, 2, 1> up; // u-prime

        const double x = u[0];

        up[0] = u[1];
        up[1] = gamma * cos(omega * t) - delta * u[1] - alpha * x - beta * x*x*x;

        return up;
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

int main() {

    /*
     * Allocate and initialize particles
     */
    int size = 4;
    Ensemble<double, 2> ensemble{size};
    thrust::fill(ensemble.t.begin(), ensemble.t.end(), 0);
    thrust::fill(ensemble.h.begin(), ensemble.h.end(), .1);
    {
        srand(392);
        thrust::host_vector<double> hv{2 * size};
        auto u0 = hv.begin();
        thrust::generate_n(u0, size, rand_gen);
        thrust::fill(u0 + size, hv.end(), 0);
        ensemble.u.data = hv;
    }

    double tmax = 10;
    auto i = integrator::make_adaptive(method::RKF78{}, 1e-8);
    i.h_max = M_PI / 2;

    const double delta = 0.02;
    const double alpha = 1;
    const double beta = 5;
    const double gamma = 8;
    const double omega = 0.5;

    auto duffing_ode = ode::Duffing{delta, alpha, beta, gamma, omega};

    const auto sols = integrate_dense(i, duffing_ode, ensemble, tmax);
    int idx = 0;
    for (const auto& isols : sols) {
        for (const auto& s : isols) {
            printf("%d : %g: %g, %g\n", idx, s.t, s.u[0], s.u[1]);
        }
        idx++;
    }

    printf("\nNumber of solutions: %d\n", sols.size());
}
