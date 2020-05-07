#include <ingot/ingot.h>
using namespace ingot;

constexpr int nparticles = 100;

using SA = StackArray<double, 2>;

struct SHO {
    double k;
    CUDA_DEV void operator()(SA& u_prime, SA const& u, double t) {
        u_prime[0] = u[1];
        u_prime[1] = -k * u[0];
    }
};

// Called on the index + initial state vector to construct i^th state
struct probfunc {
    CUDA_DEV void operator()(int i, SA& u) {
        u[0] -= double(i) / nparticles;
    }
};

int main() {

    double u0[6]{-1, 0, 0, 0, 0, 0};

    double tspan[2] = {0., 100.};

    auto prob = ODEProblem(SHO{1}, u0, tspan);

    auto eprob = EnsembleProblem(prob, probfunc{});

    // solve(eprob, RKF78{}, nparticles);
}
