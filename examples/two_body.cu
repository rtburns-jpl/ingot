#include <ingot/ingot.h>
using namespace ingot;

constexpr int nparticles = 100;

using SA = StackArray<double, 6>;

// Called on the index + initial state vector to construct i^th state
struct probfunc {
    __device__ void operator()(int i, SA& x) { x[0] -= double(i) / nparticles; }
};

int main() {

    double sv0[6]{-1, 0, 0, 0, 0, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{1}, sv0, tspan);

    // solve(prob, RKF78{});
}
