#include <ingot/ingot.h>

constexpr int nparticles = 100;

using SA = StackArray<double, 2>;

struct SHO {
    double k;
    __device__ void operator()(SA& yp, SA const& y, double t) {
        yp[0] = y[1];
        yp[1] = -k * y[0];
    }
};

// Called on the index + initial state vector to construct i^th state
struct probfunc {
    __device__
    void operator()(int i, SA& x) {
        x[0] -= double(i) / nparticles;
    }
};

int main() {

    double sv0[6] { -1, 0, 0, 0, 0, 0 };

    double tspan[2] = { 0., 100. };

    auto prob = ODEProblem(SHO{1},
                           sv0,
                           tspan);

    auto eprob = EnsembleProblem(prob,
                                 probfunc{});

    //solve(eprob, RKF78{}, nparticles);
}
