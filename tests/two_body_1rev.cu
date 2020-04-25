#include <gtest/gtest.h>

#include <ingot/ingot.h>
using namespace ingot;

TEST(TwoBody, CircularOrbit) {

    double sv0[6]{1, 0, 0, 0, 1, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{}, sv0, tspan);

    SolveArgs args;
    args.h0 = 0.1;
    auto sols = solve(prob, method::RK4{}, args);

    StackArray<double, 6> first{sv0};
    auto last = sols.back().u;

    EXPECT_NEAR((first - last).norm(), 0, 1e-8);
}

struct probfunc {
    CUDA_DEV auto operator()(int i, StackArray<double, 6> y) const {
        return y;
    }
};

TEST(TwoBody, CircularOrbitEnsemble) {

    double sv0[6]{1, 0, 0, 0, 1, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{}, sv0, tspan);
    auto eprob = EnsembleProblem(prob, probfunc{});

    SolveArgs args;
    args.h0 = 0.1;
    solve(eprob, method::RK4{}, 100, args);

    /*
    StackArray<double, 6> first{sv0};
    auto last = sols.back().u;

    EXPECT_NEAR((first - last).norm(), 0, 1e-8);
    */
}
