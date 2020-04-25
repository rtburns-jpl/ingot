#include <gtest/gtest.h>

#include <ingot/ingot.h>
using namespace ingot;

TEST(TwoBody, CircularOrbit) {

    double sv0[6]{1, 0, 0, 0, 1, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{}, sv0, tspan);

    SolveArgs args;
    args.h0 = 0.1;
    auto sols = solve(prob, RK4{}, args);

    StackArray<double, 6> first{sv0};
    auto last = sols.back().u;

    EXPECT_NEAR((first - last).norm(), 0, 1e-8);
}
