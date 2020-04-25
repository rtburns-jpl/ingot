#include <ingot/ingot.h>
using namespace ingot;

#include <gtest/gtest.h>

TEST(TwoBody, CircularOrbit) {

    double sv0[6]{1, 0, 0, 0, 1, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{1}, sv0, tspan);

    auto sols = solve(prob, RK4{});

    StackArray<double, 6> first{sv0};
    auto last = sols.back().u;

    EXPECT_NEAR((first - last).norm(), 0, 1e-5);
}
