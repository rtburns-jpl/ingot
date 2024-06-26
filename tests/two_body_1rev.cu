#include <doctest/doctest.h>

#include <ingot/ingot.h>
#include <ingot/integrator/adaptive.h>
#include <ingot/method/all.h>
#include <ingot/ode/TwoBody.h>
using namespace ingot;

TEST_CASE("cpu fixed-step two body circular") {

    double u0[6]{1, 0, 0, 0, 1, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{}, u0, tspan);

    SolveArgs args;
    args.h0 = 0.1;
    Eigen::Array<double, 6, 1> first{u0};

    auto sols = solve(prob, method::RK4{}, args);
    auto last = sols.back().u;
    double diff = (first - last).matrix().norm();
    CHECK(diff > std::numeric_limits<double>::epsilon());
    CHECK(diff < 3.3e-5);

    sols = solve(prob, method::DoPri45{}, args);
    last = sols.back().u;
    diff = (first - last).matrix().norm();
    CHECK(diff > std::numeric_limits<double>::epsilon());
    CHECK(diff < 1.1e-7);

    sols = solve(prob, method::Tsit5{}, args);
    last = sols.back().u;
    diff = (first - last).matrix().norm();
    CHECK(diff > std::numeric_limits<double>::epsilon());
    CHECK(diff < 1.1e-7);

    sols = solve(prob, method::RKF78{}, args);
    last = sols.back().u;
    diff = (first - last).matrix().norm();
    CHECK(diff > std::numeric_limits<double>::epsilon());
    CHECK(diff < 7e-12);
}

struct probfunc {
    CUDA_DEV auto operator()(int i, Eigen::Array<double, 6, 1> u) const {
        return u;
    }
};

TEST_CASE("ensemble fixed-step two body circular") {

    double u0[6]{1, 0, 0, 0, 1, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{}, u0, tspan);
    auto eprob = EnsembleProblem(prob, probfunc{});

    SolveArgs args;
    args.h0 = 0.1;
    Eigen::Array<double, 6, 1> first{u0};

    {
        const auto i = integrator::make_fixed(method::RK4{});
        const auto sols = solve(eprob, i, 1, args);
        const auto last = sols.back().u;
        const double diff = (first - last).matrix().norm();
        CHECK(diff > std::numeric_limits<double>::epsilon());
        CHECK(diff < 3.3e-5);
    }

    {
        const auto i = integrator::make_fixed(method::DoPri45{});
        const auto sols = solve(eprob, i, 1, args);
        const auto last = sols.back().u;
        const double diff = (first - last).matrix().norm();
        CHECK(diff > std::numeric_limits<double>::epsilon());
        CHECK(diff < 1.1e-7);
    }

    {
        const auto i = integrator::make_fixed(method::Tsit5{});
        const auto sols = solve(eprob, i, 1, args);
        const auto last = sols.back().u;
        const double diff = (first - last).matrix().norm();
        CHECK(diff > std::numeric_limits<double>::epsilon());
        CHECK(diff < 1.1e-7);
    }

    {
        const auto i = integrator::make_fixed(method::RKF78{});
        const auto sols = solve(eprob, i, 1, args);
        const auto last = sols.back().u;
        const double diff = (first - last).matrix().norm();
        CHECK(diff > std::numeric_limits<double>::epsilon());
        CHECK(diff < 7e-12);
    }
}

TEST_CASE("ensemble adaptive-step two body circular") {

    double u0[6]{1, 0, 0, 0, 1, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{}, u0, tspan);
    auto eprob = EnsembleProblem(prob, probfunc{});

    SolveArgs args;
    args.h0 = 0.1;
    Eigen::Array<double, 6, 1> first{u0};

    {
        const auto i = integrator::make_adaptive(method::DoPri45{}, 1e-8);
        const auto sols = solve(eprob, i, 1, args);
        const auto last = sols.back().u;
        const double diff = (first - last).matrix().norm();
        CHECK(diff > std::numeric_limits<double>::epsilon());
        CHECK(diff < 1e-6);
    }

    {
        const auto i = integrator::make_adaptive(method::Tsit5{}, 1e-2);
        const auto sols = solve(eprob, i, 1, args);
        const auto last = sols.back().u;
        const double diff = (first - last).matrix().norm();
        CHECK(diff > std::numeric_limits<double>::epsilon());
        CHECK(diff < 1e-6);
    }

    {
        const auto i = integrator::make_adaptive(method::RKF78{}, 1e-8);
        const auto sols = solve(eprob, i, 1, args);
        const auto last = sols.back().u;
        const double diff = (first - last).matrix().norm();
        CHECK(diff > std::numeric_limits<double>::epsilon());
        CHECK(diff < 1e-6);
    }
}
