#include <ingot/ingot.h>
using namespace ingot;

int main() {
    double sv0[6]{1, 0, 0, 0, 1, 0};

    double tspan[2] = {0., 2 * M_PI};

    auto prob = ODEProblem(ode::TwoBody{1}, sv0, tspan);

    SolveArgs args;
    args.save_all = true;
    args.h0 = 0.1;
    auto sols = solve(prob, method::RK4{}, args);

    for (auto const& sol : sols) {
        printf("% .4f, % .4f\n", sol.u[0], sol.u[1]);
    }
}
