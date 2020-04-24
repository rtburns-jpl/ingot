template<typename T, int N>
struct sol {
    double t;
    StackArray<T, N> u;
};
template<typename T, int N>
auto make_sol(double t, StackArray<T, N> u) {
    return sol<T, N>{t, u};
}

template<typename ODE, typename T, int N, typename Method>
auto solve(ODEProblemImpl<ODE, T, N> prob, Method method) {

    auto statevec = prob.sv0;
    double t = prob.t0;
    double h = 0.1;

    auto do_update = [&]() {
        const auto svnew = method(prob.ode, t, h, statevec);
        t += h;
        statevec = svnew;
    };

    while (t + h < prob.tf) {
        do_update();
    }

    h = prob.tf - t;
    do_update();

    return make_sol(t, statevec);
}
