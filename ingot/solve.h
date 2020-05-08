template<typename T, int N>
struct output {
    double t;
    double h;
    Eigen::Array<T, N, 1> u;
};
template<typename T, int N>
auto make_output(double t, double h, Eigen::Array<T, N, 1> const& u) {
    return output<T, N>{t, h, u};
}
template<typename T, int N>
struct indexedoutput {
    int i;
    double t;
    double h;
    StackArray<T, N> u;
};
template<typename T, int N>
auto make_indexedoutput(int i, double t, double h, StackArray<T, N> const& u) {
    return indexedoutput<T, N>{i, t, h, u};
}

struct SolveArgs {
    bool save_first = true;
    bool save_last = true;
    bool save_all = false;
    double h0 = std::numeric_limits<double>::epsilon();
};

template<typename ODE, typename T, int N, typename Method>
auto solve(ODEProblemImpl<ODE, T, N> prob, Method method,
           SolveArgs const args = {}) {

    Eigen::Array<T, N, 1> statevec = prob.u0;
    double t = prob.t0;
    double h = args.h0;

    auto do_update = [&]() {
        const auto unew = method(prob.ode, t, h, statevec);
        t += h;
        statevec = unew;
    };

    std::vector<output<T, N>> sols;

    if (args.save_first or args.save_all)
        sols.push_back(make_output(t, h, statevec));

    while (t + h < prob.tf) {
        do_update();

        if (args.save_all)
            sols.push_back(make_output(t, h, statevec));
    }

    h = prob.tf - t;
    do_update();

    if (args.save_last or args.save_all)
        sols.push_back(make_output(t, h, statevec));

    return sols;
}
