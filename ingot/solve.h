template<typename T, int N>
struct output {
    double t;
    double h;
    StackArray<T, N> u;
};
template<typename T, int N>
auto make_output(double t, double h, StackArray<T, N> const& u) {
    return output<T, N>{t, h, u};
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

    auto statevec = prob.sv0;
    double t = prob.t0;
    double h = args.h0;

    auto do_update = [&]() {
        const auto svnew = method(prob.ode, t, h, statevec);
        t += h;
        statevec = svnew;
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

template<class... Ts>
auto zip_tuple_iters(Ts... ts) {
    return thrust::make_zip_iterator(thrust::make_tuple(ts...));
}

template<typename T>
struct GreaterEqual {
    T val;
    CUDA_DEV bool operator()(const T& x) const {
        return x >= val;
    }
};
template<typename T>
auto makeGreaterEqual(T val) {
    return GreaterEqual<T>{val};
}

template<typename Method, typename ODE>
struct MethodUpdate {
    Method method;
    ODE ode;

    template<typename ZippedTuple>
    CUDA_DEV void operator()(ZippedTuple z) const {
        auto& t = thrust::get<0>(z);
        auto& h = thrust::get<1>(z);
        auto& y = thrust::get<2>(z);

        y = method(ode, t, h, y.stackarray());
        t += h;
    }
};
template<typename Method, typename ODE>
auto makeMethodUpdate(Method m, ODE o) {
    return MethodUpdate<Method, ODE>{m, o};
}

template<typename ODE, typename T, int N, typename Func, typename Method>
auto solve(EnsembleProblemImpl<ODE, T, N, Func> eprob, Method method,
           const size_t nparticles, SolveArgs const args = {}) {

    thrust::device_vector<double> t{nparticles};
    thrust::device_vector<double> h{nparticles};
    DeviceColumnArray<T, N> y{nparticles};

    thrust::fill(t.begin(), t.end(), eprob.prob.t0);
    thrust::fill(h.begin(), h.end(), args.h0);
    thrust::fill(y.begin(), y.end(), eprob.prob.sv0);

    auto zip = zip_tuple_iters(t.begin(), h.begin(), y.begin());

    auto at_least_tf = makeGreaterEqual(eprob.prob.tf);

    // Are all the particles done integrating?
    auto done = [&]() {
        return thrust::all_of(t.begin(), t.end(), at_least_tf);
    };

    // Update the positions/times/etc
    auto update = makeMethodUpdate(method, eprob.prob.ode);
    auto integration_step = [&]() {
        thrust::for_each(zip, zip + nparticles, update);
    };

    do {
        integration_step();
    } while (not done());

    /*
    auto do_update = [&]() {
        thrust::for_each(zip, zip + nparticles, method);
    };
    */
}
