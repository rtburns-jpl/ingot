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

    template<typename T, int N>
    CUDA_DEV void operator()(double& t, double& h, ColRef<T, N> y) const {
        y = method(ode, t, h, y.stackarray());
        t += h;
    }
};
template<typename Method, typename ODE>
auto makeMethodUpdate(Method m, ODE o) {
    return MethodUpdate<Method, ODE>{m, o};
}

struct Always {
    template<typename Ts>
    CUDA_HOSTDEV bool operator()(Ts) const { return true; }
};

template<typename ODE, typename T, int N, typename Func, typename Method>
auto solve(EnsembleProblemImpl<ODE, T, N, Func> eprob, Method method,
           const size_t nparticles, SolveArgs const args = {}) {

    Ensemble<T, N> ensemble{nparticles, eprob.prob.t0,
                            args.h0, eprob.prob.sv0};

    auto at_least_tf = makeGreaterEqual(eprob.prob.tf);

    // Are all the particles done integrating?
    auto done = [&]() {
        return thrust::all_of(
                ensemble.t.begin(),
                ensemble.t.end(),
                at_least_tf);
    };

    // Update the positions/times/etc
    auto update = makeMethodUpdate(method, eprob.prob.ode);
    auto integration_step = [&]() {
        for_each(ensemble.begin(), ensemble.end(),
                 thrust::apply_func(update));
    };

    // a buffer to hold coalesced outputs
    Ensemble<T, N> gpu_output_buffer{nparticles};

    auto output_cond = Always{}; // TODO user-specified

    do {
        integration_step();

        // Copy outputs
        /*
        thrust::copy_if(ensemble.begin(), ensemble.end(),
                        gpu_output_buffer.begin(),
                        output_cond);
                        */
        thrust::copy(ensemble.begin(), ensemble.end(),
                     gpu_output_buffer.begin());

    } while (not done());

    /*
    auto do_update = [&]() {
        thrust::for_each(zip, zip + nparticles, method);
    };
    */
}
