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

struct Always {
    template<typename Ts>
    CUDA_HOSTDEV bool operator()(Ts) const { return true; }
};

template<typename ODE, typename T, int N, typename Func, typename Integrator>
auto solve(EnsembleProblemImpl<ODE, T, N, Func> eprob,
           Integrator integrator,
           const size_t nparticles, SolveArgs const args = {})
    -> std::vector<output<T, N>>
{

    Ensemble<T, N> ensemble{nparticles, eprob.prob.t0,
                            args.h0, eprob.prob.u0};

    integrator.tf = eprob.prob.tf;

    // Are all the particles done integrating?
    auto at_least_tf = makeGreaterEqual(eprob.prob.tf);
    auto done = [&]() {
        return thrust::all_of(
                ensemble.t.begin(),
                ensemble.t.end(),
                at_least_tf);
    };

    auto ode = eprob.prob.ode;
    auto ode_integrator = [=] CUDA_HOSTDEV (double& t, double& h,
                                            ColRef<T, N> y) {
        integrator(ode, t, h, y);
    };

    // a buffer to hold coalesced outputs
    Ensemble<T, N> gpu_output_buffer{nparticles};
    HostEnsemble<T, N> cpu_output_buffer{nparticles};

    std::vector<output<T, N>> sols;

    auto output_cond = Always{}; // TODO user-specified

    auto fetch_output = [&]() {
        // Copy outputs
        auto nout = thrust::count_if(ensemble.begin(), ensemble.end(),
                                     output_cond);
        thrust::copy_if(ensemble.begin(), ensemble.end(),
                        gpu_output_buffer.begin(), output_cond);

        // Do device -> host memcpy
        cpu_output_buffer = gpu_output_buffer;

        // Write to output vector
        for (auto i = cpu_output_buffer.begin();
                 i != cpu_output_buffer.end(); i++) {
            sols.push_back({
                thrust::get<0>(*i),
                thrust::get<1>(*i),
                thrust::get<2>(*i),
            });
        }
    };

    fetch_output();

    do {
        // Do integration step
        thrust::for_each(ensemble.begin(), ensemble.end(),
                         thrust::apply_func(ode_integrator));

        fetch_output();

    } while (not done());

    /*
    auto do_update = [&]() {
        thrust::for_each(zip, zip + nparticles, method);
    };
    */

    return sols;
}
