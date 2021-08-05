#include "ingot.h"

namespace ingot {

template<typename T>
CUDA_HOSTDEV int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename EventFn>
struct EventFnChanged {
    EventFn e;
    template<typename T>
    CUDA_HOSTDEV bool operator()(T prev_next) const {
        const auto a = thrust::apply(e, thrust::get<0>(prev_next));
        const auto b = thrust::apply(e, thrust::get<1>(prev_next));
        return sgn(a) != sgn(b);
    }
};

struct Done {
    double t_max;
    template<typename T>
    CUDA_HOSTDEV bool operator()(T const& args) const {
        auto const t = thrust::get<0>(args);
        return t >= t_max;
    }
};

template<typename Integrator, typename ODE, typename T, int N, typename EventFn>
auto integrate_steps(Integrator integrator, ODE ode, Ensemble<T, N> prev,
                     int max_iters, EventFn eventfn
                     ) -> std::vector<output<T, N>> {

    auto eventfn_changed = EventFnChanged<EventFn>{eventfn};

    auto nparticles = prev.get_size();

    Ensemble<T, N> next{nparticles};

    auto ode_integrator = [integrator, ode] CUDA_HOSTDEV(double& t, double& h,
                                           ColRef<T, N> y) {
        integrator(ode, t, h, y);
    };

    // a buffer to hold coalesced outputs
    Ensemble<T, N> gpu_out_buffer{nparticles};

    HostEnsemble<T, N> cpu_out_buffer{nparticles};

    std::vector<output<T, N>> sols;

    next = prev;

    for (int i = 0; i < max_iters; ++i) {
        // Do integration step

        // transform in place
        thrust::for_each(next.begin(), next.end(),
                         thrust::apply_func(ode_integrator));

        do {
            auto both = zip_tuple_iters(prev.begin(), next.begin());

            auto out_start = gpu_out_buffer.begin();
            auto out_end = thrust::copy_if(prev.begin(), prev.end(),
                                           both,      // stencil
                                           out_start, // output
                                           eventfn_changed);
            auto outsize = out_end - out_start;
            if (outsize == 0) {
                break;
            }

            // Do device -> host memcpy
            cpu_out_buffer = gpu_out_buffer;

            // Write to output vector
            for (auto i = cpu_out_buffer.begin();
                    i != cpu_out_buffer.begin() + outsize; i++) {
                sols.push_back({
                        thrust::get<0>(*i),
                        thrust::get<1>(*i),
                        thrust::get<2>(*i),
                });
            }
        } while (false);

        prev = next;
    }

    return sols;
}

template<typename Integrator, typename ODE, typename T, int N, typename EventFn>
auto integrate_time(Integrator integrator, ODE ode, Ensemble<T, N> prev,
                    double t_max, EventFn eventfn
                    ) -> std::vector<output<T, N>> {

    auto eventfn_changed = EventFnChanged<EventFn>{eventfn};

    auto nparticles = prev.get_size();

    Ensemble<T, N> next{nparticles};

    auto ode_integrator = [integrator, ode] CUDA_HOSTDEV(double& t, double& h,
                                           ColRef<T, N> y) {
        integrator(ode, t, h, y);
    };

    // a buffer to hold coalesced outputs
    Ensemble<T, N> gpu_out_buffer{nparticles};

    HostEnsemble<T, N> cpu_out_buffer{nparticles};

    std::vector<output<T, N>> sols;

    next = prev;

    auto done = Done{t_max};

    auto current_end = next.end();

    // while there are particles to integrate
    while (current_end != next.begin()) {

        // Do integration step

        // transform in place
        thrust::for_each(next.begin(), current_end,
                         thrust::apply_func(ode_integrator));

        do {
            auto both = zip_tuple_iters(prev.begin(), next.begin());

            auto out_start = gpu_out_buffer.begin();
            auto out_end = thrust::copy_if(prev.begin(), prev.end(),
                                           both,      // stencil
                                           out_start, // output
                                           eventfn_changed);
            auto outsize = out_end - out_start;
            if (outsize == 0) {
                break;
            }

            // Do device -> host memcpy
            cpu_out_buffer = gpu_out_buffer;

            // Write to output vector
            for (auto i = cpu_out_buffer.begin();
                    i != cpu_out_buffer.begin() + outsize; i++) {
                sols.push_back({
                        thrust::get<0>(*i),
                        thrust::get<1>(*i),
                        thrust::get<2>(*i),
                });
            }
        } while (false);

        // remove any particles which are done
        current_end = thrust::remove_if(next.begin(), current_end, done);

        prev = next;
    }

    return sols;
}

template<typename Integrator, typename ODE, typename T, int N>
auto integrate_dense(Integrator integrator, ODE ode, Ensemble<T, N> prev,
                     double t_max) -> std::vector<std::vector<output<T, N>>> {

    auto nparticles = prev.get_size();

    Ensemble<T, N> next{nparticles};

    auto ode_integrator = [integrator, ode] CUDA_HOSTDEV(double& t, double& h,
                                           ColRef<T, N> y) {
        integrator(ode, t, h, y);
    };

    // a buffer to hold coalesced outputs
    Ensemble<T, N> gpu_out_buffer{nparticles};

    HostEnsemble<T, N> cpu_out_buffer{nparticles};

    std::vector<std::vector<output<T, N>>> sols(nparticles);

    next = prev;

    auto done = Done{t_max};

    auto current_end = next.end();

    // while there are particles to integrate
    while (current_end != next.begin()) {

        // Do integration step

        // transform in place
        thrust::for_each(next.begin(), current_end,
                         thrust::apply_func(ode_integrator));

        do {
            // Do device -> host memcpy
            cpu_out_buffer = prev;

            // Write to output vector
            for (auto i = cpu_out_buffer.begin();
                    i != cpu_out_buffer.end(); i++) {
                int idx = i - cpu_out_buffer.begin();
                sols[idx].push_back({
                        thrust::get<0>(*i),
                        thrust::get<1>(*i),
                        thrust::get<2>(*i),
                });
            }
        } while (false);

        // remove any particles which are done
        current_end = thrust::remove_if(next.begin(), current_end, done);

        prev = next;
    }

    return sols;
}

} // namespace ingot
