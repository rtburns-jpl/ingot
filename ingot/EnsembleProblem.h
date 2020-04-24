#pragma once

template<typename ODE, typename T, int N, typename Func>
struct EnsembleProblemImpl {
    ODEProblemImpl<ODE, T, N> prob;
    Func prob_func;
};

template<typename ODE, typename T, int N, typename Func>
auto EnsembleProblem(ODEProblemImpl<ODE, T, N> p, Func f) {
    return EnsembleProblemImpl<ODE, T, N, Func>{p, f};
}
