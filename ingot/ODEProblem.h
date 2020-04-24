template<typename OrdinaryDiffEq, typename T, int N // State vector info
         >
struct ODEProblemImpl {
    double t0;
    double tf;
    OrdinaryDiffEq ode;
    StackArray<T, N> sv0;

    ODEProblemImpl(OrdinaryDiffEq ode, T (&x)[N], double (&t)[2])
        : ode{ode}, sv0{x}, t0{t[0]}, tf{t[1]} {}
};

template<typename ODE, typename T, int N>
auto ODEProblem(ODE ode, T (&sv0)[N], double (&tspan)[2]) {

    return ODEProblemImpl<ODE, T, N>{ode, sv0, tspan};
}
