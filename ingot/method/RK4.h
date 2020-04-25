struct RK4 {
    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func const& f, const double t, const double h,
                                 StackArray<T, N> const& y) const {

        using Arr = StackArray<T, N>;

        const Arr k1 = f(t, y);
        const Arr k2 = f(t + h / 2, y + h * k1 / 2);
        const Arr k3 = f(t + h / 2, y + h * k2 / 2);
        const Arr k4 = f(t + h, y + h * k3);

        return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }
};
