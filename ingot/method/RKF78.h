struct RKF78 {
    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func const& f, double const t, double const h,
                                 StackArray<T, N> const& y) const {

        StackArray<T, N> yp, err;

        method(t, h, y, yp, err, f);

        return yp;
    }
};
