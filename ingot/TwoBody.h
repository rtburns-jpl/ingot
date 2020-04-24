struct TwoBody {
    double m = 1;

    template<typename T>
    CUDA_HOSTDEV constexpr void operator()(StackArray<T, 6>& yp,
                                           StackArray<T, 6> const& y) const {

        const auto rr = 1 / sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
        const auto rrcube = rr * rr * rr;

        yp[0] = y[3];
        yp[1] = y[4];
        yp[2] = y[5];

        yp[3] = -y[0] * m * rrcube;
        yp[4] = -y[1] * m * rrcube;
        yp[5] = -y[2] * m * rrcube;
    }

    template<typename T>
    CUDA_HOSTDEV constexpr auto operator()(StackArray<T, 6> const& y) const {
        StackArray<T, 6> yp = y;
        (*this)(yp, y);
        return yp;
    }

    template<typename T>
    CUDA_HOSTDEV constexpr auto operator()(const double t,
                                           StackArray<T, 6> const& y) const {
        return (*this)(y);
    }
};