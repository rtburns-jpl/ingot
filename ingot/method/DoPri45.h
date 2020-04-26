struct DoPri45 {
    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func const& f, const double t, const double h,
                                 StackArray<T, N> const& y) const {

        using Arr = StackArray<T, N>;
        using Coeff = T; // TODO use scalar type for complex

        // https://en.wikipedia.org/wiki/Dormand–Prince_method

        static constexpr Coeff c2 = Frac{1, 5};
        static constexpr Coeff a21 = Frac{1, 5};

        static constexpr Coeff c3 = Frac{3, 10};
        static constexpr Coeff a31 = Frac{3, 40};
        static constexpr Coeff a32 = Frac{9, 40};

        static constexpr Coeff c4 = Frac{4, 5};
        static constexpr Coeff a41 = Frac{44, 45};
        static constexpr Coeff a42 = Frac{-56, 15};
        static constexpr Coeff a43 = Frac{32, 9};

        static constexpr Coeff c5 = Frac{8, 9};
        static constexpr Coeff a51 = Frac{19372, 6561};
        static constexpr Coeff a52 = Frac{-25360, 2187};
        static constexpr Coeff a53 = Frac{64448, 6561};
        static constexpr Coeff a54 = Frac{-212, 729};

        static constexpr Coeff c6 = 1;
        static constexpr Coeff a61 = Frac{9017, 3168};
        static constexpr Coeff a62 = Frac{-355, 33};
        static constexpr Coeff a63 = Frac{46732, 5247};
        static constexpr Coeff a64 = Frac{49, 176};
        static constexpr Coeff a65 = Frac{-5103, 18656};

        // static constexpr Coeff c7 = 1;
        static constexpr Coeff a71 = Frac{35, 384};
        static constexpr Coeff a73 = Frac{500, 1113};
        static constexpr Coeff a74 = Frac{125, 192};
        static constexpr Coeff a75 = Frac{-2187, 6784};
        static constexpr Coeff a76 = Frac{11, 84};

        const Arr k1 = f(t, y);
        const Arr k2 = f(t + c2 * h, y + a21 * k1 * h);
        const Arr k3 = f(t + c3 * h, y + (a31 * k1 + a32 * k2) * h);
        const Arr k4 = f(t + c4 * h, y + (a41 * k1 + a42 * k2 + a43 * k3) * h);
        const Arr k5 = f(t + c5 * h, y + (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4) * h);
        const Arr k6 = f(t + c6 * h, y + (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5) * h);

        return y + (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6) * h;
    }
};
