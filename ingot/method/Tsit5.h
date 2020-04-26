template<typename T, typename U, int N, typename Func>
CUDA_HOSTDEV
auto rk_tableau(T (&a)[N-1][N-1], T (&b)[N], T (&c)[N], Func f,
                const double t, const double h, U const& y) {
    auto get_t = [=](int i) {
        return t + c[i] * h;
    };

    U k[N];

    auto get_ak = [&](const int i) {
        U ret = U::Zero();
        for (int j = 0; j < i; ++j) {
            ret += a[i-1][j] * k[j];
        }
        return ret * h;
    };

    for (int i = 0; i < N; ++i) {
        k[i] = f(get_t(i), y + get_ak(i));
    }

    U bk = U::Zero();
    for (int i = 0; i < N; ++i) {
        bk += b[i] * k[i];
    }
    return y + bk * h;
}

struct Tsit5 {
    template<typename Func, typename T, int N>
    CUDA_HOSTDEV auto operator()(Func const& f, const double t, const double h,
                                 StackArray<T, N> const& y) const {

        using Coeff = T; // TODO use scalar type for complex

        // https://en.wikipedia.org/wiki/Dormand–Prince_method

        static constexpr int RKN = 7;

        static constexpr Coeff a[RKN-1][RKN-1] {
            {
                0.161,
            }, {
                -0.008480655492357,
                0.3354806554923570,
            }, {
                2.897153057105494,
                -6.359448489975075,
                4.362295432869581,
            }, {
                5.32586482843925895,
                -11.74888356406283,
                7.495539342889836,
                -0.09249506636175525,
            }, {
                5.86145544294642038,
                -12.92096931784711,
                8.159367898576159,
                -0.071584973281401006,
                -0.02826905039406838,
            }, {
                0.09646076681806523,
                0.01,
                0.4798896504144996,
                1.379008574103742,
                -3.290069515436081,
                2.324710524099774,
            }
        };

        static constexpr Coeff b[RKN] {
            0.09646076681806523,
            0.01,
            0.4798896504144996,
            1.379008574103742,
            -3.290069515436081,
            2.324710524099774,
            0,
        };

        static constexpr Coeff c[RKN] {
            0,
            0.161,
            0.327,
            0.9,
            0.9800255409045097,
            1,
            1,
        };

        return rk_tableau(a, b, c, f, t, h, y);
    }
};
