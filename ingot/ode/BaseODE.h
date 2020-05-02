template<typename Derived>
struct HostDevTimeInvariantODE {

    template<typename SA>
    CUDA_HOSTDEV constexpr auto operator()(SA const& y) const {
        SA yp = y;
        static_cast<Derived const&> (*this)(yp, y);
        return yp;
    }

    template<typename SA>
    CUDA_HOSTDEV constexpr auto operator()(const double t, SA const& y) const {
        return (*this)(y);
    }
};
