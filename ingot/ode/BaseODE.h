template<typename Derived>
struct HostDevTimeInvariantODE {

    template<typename SA>
    CUDA_HOSTDEV constexpr auto operator()(SA const& u) const {
        SA up = u;
        static_cast<Derived const&> (*this)(up, u);
        return up;
    }

    template<typename SA>
    CUDA_HOSTDEV constexpr auto operator()(const double t, SA const& u) const {
        return (*this)(u);
    }
};
