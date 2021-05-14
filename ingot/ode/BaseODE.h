#pragma once

template<typename Derived>
struct HostDevTimeInvariantODE {

    template<typename T, int N>
    CUDA_HOSTDEV constexpr auto operator()(Eigen::Array<T, N, 1> const& u) const {
        Eigen::Array<T, N, 1> up = u;
        static_cast<Derived const&>(*this)(up, u);
        return up;
    }

    template<typename T, int N>
    CUDA_HOSTDEV constexpr auto operator()(const double t, Eigen::Array<T, N, 1> const& u) const {
        return (*this)(u);
    }
};
