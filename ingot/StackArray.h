#pragma once

template<typename T, int N>
struct StackArray {

    static_assert(N > 0, "Size must be positive");

    T data[N];

    CUDA_HOSTDEV
    StackArray() {}

    StackArray(StackArray const& other) = default;

    StackArray(T (&x)[N]) {
        for (int i = 0; i < N; i++)
            data[i] = x[i];
    }

    template<typename U>
    CUDA_HOSTDEV
    StackArray(Eigen::ArrayBase<U> const& e) {
        for (int i = 0; i < N; i++)
            data[i] = e[i];
    }

    CUDA_HOSTDEV
    operator Eigen::Array<T, N, 1>() const {
        Eigen::Array<T, N, 1> ret;
        for (int i = 0; i < N; i++)
            ret[i] = data[i];
        return ret;
    }

    CUDA_HOSTDEV
    auto& operator+=(StackArray const& other) {
        for (int i = 0; i < N; i++)
            data[i] += other.data[i];
        return *this;
    }

    CUDA_HOSTDEV
    auto& operator-=(StackArray const& other) {
        for (int i = 0; i < N; i++)
            data[i] -= other.data[i];
        return *this;
    }

    CUDA_HOSTDEV
    auto& operator*=(StackArray const& other) {
        for (int i = 0; i < N; i++)
            data[i] *= other.data[i];
        return *this;
    }
    CUDA_HOSTDEV
    auto& operator*=(double const x) {
        for (int i = 0; i < N; i++)
            data[i] *= x;
        return *this;
    }

    CUDA_HOSTDEV
    auto operator+(StackArray const& other) const {
        StackArray s{*this};
        return s += other;
    }
    CUDA_HOSTDEV
    auto operator-(StackArray const& other) const {
        StackArray s{*this};
        return s -= other;
    }

    CUDA_HOSTDEV
    auto operator+(double const x) const {
        StackArray s{*this};
        return s += x;
    }

    CUDA_HOSTDEV
    auto& operator+=(double const x) {
        for (int i = 0; i < N; i++)
            data[i] += x;
        return *this;
    }

    CUDA_HOSTDEV
    auto operator*(double const x) const {
        StackArray s{*this};
        return s *= x;
    }

    CUDA_HOSTDEV
    auto operator*(double const x) {
        for (int i = 0; i < N; i++)
            data[i] *= x;
        return *this;
    }

    CUDA_HOSTDEV
    auto operator/(double const x) const {
        StackArray ret{*this};
        for (int i = 0; i < N; i++)
            ret[i] /= x;
        return ret;
    }

    CUDA_HOSTDEV
    auto& operator/=(double const x) {
        for (int i = 0; i < N; i++)
            data[i] /= x;
        return *this;
    }

    CUDA_HOSTDEV
    auto operator[](int i) const { return data[i]; };
    CUDA_HOSTDEV
    auto& operator[](int i) { return data[i]; };
};

template<typename T, int N>
CUDA_HOSTDEV
auto operator+(const double x, StackArray<T, N> const& a) {
    StackArray<T, N> ret;
    for (int i = 0; i < N; i++)
        ret[i] = x * a[i];
    return ret;
}

template<typename T, int N>
CUDA_HOSTDEV
auto operator*(const double x, StackArray<T, N> const& a) {
    StackArray<T, N> ret;
    for (int i = 0; i < N; i++)
        ret[i] = x * a[i];
    return ret;
}
