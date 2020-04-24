#pragma once

template<typename T, int N>
struct StackArray {

    static_assert(N > 0, "Size must be positive");

    T data[N];

    __host__ __device__
    StackArray() {}

    StackArray(StackArray const& other) = default;

    template<typename U>
    __host__ __device__
    StackArray(Eigen::ArrayBase<U> const& e) {
        for (int i = 0; i < N; i++)
            data[i] = e[i];
    }

    __host__ __device__
    operator Eigen::Array<T, N, 1>() const {
        Eigen::Array<T, N, 1> ret;
        for (int i = 0; i < N; i++)
            ret[i] = data[i];
        return ret;
    }

    __host__ __device__
    auto& operator+=(StackArray const& other) {
        for (int i = 0; i < N; i++)
            data[i] += other.data[i];
        return *this;
    }

    __host__ __device__
    auto& operator-=(StackArray const& other) {
        for (int i = 0; i < N; i++)
            data[i] -= other.data[i];
        return *this;
    }

    __host__ __device__
    auto& operator*=(StackArray const& other) {
        for (int i = 0; i < N; i++)
            data[i] *= other.data[i];
        return *this;
    }
    __host__ __device__
    auto& operator*=(double const x) {
        for (int i = 0; i < N; i++)
            data[i] *= x;
        return *this;
    }

    __host__ __device__
    auto operator+(StackArray const& other) const {
        StackArray s{*this};
        return s += other;
    }
    __host__ __device__
    auto operator-(StackArray const& other) const {
        StackArray s{*this};
        return s -= other;
    }

    __host__ __device__
    auto operator+(double const x) const {
        StackArray s{*this};
        return s += x;
    }

    __host__ __device__
    auto& operator+=(double const x) {
        for (int i = 0; i < N; i++)
            data[i] += x;
        return *this;
    }

    __host__ __device__
    auto operator*(double const x) const {
        StackArray s{*this};
        return s *= x;
    }

    __host__ __device__
    auto operator*(double const x) {
        for (int i = 0; i < N; i++)
            data[i] *= x;
        return *this;
    }

    __host__ __device__
    auto operator/(double const x) const {
        StackArray ret{*this};
        for (int i = 0; i < N; i++)
            ret[i] /= x;
        return ret;
    }

    __host__ __device__
    auto& operator/=(double const x) {
        for (int i = 0; i < N; i++)
            data[i] /= x;
        return *this;
    }

    __host__ __device__
    auto operator[](int i) const { return data[i]; };
    __host__ __device__
    auto& operator[](int i) { return data[i]; };
};

template<typename T, int N>
__host__ __device__
auto operator+(const double x, StackArray<T, N> const& a) {
    StackArray<T, N> ret;
    for (int i = 0; i < N; i++)
        ret[i] = x * a[i];
    return ret;
}

template<typename T, int N>
__host__ __device__
auto operator*(const double x, StackArray<T, N> const& a) {
    StackArray<T, N> ret;
    for (int i = 0; i < N; i++)
        ret[i] = x * a[i];
    return ret;
}
