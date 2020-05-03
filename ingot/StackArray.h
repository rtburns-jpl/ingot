template<typename T, int N>
class StackArray {

    static_assert(N > 0, "Size must be positive");

    T data[N];

public:
    StackArray() = default;
    StackArray(StackArray const& other) = default;

    CUDA_HOSTDEV
    static auto Zero() {
        auto ret = StackArray{};
        ret.zero();
        return ret;
    }

    CUDA_HOSTDEV
    StackArray(T (&x)[N]) {
        for (int i = 0; i < N; i++)
            data[i] = x[i];
    }

    template<typename U>
    CUDA_HOSTDEV StackArray(Eigen::ArrayBase<U> const& e) {
        for (int i = 0; i < N; i++)
            data[i] = e[i];
    }

    CUDA_HOSTDEV void fill(T x) {
        for (int i = 0; i < N; i++)
            data[i] = x;
    }
    CUDA_HOSTDEV void zero() { fill(0); }

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
    auto& operator*=(T const x) {
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
    auto operator+(T const x) const {
        StackArray s{*this};
        return s += x;
    }

    CUDA_HOSTDEV
    auto& operator+=(T const x) {
        for (int i = 0; i < N; i++)
            data[i] += x;
        return *this;
    }

    CUDA_HOSTDEV
    auto operator*(T const x) const {
        StackArray s{*this};
        return s *= x;
    }

    CUDA_HOSTDEV
    auto operator*(T const x) {
        for (int i = 0; i < N; i++)
            data[i] *= x;
        return *this;
    }

    CUDA_HOSTDEV
    auto operator/(T const x) const {
        StackArray ret{*this};
        for (int i = 0; i < N; i++)
            ret[i] /= x;
        return ret;
    }

    CUDA_HOSTDEV
    auto& operator/=(T const x) {
        for (int i = 0; i < N; i++)
            data[i] /= x;
        return *this;
    }

    CUDA_HOSTDEV
    auto operator[](int i) const { return data[i]; };
    CUDA_HOSTDEV
    auto& operator[](int i) { return data[i]; };

    auto norm() const {
        T ret = 0;
        for (int i = 0; i < N; i++)
            ret += data[i] * data[i];
        return sqrt(ret);
    }
};

template<typename T, int N>
CUDA_HOSTDEV auto operator+(const double x, StackArray<T, N> const& a) {
    StackArray<T, N> ret;
    for (int i = 0; i < N; i++)
        ret[i] = x * a[i];
    return ret;
}

template<typename T, int N>
CUDA_HOSTDEV auto operator*(const double x, StackArray<T, N> const& a) {
    StackArray<T, N> ret;
    for (int i = 0; i < N; i++)
        ret[i] = x * a[i];
    return ret;
}
