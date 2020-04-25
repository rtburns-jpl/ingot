template<int N>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
Derived& operator=(const ingot::StackArray<value_type, N>& arr) {
    static_assert(ColsAtCompileTime == 1);
    static_assert(RowsAtCompileTime == N);

    for (int i = 0; i < N; i++)
        (*this)[i] = arr[i];

    return derived();
}
