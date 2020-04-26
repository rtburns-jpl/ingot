EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Derived&
operator=(const ingot::StackArray<value_type, RowsAtCompileTime>& arr) {
    static_assert(ColsAtCompileTime == 1);
    static_assert(RowsAtCompileTime != Eigen::Dynamic);

    for (int i = 0; i < RowsAtCompileTime; i++)
        (*this)[i] = arr[i];

    return derived();
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE auto stackarray() const {
    static_assert(ColsAtCompileTime == 1);
    static_assert(RowsAtCompileTime != Eigen::Dynamic);

    ingot::StackArray<value_type, RowsAtCompileTime> ret;
    for (int i = 0; i < RowsAtCompileTime; i++)
        ret[i] = (*this)[i];
    return ret;
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
operator ingot::StackArray<value_type, RowsAtCompileTime>() const {
    return stackarray();
}
