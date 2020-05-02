struct Frac {
    int numer;
    int denom;

    CUDA_HOSTDEV
    constexpr operator double() const { return double(numer) / denom; }
};
