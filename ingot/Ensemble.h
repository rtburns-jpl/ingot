template<typename T, int N>
class HostEnsemble;

template<typename T, int N>
class Ensemble {
    size_t size;

    friend class HostEnsemble<T, N>;

public:
    thrust::device_vector<double> t;
    thrust::device_vector<double> h;
    DeviceColumnArray<T, N> y;

    Ensemble(size_t size)
        : size{size}, t{size}, h{size}, y{static_cast<int>(size)} {}

    Ensemble(size_t size, double t0, double h0, Eigen::Array<T, N, 1> y0)
        : Ensemble{size} {
        thrust::fill(t.begin(), t.end(), t0);
        thrust::fill(h.begin(), h.end(), h0);
        thrust::fill(y.begin(), y.end(), y0);
    }

    auto begin() { return zip_tuple_iters(t.begin(), h.begin(), y.begin()); }
    auto end() { return begin() + size; }
};

template<typename T, int N>
class HostEnsemble {
    size_t size;

public:
    thrust::host_vector<double> t;
    thrust::host_vector<double> h;
    HostColumnArray<T, N> y;

    HostEnsemble(size_t size)
        : size{size}, t{size}, h{size}, y{static_cast<int>(size)} {}

    HostEnsemble(size_t size, double t0, double h0, Eigen::Array<T, N, 1> y0)
        : HostEnsemble{size} {
        thrust::fill(t.begin(), t.end(), t0);
        thrust::fill(h.begin(), h.end(), h0);
        thrust::fill(y.begin(), y.end(), y0);
    }

    auto begin() { return zip_tuple_iters(t.begin(), h.begin(), y.begin()); }
    auto end() { return begin() + size; }

    auto& operator=(Ensemble<T, N>& other) {
        t = other.t;
        h = other.h;
        y = other.y;
        return *this;
    }
};
