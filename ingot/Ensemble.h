template<typename T, int N>
class HostEnsemble;

template<typename T, int N>
class Ensemble {
    size_t size;

    friend class HostEnsemble<T, N>;

public:
    thrust::device_vector<double> t;
    thrust::device_vector<double> h;
    DeviceColumnArray<T, N> u;

    Ensemble(size_t size)
        : size{size}, t{size}, h{size}, u{static_cast<int>(size)} {}

    Ensemble(size_t size, double t0, double h0, Eigen::Array<T, N, 1> u0)
        : Ensemble{size} {
        thrust::fill(t.begin(), t.end(), t0);
        thrust::fill(h.begin(), h.end(), h0);
        thrust::fill(u.begin(), u.end(), u0);
    }

    auto& operator=(const Ensemble& other) {
        t = other.t;
        h = other.h;
        u = other.u;
        return *this;
    }

    auto begin() { return zip_tuple_iters(t.begin(), h.begin(), u.begin()); }
    auto end() { return begin() + size; }

    auto get_size() const { return size; }

    void swap(Ensemble& other) {
        std::swap(size, other.size);
        t.swap(other.t);
        h.swap(other.h);
        u.swap(other.u);
    }
};

template<typename T, int N>
class HostEnsemble {
    size_t size;

public:
    thrust::host_vector<double> t;
    thrust::host_vector<double> h;
    HostColumnArray<T, N> u;

    HostEnsemble(size_t size)
        : size{size}, t{size}, h{size}, u{static_cast<int>(size)} {}

    HostEnsemble(size_t size, double t0, double h0, Eigen::Array<T, N, 1> u0)
        : HostEnsemble{size} {
        thrust::fill(t.begin(), t.end(), t0);
        thrust::fill(h.begin(), h.end(), h0);
        thrust::fill(u.begin(), u.end(), u0);
    }

    auto begin() { return zip_tuple_iters(t.begin(), h.begin(), u.begin()); }
    auto end() { return begin() + size; }

    auto& operator=(Ensemble<T, N>& other) {
        t = other.t;
        h = other.h;
        u = other.u;
        return *this;
    }
};
