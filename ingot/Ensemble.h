template<typename T, int N>
class Ensemble {
    size_t size;

public:
    thrust::device_vector<double> t;
    thrust::device_vector<double> h;
    DeviceColumnArray<T, N> y;

    Ensemble(size_t size, double t0, double h0, Eigen::Array<T, N, 1> y0) :
        size{size},
        t{size},
        h{size},
        y{int{size}}
    {
        thrust::fill(t.begin(), t.end(), t0);
        thrust::fill(h.begin(), h.end(), h0);
        thrust::fill(y.begin(), y.end(), y0);
    }

    auto begin() {
        return zip_tuple_iters(t.begin(), h.begin(), y.begin());
    }
    auto end() { return begin() + size; }
};
