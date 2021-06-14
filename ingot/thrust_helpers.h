namespace thrust {

namespace detail {
template<typename Func>
struct Applier {
    Func f;
    template<typename Tuple>
    CUDA_HOSTDEV auto operator()(Tuple&& t) {
        return apply(f, std::forward<Tuple>(t));
    }
};
} // namespace detail

template<typename Func>
CUDA_HOSTDEV auto apply_func(Func f) {
    return detail::Applier<Func>{f};
}

template<typename Iterator, typename Func>
decltype(auto) for_each_apply(Iterator&& begin, Iterator&& end, Func&& f) {
    return thrust::for_each(begin, end, apply_func(std::forward<Func>(f)));
}

} // namespace thrust
