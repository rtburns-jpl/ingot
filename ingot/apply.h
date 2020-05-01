#include <tuple>
#include <utility>
#include <thrust/optional.h> // invoke

namespace thrust {
namespace detail {
template<typename F, typename Tuple, std::size_t... I>
CUDA_HOSTDEV constexpr decltype(auto) apply_impl(F&& f, Tuple&& t,
                                                 std::index_sequence<I...>) {
    return invoke(std::forward<F>(f), get<I>(std::forward<Tuple>(t))...);
}
} // namespace detail

template<typename F, typename Tuple>
CUDA_HOSTDEV constexpr decltype(auto) apply(F&& f, Tuple&& t) {
    return detail::apply_impl(
            std::forward<F>(f), std::forward<Tuple>(t),
            std::make_index_sequence<
                    tuple_size<std::remove_reference_t<Tuple>>::value>{});
}
} // namespace thrust
