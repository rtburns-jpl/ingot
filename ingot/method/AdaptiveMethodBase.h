template<typename Derived>
class AdaptiveMethodBase {
    static constexpr double error_exponent() {
        return Frac{1, Derived::order};
    }
};
