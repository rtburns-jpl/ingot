#pragma once

// Did the sign of the a single component of the state vector change?
template<typename T, int N, int Idx>
struct Cartesian {
    CUDA_HOSTDEV
    T operator()(double t, double h, ingot::ColVal<T, N> const& u) const {
        static_assert(Idx < N, "index cannot exceed size of state vector");
        return u[Idx];
    }
};

// Distance from central body
struct DistanceFromB1 {
    double dist;
    double mu;
    CUDA_HOSTDEV
    double operator()(double t, double h, ingot::ColVal<double, 6> const& u) const {
        double dx = u[0] + mu;
        return dist - sqrt(dx*dx + u[1]*u[1] + u[2]*u[2]);
    }
};

// Distance from secondary body
struct DistanceFromB2 {
    double dist;
    double mu;
    CUDA_HOSTDEV
    double operator()(double t, double h, ingot::ColVal<double, 6> const& u) const {
        double dx = u[0] - (1 - mu);
        return dist - sqrt(dx*dx + u[1]*u[1] + u[2]*u[2]);
    }
};

// Distance from a fixed point
struct DistanceFromPoint {
    double x, y, z;
    double dist;
    double mu;
    CUDA_HOSTDEV
    double operator()(double t, double h, ingot::ColVal<double, 6> const& u) const {
        double dx = u[0] - x;
        double dy = u[1] - y;
        double dz = u[2] - z;
        return dist - sqrt(dx*dx + dy*dy + dz*dz);
    }
};

// Periapsis/apoapsis with respect to a fixed point
struct ApsisPoint {
    double x, y, z;
    double mu;
    CUDA_HOSTDEV
    double operator()(double t, double h, ingot::ColVal<double, 6> const& u) const {
        double dx = u[0] - x;
        double dy = u[1] - y;
        double dz = u[2] - z;
        // Dot product of offset from point with velocity
        return dx*u[3] + dy*u[4] + dz*u[5];
    }
};
