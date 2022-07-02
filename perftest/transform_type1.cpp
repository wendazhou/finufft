#include <array>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <finufft.h>

#include "../src/memory.h"
#include "../test/spread_test_utils.h"

namespace {

template <typename T, std::size_t Dim> struct fi_plan;
template <std::size_t Dim> struct fi_plan<float, Dim> {
    typedef finufftf_plan plan_t;
    plan_t plan_;

    fi_plan(std::array<int64_t, Dim> const &sizes, double tol, finufft_opts const *opts) : plan_() {
        finufftf_makeplan(
            1,
            sizes.size(),
            const_cast<int64_t *>(sizes.data()),
            0,
            1,
            tol,
            &plan_,
            const_cast<finufft_opts *>(opts));
    }

    ~fi_plan() noexcept { finufftf_destroy(plan_); }

    void set_points(int64_t num_points, std::array<float *, Dim> const &points) {
        finufftf_setpts(
            plan_,
            num_points,
            points[0],
            Dim > 1 ? points[1] : nullptr,
            Dim > 2 ? points[2] : nullptr,
            0,
            nullptr,
            nullptr,
            nullptr);
    }

    void execute(std::complex<float> const *strengths, std::complex<float> *output) {
        finufftf_execute(plan_, const_cast<std::complex<float> *>(strengths), output);
    }
};

template <typename T>
std::vector<std::complex<T>> make_random_strengths(std::size_t num_points, int seed) {
    std::minstd_rand rng(seed);
    std::uniform_real_distribution<T> dist(-1, 1);

    std::vector<std::complex<T>> strengths(num_points);

    for (std::size_t i = 0; i < num_points; ++i) {
        auto re = dist(rng);
        auto im = dist(rng);
        strengths[i] = {re, im};
    }

    return strengths;
}

template <typename T, std::size_t Dim> void benchmark_type1(benchmark::State &state) {
    std::int64_t n = state.range(0);

    std::array<int64_t, Dim> sizes;
    sizes.fill(n);

    int64_t num_modes = std::reduce(sizes.begin(), sizes.end(), 1, std::multiplies<int64_t>());

    fi_plan<T, Dim> plan(sizes, 1e-5, nullptr);

    auto points = make_random_point_collection<Dim, T>(n, 0, {-M_PI, M_PI});
    finufft::spreading::nu_point_collection<Dim, T> points_view = points;

    auto output = finufft::allocate_aligned_array<std::complex<T>>(num_modes, 64);

    for (auto _ : state) {
        plan.set_points(n, points_view.coordinates);
        plan.execute(reinterpret_cast<std::complex<T> *>(points_view.strengths), output.get());
        benchmark::DoNotOptimize(output[0]);
        benchmark::DoNotOptimize(points.strengths[0]);
        benchmark::DoNotOptimize(points.coordinates[0]);
        benchmark::ClobberMemory();
    }
}

} // namespace

BENCHMARK(benchmark_type1<float, 1>)->Arg(32678)->Unit(benchmark::kMillisecond);
BENCHMARK(benchmark_type1<float, 2>)->Arg(128)->Unit(benchmark::kMillisecond);
BENCHMARK(benchmark_type1<float, 3>)->Arg(32)->Unit(benchmark::kMillisecond);
