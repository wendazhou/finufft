#pragma once

#include <complex>
#include <cstddef>

#include "../test/spread_test_utils.h"
#include <finufft/finufft.hpp>

#include "../src/memory.h"

/** @file
 *
 * Utilities for benchmarking transforms end-to-end.
 *
 */

namespace {

template <typename T, std::size_t Dim> void benchmark_type1(benchmark::State &state) {
    std::int64_t n = state.range(0);

    std::array<int64_t, Dim> sizes;
    sizes.fill(n);

    int64_t num_modes = std::accumulate(
        sizes.begin(), sizes.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());

    finufft_opts opts;
    finufft_default_opts(&opts);

    opts.upsampfac = 2.0;

    finufft::nuft_plan_type1<T, Dim> plan(sizes, 1e-5, &opts);

    auto points = make_random_point_collection<Dim, T>(num_modes, 0, {-M_PI, M_PI});
    finufft::spreading::nu_point_collection<Dim, T> points_view = points;

    auto output = finufft::allocate_aligned_array<std::complex<T>>(num_modes, 64);

    for (auto _ : state) {
        plan.set_points(num_modes, points_view.coordinates);
        plan.execute(reinterpret_cast<std::complex<T> *>(points_view.strengths), output.get());
        benchmark::DoNotOptimize(output[0]);
        benchmark::DoNotOptimize(points.strengths[0]);
        benchmark::DoNotOptimize(points.coordinates[0]);
        benchmark::ClobberMemory();
    }
}

} // namespace
