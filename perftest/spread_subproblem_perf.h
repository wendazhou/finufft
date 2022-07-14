#pragma once

/** @file Utilities for testin performance of spreading subproblem.
 *
 * This file contains baseline implementations of performance tests
 * for the spreading subproblem. These benchmarks are instantiated
 * separately for different dimensions in their respective test
 * executables.
 *
 */

#include <benchmark/benchmark.h>
#include <cmath>

#include "../src/kernels/avx512/spread_avx512.h"
#include "../src/kernels/legacy/spread_subproblem_legacy.h"
#include "../src/kernels/reference/spread_subproblem_reference.h"
#include "../src/kernels/spreading.h"

#include "../src/kernels/dispatch.h"
#include "../test/spread_test_utils.h"

namespace fs = finufft::spreading;

namespace {

template <typename T, std::size_t Dim>
void benchmark_spread_subproblem(
    benchmark::State &state, fs::SpreadSubproblemFunctor<T, Dim> const &functor,
    std::size_t num_points, fs::grid_specification<Dim> const &grid_spec,
    bool sort_points = true) {

    auto input = make_spread_subproblem_input<T>(num_points, 0, grid_spec, functor.target_padding());

    if (sort_points) {
        sort_point_collection(input);
    }

    auto output = finufft::allocate_aligned_array<T>(2 * grid_spec.num_elements(), 64);

    for (auto _ : state) {
        benchmark::DoNotOptimize(input.strengths[0]);
        functor(input, grid_spec, output.get());
        benchmark::DoNotOptimize(output[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);
}

template <typename T, std::size_t Dim>
void benchmark_spread_subproblem(
    benchmark::State &state, fs::SpreadSubproblemFunctor<T, Dim> const &functor) {
    auto num_points = state.range(0);

    num_points = fs::round_to_next_multiple(num_points, functor.num_points_multiple());
    auto padding = functor.target_padding();

    // Currently use density with ~ 1 uniform point / 1 non-uniform point.
    auto extent_multiple = functor.extent_multiple();
    auto extent_constant = static_cast<std::size_t>(std::round(std::pow(num_points, 1.0 / Dim)));
    std::array<std::size_t, Dim> extent;
    std::fill_n(extent.begin(), Dim, extent_constant);
    for (std::size_t i = 0; i < Dim; ++i) {
        // Set the extent according to the requirements of the functor,
        // with the addition that the base range of the points within
        // the axis must be of size extent_constant.
        // This implies that the actual extent is padded and rounded
        // up to the correct multiple, slightly (correctly) penalizing methods
        // which require large amounts of padding or alignment.
        extent[i] = fs::round_to_next_multiple(
            static_cast<std::size_t>(
                std::ceil(extent[i] - padding[i].offset + padding[i].grid_right)),
            extent_multiple[i]);
    }

    // Fill grid specification
    fs::grid_specification<Dim> grid;
    std::fill(grid.offsets.begin(), grid.offsets.end(), 3);
    std::copy(extent.begin(), extent.end(), grid.extents.begin());

    benchmark_spread_subproblem<T, Dim>(state, functor, num_points, grid);
}

template <typename T, std::size_t Dim, typename Factory>
void benchmark_for_width(benchmark::State &state, Factory const &factory) {
    auto kernel_spec = specification_from_width(state.range(1), 2.0);
    benchmark_spread_subproblem<T, Dim>(state, factory(kernel_spec));
}

template <typename T, std::size_t Dim> void benchmark_legacy(benchmark::State &state) {
    benchmark_for_width<T, Dim>(state, [](fs::kernel_specification const &kernel_spec) {
        return fs::SpreadSubproblemLegacyFunctor<T, Dim>{kernel_spec};
    });
}

template <typename T, std::size_t Dim> void benchmark_direct(benchmark::State &state) {
    benchmark_for_width<T, Dim>(state, [](fs::kernel_specification const &kernel_spec) {
        return fs::SpreadSubproblemDirectReference<T, Dim>{kernel_spec};
    });
}

template <typename T, std::size_t Dim> void benchmark_reference(benchmark::State &state) {
    benchmark_for_width<T, Dim>(state, [](fs::kernel_specification const &kernel_spec) {
        return fs::get_subproblem_polynomial_reference_functor<T, Dim>(kernel_spec);
    });
}

template <typename T, std::size_t Dim> void benchmark_avx512(benchmark::State &state) {
    if (finufft::get_current_capability() < finufft::DispatchCapability::AVX512) {
        state.SkipWithError("AVX512 not supported");
        return;
    }

    benchmark_for_width<T, Dim>(state, [](fs::kernel_specification const &kernel_spec) {
        return fs::get_subproblem_polynomial_avx512_functor<T, Dim>(kernel_spec);
    });
}

} // namespace
