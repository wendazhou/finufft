/** Experimental benchmark to demonstrate performance for cache-aligned
 * local block accumulation.
 *
 * Currently, this benchmark demonstrates that when ensuring that local
 * block accumulation is cache-aligned, the main performance bottleneck
 * is the gather stage, which becomes more expensive as the number of
 * points increases.
 *
 */

#include <benchmark/benchmark.h>

#include <algorithm>

#include <finufft_spread_opts.h>

#include "../src/kernels/spreading.h"
#include "../src/tracing.h"

#include "../src/kernels/reference/spread_bin_sort_int.h"
#include "../src/kernels/reference/spread_bin_sort_reference.h"
#include "../src/kernels/reference/spread_processor_reference.h"

#include "../src/kernels/avx512/spread_avx512.h"

#include <tcb/span.hpp>

#include "../test/spread_test_utils.h"

namespace {

using finufft::spreading::IntGridBinInfo;

template <typename T>
std::vector<finufft::spreading::grid_specification<1>>
make_bin_grids(IntGridBinInfo<T, 1> const &info) {
    std::vector<finufft::spreading::grid_specification<1>> grids(info.num_bins_total());

    for (std::size_t i = 0; i < info.num_bins[0]; ++i) {
        grids[i].extents[0] = info.grid_size[0];
        grids[i].offsets[0] = info.global_offset[0] + i * info.bin_size[0];
    }

    return grids;
}

template <typename T>
std::vector<finufft::spreading::grid_specification<2>>
make_bin_grids(IntGridBinInfo<T, 2> const &info) {
    std::vector<finufft::spreading::grid_specification<2>> grids(info.num_bins_total());

    std::size_t idx = 0;

    for (std::size_t j = 0; j < info.num_bins[1]; ++j) {
        for (std::size_t i = 0; i < info.num_bins[0]; ++i) {
            grids[idx].extents[0] = info.grid_size[0];
            grids[idx].extents[1] = info.grid_size[1];

            grids[idx].offsets[0] = info.global_offset[0] + i * info.bin_size[0];
            grids[idx].offsets[1] = info.global_offset[1] + j * info.bin_size[1];

            idx += 1;
        }
    }

    return grids;
}

/** Computation of bin grid index based on integer grid indices.
 *
 */
template <typename T, std::size_t Dim, typename FoldRescale>
void compute_bin_index_impl(
    std::size_t *index, std::size_t num_points, tcb::span<T const *const, Dim> coordinates,
    IntGridBinInfo<T, Dim> const &info, uint32_t bin_key_shift, FoldRescale &&fold_rescale) {

    auto compute_single = finufft::spreading::reference::ComputeBinIndexSingle<T, Dim, FoldRescale>(
        info, fold_rescale);

    for (std::size_t i = 0; i < num_points; ++i) {
        std::array<T, Dim> coords;
        for (std::size_t j = 0; j < Dim; ++j) {
            coords[j] = coordinates[j][i];
        }

        auto bin_index = compute_single(coords);
        index[i] = (bin_index << bin_key_shift) + i;
    }
}

template <typename T, std::size_t Dim>
std::vector<std::size_t> preprocess_points(
    int64_t *sort_idx, finufft::spreading::nu_point_collection<Dim, const T> const &input,
    IntGridBinInfo<T, Dim> const &info) {

    auto fold_rescale = finufft::spreading::FoldRescalePi<T>{};
    auto bin_key_shift = finufft::bit_width(input.num_points);

    // Compute all bin indexes
    compute_bin_index_impl<T, Dim>(
        reinterpret_cast<std::size_t *>(sort_idx),
        input.num_points,
        input.coordinates,
        info,
        bin_key_shift,
        fold_rescale);

    // Sort the bin indexes
    std::sort(sort_idx, sort_idx + input.num_points);

    // Count number of points in each bin
    std::vector<std::size_t> bin_counts(info.num_bins_total() + 1, 0);

    auto mask = (std::size_t(1) << bin_key_shift) - 1;
    for (std::size_t i = 0; i < input.num_points; ++i) {
        ++bin_counts[(sort_idx[i] >> bin_key_shift) + 1];
        sort_idx[i] &= mask;
    }

    std::partial_sum(bin_counts.begin(), bin_counts.end(), bin_counts.begin());

    return bin_counts;
}

struct SpreadBlockedTimers {
    SpreadBlockedTimers(finufft::Timer &timer)
        : gather(timer.make_timer("gather")), subproblem(timer.make_timer("subproblem")),
          accumulate(timer.make_timer("accumulate")) {}

    finufft::Timer gather;
    finufft::Timer subproblem;
    finufft::Timer accumulate;
};

void process_points(
    finufft::spreading::nu_point_collection<2, const float> const &input, std::size_t *sort_indices,
    std::size_t num_blocks, tcb::span<const finufft::spreading::grid_specification<2>> grids,
    tcb::span<const std::size_t> point_block_boundaries, std::array<std::size_t, 2> const &sizes,
    float *output, finufft::spreading::SpreadFunctorConfiguration<float, 2> const &config,
    SpreadBlockedTimers &timers) {

    typedef float T;
    const std::size_t Dim = 2;

    std::size_t max_grid_size = 0;
    std::size_t max_num_points = 0;

    for (std::size_t i = 0; i < num_blocks; ++i) {
        max_grid_size = std::max(max_grid_size, std::size_t(grids[i].num_elements()));
        max_num_points =
            std::max(max_num_points, point_block_boundaries[i + 1] - point_block_boundaries[i]);
    }

    std::array<int64_t, Dim> sizes_signed;
    std::copy(sizes.begin(), sizes.end(), sizes_signed.begin());

    auto subgrid_output = finufft::allocate_aligned_array<float>(2 * max_grid_size, 64);
    auto local_points = finufft::spreading::SpreaderMemoryInput<2, float>(max_num_points);
    auto const &padding_info = config.spread_subproblem.target_padding();

    auto const &accumulate_subgrid = config.make_synchronized_accumulate(output, sizes);

    for (std::size_t i = 0; i < num_blocks; ++i) {
        auto block_num_points = point_block_boundaries[i + 1] - point_block_boundaries[i];
        auto &grid = grids[i];

        local_points.num_points = block_num_points;

        // Gather local points
        {
            finufft::ScopedTimerGuard guard(timers.gather);
            config.gather_rescale(
                local_points,
                input,
                sizes_signed,
                reinterpret_cast<int64_t const *>(sort_indices + point_block_boundaries[i]));
        }

        auto num_points_padded = finufft::round_to_next_multiple(
            block_num_points, config.spread_subproblem.num_points_multiple());

        // Pad the input points to the required multiple, using a pad coordinate derived from the
        // subgrid. The pad coordinate is given by the leftmost valid coordinate in the subgrid.
        {
            std::array<T, Dim> pad_coordinate;
            for (std::size_t i = 0; i < Dim; ++i) {
                pad_coordinate[i] =
                    padding_info[i].min_valid_value(grid.offsets[i], grid.extents[i]);
            }
            finufft::spreading::pad_nu_point_collection(
                local_points, num_points_padded, pad_coordinate);
        }

        // Spread to local subgrid
        {
            // Zero local memory
            std::memset(subgrid_output.get(), 0, 2 * grid.num_elements() * sizeof(float));
            finufft::ScopedTimerGuard guard(timers.subproblem);
            config.spread_subproblem(local_points, grid, subgrid_output.get());
        }

        // Accumulate to main grid
        {
            finufft::ScopedTimerGuard guard(timers.accumulate);
            accumulate_subgrid(subgrid_output.get(), grid);
        }
    }
}

///! Debugging utility to check that all points are within the corresponding grid.
template <typename T, std::size_t Dim, typename FoldRescale>
std::size_t check_grid_boundaries(
    std::size_t *sort_idx, tcb::span<const std::size_t> block_boundaries,
    tcb::span<const finufft::spreading::grid_specification<Dim>> grids,
    finufft::spreading::nu_point_collection<Dim, const T> points, FoldRescale &&fold_rescale,
    IntGridBinInfo<T, Dim> const &info) {

    for (std::size_t dim = 0; dim < Dim; ++dim) {
        auto const &coordinates = points.coordinates[dim];
        auto padding = info.padding[dim];

        for (std::size_t b = 0; b < block_boundaries.size() - 1; ++b) {
            auto const &grid = grids[b];
            auto block_start = block_boundaries[b];
            auto block_end = block_boundaries[b + 1];

            for (std::size_t i = block_start; i < block_end; ++i) {
                auto coord = fold_rescale(coordinates[sort_idx[i]], static_cast<T>(info.size[dim]));
                auto coord_grid = std::ceil(coord - padding.offset);
                coord_grid -= grid.offsets[dim];

                if (coord_grid - padding.grid_left < 0 ||
                    coord_grid + padding.grid_right >= grid.extents[dim]) {
                    return i;
                }
            }
        }
    }

    return -1;
}

template <typename T, std::size_t Dim>
void apply_permutation(
    finufft::spreading::nu_point_collection<Dim, T> const &points, std::size_t *sort_idx) {
    auto coord_buffer = finufft::allocate_aligned_array<T>(points.num_points, 64);

    for (std::size_t dim = 0; dim < Dim; ++dim) {
        // Gather according to sorted index
        for (std::size_t i = 0; i < points.num_points; ++i) {
            coord_buffer[i] = points.coordinates[dim][sort_idx[i]];
        }

        // Copy back in sorted order
        std::memcpy(points.coordinates[dim], coord_buffer.get(), points.num_points * sizeof(T));
    }

    // Fill sort index with identity permutation
    std::iota(sort_idx, sort_idx + points.num_points, std::size_t(0));
}

void benchmark_spread_2d(benchmark::State &state, bool resolve_indirect_sort) {
    auto dim = state.range(0);
    auto num_points = dim * dim;

    auto points = make_random_point_collection<2, float>(num_points, 0, {-3 * M_PI, 3 * M_PI});
    std::array<std::size_t, 2> sizes;
    sizes.fill(dim);

    auto sort_idx = finufft::allocate_aligned_array<int64_t>(num_points, 64);

    auto output = finufft::allocate_aligned_array<float>(
        2 * std::accumulate(sizes.begin(), sizes.end(), size_t(1), std::multiplies<size_t>{}), 64);

    auto kernel_spec = specification_from_width(8, 2.0);

    finufft_spread_opts opts;
    opts.pirange = true;
    opts.ES_beta = kernel_spec.es_beta;
    opts.nspread = kernel_spec.width;

    auto config = finufft::spreading::get_spread_configuration_avx512<float, 2>(opts);

    std::size_t block_y = 32;
    std::size_t block_x = 4096 / block_y;

    std::array<std::size_t, 2> grid_size = {block_x, block_y};
    auto functor_padding = config.spread_subproblem.target_padding();

    IntGridBinInfo<float, 2> info(sizes, grid_size, functor_padding);

    auto bin_boundaries = preprocess_points<float, 2>(sort_idx.get(), points, info);

    auto grids = make_bin_grids(info);

    auto check = check_grid_boundaries<float, 2>(
        reinterpret_cast<std::size_t *>(sort_idx.get()),
        bin_boundaries,
        grids,
        points,
        finufft::spreading::FoldRescalePi<float>{},
        info);
    if (check != -1) {
        throw std::runtime_error("Invalid grid boundaries");
    }

    if (resolve_indirect_sort) {
        apply_permutation<float, 2>(points, reinterpret_cast<std::size_t *>(sort_idx.get()));
    }

    finufft::TimerRoot timer_root("bench");
    auto timer = timer_root.make_timer("spread");
    SpreadBlockedTimers timers(timer);

    for (auto _ : state) {
        process_points(
            points,
            reinterpret_cast<std::size_t *>(sort_idx.get()),
            bin_boundaries.size() - 1,
            grids,
            bin_boundaries,
            sizes,
            output.get(),
            config,
            timers);
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    auto timer_result = timer_root.report("spread");

    for (auto &name_and_time : timer_result) {
        auto name = std::get<0>(name_and_time);

        if (name.empty()) {
            continue;
        }

        auto time = std::chrono::duration<double>(std::get<1>(name_and_time)).count();
        if (time == 0) {
            continue;
        }

        state.counters[name] = benchmark::Counter(time, benchmark::Counter::kIsRate);
    }
}

void bm_spread_2d_indirect(benchmark::State &state) { benchmark_spread_2d(state, false); }

void bm_spread_2d_direct(benchmark::State &state) { benchmark_spread_2d(state, true); }

} // namespace

BENCHMARK(bm_spread_2d_indirect)
    ->Range(1 << 9, 1 << 11)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(bm_spread_2d_direct)
    ->Range(1 << 9, 1 << 11)
    ->RangeMultiplier(2)
    ->Unit(benchmark::kMillisecond);
