#pragma once

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../spreading.h"

namespace finufft {
namespace spreading {

/** Standard templatized implementation of spreading for a given block.
 *
 */
template <
    std::size_t Dim, typename T, typename IdxT, typename SubproblemFn, typename GatherRescaleFn>
SubgridData<Dim, T> spread_block_impl(
    std::size_t num_points, IdxT const *sort_indices,
    nu_point_collection<Dim, const T> const &input, std::array<std::int64_t, Dim> const &sizes,
    T *output, SubproblemFn const &spread_subproblem, GatherRescaleFn const &gather_rescale) {

    // round up to required number of points
    auto num_points_padded =
        round_to_next_multiple(num_points, spread_subproblem.num_points_multiple());

    SpreaderMemoryInput<Dim, T> memory(num_points_padded);
    nu_point_collection<Dim, const T> memory_reference(memory);

    // Temporarily reduce number of points
    memory.num_points = num_points;
    gather_rescale(memory, input, sizes, sort_indices);

    // Compute subgrid for given set of points.
    auto padding = spread_subproblem.target_padding();
    auto subgrid = compute_subgrid<Dim, T>(memory.num_points, memory_reference.coordinates, padding);
    // Round up subgrid extent to required multiple for subproblem implementation.
    auto extent_multiple = spread_subproblem.extent_multiple();
    for (std::size_t i = 0; i < Dim; ++i) {
        subgrid.extents[i] = round_to_next_multiple(subgrid.extents[i], extent_multiple[i]);
    }

    // Pad the input points to the required multiple, using a pad coordinate derived from the
    // subgrid. The pad coordinate is given by the leftmost valid coordinate in the subgrid.
    {
        std::array<T, Dim> pad_coordinate;
        for (std::size_t i = 0; i < Dim; ++i) {
            pad_coordinate[i] = padding[i].min_valid_value(subgrid.offsets[i], subgrid.extents[i]);
        }
        pad_nu_point_collection(memory, num_points_padded, pad_coordinate);
    }

    auto output_size = 2 * subgrid.num_elements();
    auto spread_weights = allocate_aligned_array<T>(output_size, 64);

    spread_subproblem(memory, subgrid, spread_weights.get());
    return {std::move(spread_weights), subgrid};
}

/** Process by block with a single thread.
 *
 * This processor performs no multithreading and simply
 * processes the input points in order. If specified, it
 * may still block up sub-problems, which may be useful
 * for cache efficiency.
 *
 */
struct SingleThreadedProcessor {
    std::size_t max_subproblem_size_;

    SingleThreadedProcessor(
        int /* compatibility with OmpProcessor, unused here */, std::size_t max_subproblem_size)
        : max_subproblem_size_{max_subproblem_size} {}

    template <typename T, std::size_t Dim>
    void operator()(
        SpreadFunctorConfiguration<T, Dim> const &config,
        nu_point_collection<Dim, typename identity<T>::type const> const &input,
        int64_t const *sort_index, std::array<int64_t, Dim> const &sizes, T *output) const {

        auto total_size =
            std::reduce(sizes.begin(), sizes.end(), static_cast<int64_t>(1), std::multiplies<>{});
        std::fill_n(output, 2 * total_size, 0);

        auto max_num_threads = static_cast<std::size_t>(1);

        std::size_t num_blocks = std::min({max_num_threads, input.num_points});
        if (num_blocks * max_subproblem_size_ < input.num_points) {
            num_blocks = 1 + (input.num_points - 1) / max_subproblem_size_;
        }

        std::vector<std::size_t> breaks(num_blocks + 1);
        for (int p = 0; p <= num_blocks; ++p) {
            breaks[p] = (std::size_t)(0.5 + input.num_points * p / (double)num_blocks);
        }

        // TODO: check which dtype we wish to keep
        std::array<std::size_t, Dim> sizes_unsigned;
        std::copy(sizes.begin(), sizes.end(), sizes_unsigned.begin());
        auto accumulate_subgrid = config.make_synchronized_accumulate(output, sizes_unsigned);

        for (std::size_t b = 0; b < num_blocks; ++b) {
            std::size_t num_points_block = breaks[b + 1] - breaks[b];

            SubgridData<Dim, T> block = spread_block_impl<Dim, T>(
                num_points_block,
                sort_index + breaks[b],
                input,
                sizes,
                output,
                config.spread_subproblem,
                config.gather_rescale);

            accumulate_subgrid(block.strengths.get(), block.grid);
        }
    }
};

#ifdef _OPENMP
/** OpenMP multi-threaded processor.
 *
 * This spreading processor parallelizes the problem in blocks
 * of points according to the sorted point order.
 *
 */
struct OmpSpreadProcessor {
    int num_threads_;
    std::size_t max_subproblem_size_;

    template <typename T, std::size_t Dim>
    void operator()(
        SpreadFunctorConfiguration<T, Dim> const &config,
        nu_point_collection<Dim, typename identity<T>::type const> const &input,
        int64_t const *sort_index, std::array<int64_t, Dim> const &sizes, T *output) const {

        auto total_size =
            std::reduce(sizes.begin(), sizes.end(), static_cast<int64_t>(1), std::multiplies<>{});
        std::fill_n(output, 2 * total_size, 0);

        auto max_num_threads = static_cast<std::size_t>(omp_get_num_threads());
        if (num_threads_ > 0) {
            max_num_threads = std::min(max_num_threads, static_cast<std::size_t>(num_threads_));
        }

        std::size_t num_blocks = std::min(max_num_threads, input.num_points);
        if (num_blocks * max_subproblem_size_ < input.num_points) {
            num_blocks = 1 + (input.num_points - 1) / max_subproblem_size_;
        }

        std::vector<std::size_t> breaks(num_blocks + 1);
        for (int p = 0; p <= num_blocks; ++p) {
            breaks[p] = (std::size_t)(0.5 + input.num_points * p / (double)num_blocks);
        }

        // TODO: check which dtype we wish to keep
        std::array<std::size_t, Dim> sizes_unsigned;
        std::copy(sizes.begin(), sizes.end(), sizes_unsigned.begin());
        auto accumulate_subgrid = config.make_synchronized_accumulate(output, sizes_unsigned);

#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t b = 0; b < num_blocks; ++b) {
            std::size_t num_points_block = breaks[b + 1] - breaks[b];

            SubgridData<Dim, T> block = spread_block_impl<Dim, T>(
                num_points_block,
                sort_index + breaks[b],
                input,
                sizes,
                output,
                config.spread_subproblem,
                config.gather_rescale);

            accumulate_subgrid(block.strengths.get(), block.grid);
        }
    }
};
#else
// When no OpenMP is available, we use a single-threaded processor.
typedef SingleThreadedProcessor OmpSpreadProcessor;
#endif

} // namespace spreading
} // namespace finufft
