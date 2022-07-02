#pragma once

#include <finufft/defs.h>

#include "kernels/legacy/spreading_legacy.h"
#include "kernels/legacy/synchronized_accumulate_legacy.h"
#include "kernels/reference/gather_fold_reference.h"
#include "spreading.h"

namespace finufft {
namespace spreading {

/** Standard templatized implementation of spreading for a given block.
 * 
 */
template <
    std::size_t Dim, typename T, typename IdxT, typename SubproblemFn, typename GatherRescaleFn>
SubgridData<Dim, T> spread_block_impl(
    IdxT const *sort_indices, nu_point_collection<Dim, const T> const &input,
    std::array<std::int64_t, Dim> const &sizes, T *output, FoldRescaleRange range,
    SubproblemFn const &spread_subproblem, GatherRescaleFn const &gather_rescale) {

    // round up to required number of points
    auto num_points_padded =
        round_to_next_multiple(input.num_points, spread_subproblem.num_points_multiple());

    SpreaderMemoryInput<Dim, T> memory(num_points_padded);
    nu_point_collection<Dim, const T> memory_reference(memory);

    gather_rescale(memory, input, sizes, sort_indices, range);

    // Compute subgrid for given set of points.
    auto padding = spread_subproblem.target_padding();
    auto subgrid = compute_subgrid<Dim, T>(input.num_points, memory_reference.coordinates, padding);
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
            pad_coordinate[i] = subgrid.offsets[i] + padding[i].first;
        }
        pad_nu_point_collection(memory, num_points_padded, pad_coordinate);
    }

    auto output_size = 2 * subgrid.num_elements();
    auto spread_weights = allocate_aligned_array<T>(output_size, 64);

    spread_subproblem(memory, subgrid, spread_weights.get());
    return {std::move(spread_weights), subgrid};
}

template <std::size_t Dim, typename T, typename IdxT>
inline void spread(
    IdxT const *sort_indices, std::array<std::int64_t, Dim> const &sizes, std::size_t num_points,
    std::array<T const *, Dim> const &coordinates, T const *strengths, T *output,
    const finufft_spread_opts &opts) {

    auto total_size = std::reduce(sizes.begin(), sizes.end(), 1, std::multiplies<>());
    std::fill_n(output, 2 * total_size, 0);

    auto nthr = MY_OMP_GET_NUM_THREADS();

    std::size_t nb = std::min(
        {static_cast<std::size_t>(nthr), num_points}); // simply split one subprob per thr...

    if (nb * opts.max_subproblem_size < num_points) { // ...or more subprobs to cap size
        nb = 1 + (num_points - 1) /
                     opts.max_subproblem_size; // int div does ceil(M/opts.max_subproblem_size)
        if (opts.debug)
            printf("\tcapping subproblem sizes to max of %d\n", opts.max_subproblem_size);
    }
    if (num_points * 1000 < total_size) { // low-density heuristic: one thread per NU pt!
        nb = num_points;
        if (opts.debug)
            printf("\tusing low-density speed rescue nb=M...\n");
    }

    std::vector<std::size_t> breaks(nb + 1); // NU index breakpoints defining nb subproblems
    for (int p = 0; p <= nb; ++p)
        breaks[p] = (std::size_t)(0.5 + num_points * p / (double)nb);

    auto accumulate_subgrid_factory = get_legacy_locking_accumulator<T, Dim>();
    std::array<std::size_t, Dim> sizes_unsigned;
    std::copy(sizes.begin(), sizes.end(), sizes_unsigned.begin());
    auto accumulate_subgrid = accumulate_subgrid_factory(output, sizes_unsigned);

    nu_point_collection<Dim, const T> input{num_points, coordinates, strengths};

    // subproblem implementation being used (TODO: make parameter).
    kernel_specification kernel_spec{opts.ES_beta, opts.nspread};
    auto spread_subproblem = SpreadSubproblemLegacyFunctor{kernel_spec};
    auto gather_rescale = &gather_and_fold<Dim, int64_t, T>;

#pragma omp parallel for schedule(dynamic, 1) // each is big
    for (int isub = 0; isub < nb; isub++) {   // Main loop through the subproblems
        std::size_t num_points_block =
            breaks[isub + 1] - breaks[isub]; // # NU pts in this subproblem
        SubgridData<Dim, T> block = spread_block_impl(
            sort_indices + breaks[isub],
            input,
            sizes,
            output,
            opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity,
            spread_subproblem,
            gather_rescale);

        accumulate_subgrid(block.strengths.get(), block.grid);
    }
}

} // namespace spreading
} // namespace finufft
