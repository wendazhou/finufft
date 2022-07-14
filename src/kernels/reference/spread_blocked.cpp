#include "spread_blocked.h"

#include <cstring>
#include <vector>

#include "../sorting.h"

namespace finufft {
namespace spreading {
namespace reference {

namespace {

template <typename T>
std::vector<grid_specification<1>> make_bin_grids(IntGridBinInfo<T, 1> const& info) {
    std::vector<grid_specification<1>> grids(info.num_bins_total());

    for (std::size_t i = 0; i < info.num_bins[0]; ++i) {
        grids[i].extents[0] = info.grid_size[0];
        grids[i].offsets[0] = info.global_offset[0] + i * info.bin_size[0];
    }

    return grids;
}

template <typename T>
std::vector<grid_specification<2>> make_bin_grids(IntGridBinInfo<T, 2> const &info) {
    std::vector<grid_specification<2>> grids(info.num_bins_total());

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

/** Main implementation of blocked spreading, with parallelization through OpenMP.
 * 
 * This function assembles a subproblem functor and an accumulate functor to spread
 * the given set of non-uniform points. The parallelization is performed per block.
 * 
 */
template <typename T, std::size_t Dim> struct OmpSpreadBlockedImplementation {
    SpreadSubproblemFunctor<T, Dim> spread_subproblem_;
    SynchronizedAccumulateFactory<T, Dim> accumulate_factory_;
    SpreadBlockedTimers timers_;

    void operator()(
        nu_point_collection<Dim, const T> const &input, IntGridBinInfo<T, Dim> const &info,
        std::size_t const *bin_boundaries, T *output) const {

        auto grids = make_bin_grids(info);

        std::size_t grid_size = std::accumulate(
            info.grid_size.begin(), info.grid_size.end(), 1, std::multiplies<std::size_t>());
        std::size_t max_num_points = 0;
        std::size_t num_blocks = info.num_bins_total();

        // Determine amount of memory to allocate
        for (std::size_t i = 0; i < num_blocks; ++i) {
            max_num_points =
                std::max(max_num_points, bin_boundaries[i + 1] - bin_boundaries[i]);
        }

        auto const &padding_info = spread_subproblem_.target_padding();
        auto const &accumulate_subgrid = accumulate_factory_(output, info.size);

#pragma omp parallel
        {
            // Allocate per-thread local input and output buffer
            auto subgrid_output = finufft::allocate_aligned_array<T>(2 * grid_size, 64);
            auto local_points = finufft::spreading::SpreaderMemoryInput<Dim, T>(max_num_points);
            SpreadBlockedTimers timers(timers_);

#pragma omp for
            for (std::size_t i = 0; i < num_blocks; ++i) {
                // Set number of points for subproblem
                auto block_num_points = bin_boundaries[i + 1] - bin_boundaries[i];
                local_points.num_points = block_num_points;

                auto &grid = grids[i];

                // Zero local memory
                std::memset(subgrid_output.get(), 0, 2 * grid.num_elements() * sizeof(float));

                // Gather local points
                {
                    finufft::ScopedTimerGuard guard(timers.gather);

                    // Copy points to local buffer
                    std::memcpy(
                        local_points.strengths,
                        input.strengths + bin_boundaries[i],
                        block_num_points * sizeof(T) * 2);

                    for (std::size_t dim = 0; dim < Dim; ++dim) {
                        std::memcpy(
                            local_points.coordinates[dim],
                            input.coordinates[dim] + bin_boundaries[i],
                            block_num_points * sizeof(T));
                    }

                    auto num_points_padded = finufft::spreading::round_to_next_multiple(
                        block_num_points, spread_subproblem_.num_points_multiple());

                    // Pad the input points to the required multiple, using a pad coordinate derived
                    // from the subgrid. The pad coordinate is given by the leftmost valid
                    // coordinate in the subgrid.
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
                    finufft::ScopedTimerGuard guard(timers.subproblem);
                    spread_subproblem_(local_points, grid, subgrid_output.get());
                }

                // Accumulate to main grid
                {
                    finufft::ScopedTimerGuard guard(timers.accumulate);
                    accumulate_subgrid(subgrid_output.get(), grid);
                }
            }
        }
    }
};

} // namespace

template <typename T, std::size_t Dim>
SpreadBlockedFunctor<T, Dim> make_omp_spread_blocked(
    SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,
    SynchronizedAccumulateFactory<T, Dim> &&accumulate_factory,
    SpreadBlockedTimers const &timers_ref) {
    return OmpSpreadBlockedImplementation<T, Dim>{
        std::move(spread_subproblem), std::move(accumulate_factory), timers_ref};
}

#define INSTANTIATE(T, Dim)                                                                        \
    template SpreadBlockedFunctor<T, Dim> make_omp_spread_blocked(                                 \
        SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,                                       \
        SynchronizedAccumulateFactory<T, Dim> &&accumulate_factory,                                \
        SpreadBlockedTimers const &timers_ref);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)

} // namespace reference
} // namespace spreading
} // namespace finufft
