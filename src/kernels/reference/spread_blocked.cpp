#include "spread_blocked.h"

#include <cstring>
#include <vector>

#include "../sorting.h"
#include "spread_blocked_impl.h"

namespace finufft {
namespace spreading {
namespace reference {

namespace {

/** Create array of grid information for each bin.
 *
 */
template <typename T, std::size_t Dim>
std::vector<grid_specification<Dim>> make_bin_grids(IntGridBinInfo<T, Dim> const &info) {
    std::array<std::size_t, Dim> grid_extents;
    for (std::size_t i = 0; i < Dim; ++i) {
        grid_extents[i] = info.grid_size[0];
    }

    auto grid_fn = [&](auto... idxs) -> grid_specification<Dim> {
        std::array<std::size_t, Dim> idx{idxs...};
        grid_specification<Dim> grid;

        for (std::size_t i = 0; i < Dim; ++i) {
            grid.offsets[i] = info.global_offset[i] + idx[i] * info.bin_size[i];
        }

        std::copy(grid_extents.begin(), grid_extents.end(), grid.extents.begin());

        return grid;
    };

    std::vector<grid_specification<Dim>> grids(info.num_bins_total());
    generate_cartesian_grid<std::size_t, Dim>(grids.begin(), info.num_bins, std::move(grid_fn));

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
    LocalPointsBufferFactory<T, Dim> local_points_buffer_factory_;
    SpreadBlockedTimers timers_;

    void operator()(
        nu_point_collection<Dim, const T> const &input, IntGridBinInfo<T, Dim> const &info,
        std::size_t const *bin_boundaries, T *output) const {

        auto grids = make_bin_grids(info);

        auto grid_size = info.grid_size;
        {
            auto grid_size_multiple = spread_subproblem_.extent_multiple();
            for (std::size_t i = 0; i < Dim; ++i) {
                grid_size[i] = round_to_next_multiple(grid_size[i], grid_size_multiple[i]);
            }
        }

        std::size_t grid_size_total =
            std::accumulate(grid_size.begin(), grid_size.end(), 1, std::multiplies<std::size_t>());
        std::size_t max_num_points = 0;
        std::size_t num_blocks = info.num_bins_total();
        std::size_t target_num_points = std::accumulate(
            info.size.begin(), info.size.end(), std::size_t(1), std::multiplies<std::size_t>());

        // Determine amount of memory to allocate
        for (std::size_t i = 0; i < num_blocks; ++i) {
            max_num_points = std::max(max_num_points, bin_boundaries[i + 1] - bin_boundaries[i]);
        }

        auto const &padding_info = spread_subproblem_.target_padding();
        auto const &accumulate_subgrid = accumulate_factory_(output, info.size);

        // Number of pages to zero
        std::size_t page_size = 2 * 1024 * 1024; // 2 MB "Huge pages".
        std::size_t output_size_bytes = 2 * target_num_points * sizeof(T);
        auto num_pages = (output_size_bytes + page_size - 1) / page_size;

#pragma omp parallel
        {
            // Allocate per-thread local input and output buffer
            auto subgrid_output = finufft::allocate_aligned_array<T>(2 * grid_size_total, 64);
            auto local_points_buffer = local_points_buffer_factory_(max_num_points);

            SpreadBlockedTimers timers(timers_);

            // Zero out output buffer. Chunk by 2MB (page) to avoid unnecessary communication across
            // cores / NUMA nodes.
            {
                finufft::ScopedTimerGuard guard(timers.zero);

                std::size_t num_threads = omp_get_num_threads();
                std::size_t pages_per_thread = num_pages / num_threads;
                std::size_t remainder = num_pages % num_threads;

                std::size_t current_thread = omp_get_thread_num();
                std::size_t start_page =
                    (current_thread + std::min(current_thread, remainder)) * pages_per_thread;
                std::size_t num_pages = pages_per_thread + (current_thread < remainder ? 1 : 0);

                std::size_t remaining_bytes = output_size_bytes - start_page * page_size;
                remaining_bytes = std::min(remaining_bytes, num_pages * page_size);

                char *out = reinterpret_cast<char *>(output);
                std::memset(out + start_page * page_size, 0, remaining_bytes);

                // Barrier to make sure all threads have zeroed out the output buffer.
#pragma omp barrier
            }

#pragma omp for
            for (std::size_t i = 0; i < num_blocks; ++i) {
                // Set number of points for subproblem
                auto block_num_points = bin_boundaries[i + 1] - bin_boundaries[i];

                if (block_num_points == 0) {
                    // Skip empty bins
                    continue;
                }

                auto grid = grids[i];
                for (std::size_t j = 0; j < Dim; ++j) {
                    grid.extents[j] = grid_size[j];
                }

                // Gather local points
                auto const &local_points = [&]() {
                    finufft::ScopedTimerGuard guard(timers.gather);
                    return local_points_buffer(input, bin_boundaries[i], block_num_points, grid);
                }();

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
    SynchronizedAccumulateFactory<T, Dim> &&accumulate_factory, finufft::Timer const &timer) {
    return OmpSpreadBlockedImplementation<T, Dim>{
        std::move(spread_subproblem),
        std::move(accumulate_factory),
        make_default_points_buffer(spread_subproblem),
        SpreadBlockedTimers(timer)};
}

namespace {

template <typename T, std::size_t Dim> struct PackedSortBlockedSpreadFunctorImplementation {
    SortPointsFunctor<T, Dim> sort_points_;
    OmpSpreadBlockedImplementation<T, Dim> spread_blocked_;
    IntGridBinInfo<T, Dim> info_;
    finufft::Timer sort_timer_;
    finufft::Timer spread_timer_;
    FoldRescaleRange input_range_;

    PackedSortBlockedSpreadFunctorImplementation(
        SortPointsFunctor<T, Dim> &&sort_points,
        SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,
        SynchronizedAccumulateFactory<T, Dim> &&accumulate_factory, FoldRescaleRange input_range,
        tcb::span<const std::size_t, Dim> target_size, tcb::span<const std::size_t, Dim> grid_size,
        SpreadTimers const &timers)
        : sort_points_(std::move(sort_points)),
          spread_blocked_{
              std::move(spread_subproblem),
              std::move(accumulate_factory),
              make_default_points_buffer(spread_blocked_.spread_subproblem_),
              timers.spread_blocked_timers},
          info_(target_size, grid_size, spread_blocked_.spread_subproblem_.target_padding()),
          sort_timer_(timers.sort_packed), spread_timer_(timers.spread_blocked),
          input_range_(input_range) {}

    void operator()(nu_point_collection<Dim, const T> points, T *output) const {
        SpreaderMemoryInput<Dim, T> points_sorted(points.num_points);
        auto bin_counts = finufft::allocate_aligned_array<size_t>(info_.num_bins_total() + 1, 64);
        std::memset(bin_counts.get(), 0, (info_.num_bins_total() + 1) * sizeof(size_t));

        {
            finufft::Timer sort_timer(sort_timer_);
            finufft::ScopedTimerGuard guard(sort_timer);
            sort_points_(points, input_range_, points_sorted, bin_counts.get() + 1, info_);
        }

        // Compute bin boundaries
        {
            std::partial_sum(
                bin_counts.get(), bin_counts.get() + info_.num_bins_total() + 1, bin_counts.get());
        }

        {
            finufft::Timer spread_timer(spread_timer_);
            finufft::ScopedTimerGuard guard(spread_timer);

            auto num_values_output = std::accumulate(
                info_.size.begin(),
                info_.size.end(),
                std::size_t(1),
                std::multiplies<std::size_t>{});
            spread_blocked_(points_sorted, info_, bin_counts.get(), output);
        }
    }
};

} // namespace

template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> make_packed_sort_spread_blocked(
    SortPointsFunctor<T, Dim> &&sort_points, SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,
    SynchronizedAccumulateFactory<T, Dim> &&accumulate, FoldRescaleRange input_range,
    tcb::span<const std::size_t, Dim> target_size, tcb::span<const std::size_t, Dim> grid_size,
    finufft::Timer const &timer) {
    return PackedSortBlockedSpreadFunctorImplementation<T, Dim>(
        std::move(sort_points),
        std::move(spread_subproblem),
        std::move(accumulate),
        input_range,
        target_size,
        grid_size,
        SpreadTimers(timer));
}

#define INSTANTIATE(T, Dim)                                                                        \
    template SpreadBlockedFunctor<T, Dim> make_omp_spread_blocked(                                 \
        SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,                                       \
        SynchronizedAccumulateFactory<T, Dim> &&accumulate_factory,                                \
        finufft::Timer const &timers);                                                             \
    template SpreadFunctor<T, Dim> make_packed_sort_spread_blocked(                                \
        SortPointsFunctor<T, Dim> &&sort_points,                                                   \
        SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,                                       \
        SynchronizedAccumulateFactory<T, Dim> &&accumulate,                                        \
        FoldRescaleRange input_range,                                                              \
        tcb::span<const std::size_t, Dim>                                                          \
            target_size,                                                                           \
        tcb::span<const std::size_t, Dim>                                                          \
            grid_size,                                                                             \
        finufft::Timer const &timer);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

} // namespace reference
} // namespace spreading
} // namespace finufft
