#include "spread_avx512.h"

#include "../reference/spread_bin_sort_int.h"
#include "../reference/spread_blocked.h"
#include "../reference/synchronized_accumulate_reference.h"

#include "gather_fold_avx512.h"
#include "spread_bin_sort_int.h"

#include <finufft_spread_opts.h>

namespace finufft {
namespace spreading {

template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_avx512(finufft_spread_opts const &opts) {
    GatherFoldAvx512Functor gather_rescale{
        opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity};
    auto spread_supbroblem =
        get_subproblem_polynomial_avx512_functor<T, Dim>({opts.ES_beta, opts.nspread});
    auto accumulate_subgrid_factory = get_reference_block_locking_accumulator<T, Dim>();

    return SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_supbroblem),
        std::move(accumulate_subgrid_factory),
    };
}

namespace {

/** Compute grid size from cache size.
 *
 * Note, the current arrangement is only reasonable for 1D and 2D problems.
 * For 3D problems - probably better to target baseline size on L2 cache instead of L1.
 * Cache size current hard-coded to Skylake+ Xeon.
 * TODO: write auto-detection code for cache size (hwloc?).
 *
 */
template <typename T, std::size_t Dim> std::array<std::size_t, Dim> get_grid_size() {
    std::array<std::size_t, Dim> grid_size;

    std::size_t cache_size = 1 << 15;         // 32 kiB cache (L1D cache size)
    std::size_t element_size = sizeof(T) * 2; // 2x real numbers per point

    // Baseline size
    grid_size[0] = (1 << 15) / element_size;

    // Use 32-wide in the non-contiguous dimension.
    for (std::size_t i = 1; i < Dim; ++i) {
        grid_size[i] = 32;
        grid_size[0] /= 32;
    }

    // Adjust initial dimension if it becomes too small
    if (grid_size[0] < 32) {
        grid_size[0] = 32;
    }

    return grid_size;
}
} // namespace

template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> make_avx512_blocked_spread_functor(
    kernel_specification const &kernel, tcb::span<const std::size_t, Dim> target_size,
    FoldRescaleRange input_range, finufft::Timer const &timer) {
    return reference::make_packed_sort_spread_blocked<T, Dim>(
        avx512::get_sort_functor<T, Dim>(timer.make_timer("sp")),
        get_subproblem_polynomial_avx512_functor<T, Dim>(kernel),
        get_reference_block_locking_accumulator<T, Dim>(),
        input_range,
        target_size,
        get_grid_size<T, Dim>(),
        timer);
}

#define INSTANTIATE(T, Dim)                                                                        \
    template SpreadFunctorConfiguration<T, Dim> get_spread_configuration_avx512(                   \
        finufft_spread_opts const &opts);                                                          \
    template SpreadFunctor<T, Dim> make_avx512_blocked_spread_functor(                             \
        kernel_specification const &kernel,                                                        \
        tcb::span<const std::size_t, Dim>                                                          \
            target_size,                                                                           \
        FoldRescaleRange input_range,                                                              \
        finufft::Timer const &timer);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

#undef INSTANTIATE

template SpreadFunctorConfiguration<double, 1>
get_spread_configuration_avx512(finufft_spread_opts const &opts);
template SpreadFunctorConfiguration<double, 2>
get_spread_configuration_avx512(finufft_spread_opts const &opts);
template SpreadFunctorConfiguration<double, 3>
get_spread_configuration_avx512(finufft_spread_opts const &opts);

} // namespace spreading
} // namespace finufft
