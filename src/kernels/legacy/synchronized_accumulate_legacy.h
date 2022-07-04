#pragma once

#include <finufft/defs.h>

#include "../reference/synchronized_accumulate_reference.h"

// Forward declaration of reference implementations.
namespace finufft {
namespace spreadinterp {
void add_wrapped_subgrid(
    BIGINT offset1, BIGINT offset2, BIGINT offset3, BIGINT size1, BIGINT size2, BIGINT size3,
    BIGINT N1, BIGINT N2, BIGINT N3, float *data_uniform, float *du0);
void add_wrapped_subgrid(
    BIGINT offset1, BIGINT offset2, BIGINT offset3, BIGINT size1, BIGINT size2, BIGINT size3,
    BIGINT N1, BIGINT N2, BIGINT N3, double *data_uniform, double *du0);
void add_wrapped_subgrid_thread_safe(
    BIGINT offset1, BIGINT offset2, BIGINT offset3, BIGINT size1, BIGINT size2, BIGINT size3,
    BIGINT N1, BIGINT N2, BIGINT N3, float *data_uniform, float *du0);
void add_wrapped_subgrid_thread_safe(
    BIGINT offset1, BIGINT offset2, BIGINT offset3, BIGINT size1, BIGINT size2, BIGINT size3,
    BIGINT N1, BIGINT N2, BIGINT N3, double *data_uniform, double *du0);
} // namespace spreadinterp
} // namespace finufft

namespace finufft {
namespace spreading {

struct LegacyAccumulateWrappedSubgrid {
    template <typename T, std::size_t Dim>
    void operator()(
        T const *input, grid_specification<Dim> const &subgrid, T *output,
        std::array<std::size_t, Dim> const &output_grid) const {

        static_assert(Dim >= 1, "Dimension must be at least 1");
        static_assert(Dim <= 3, "Legacy add wrapped subgrid only supports up to dimension 3.");

        finufft::spreadinterp::add_wrapped_subgrid(
            subgrid.offsets[0],
            Dim > 1 ? subgrid.offsets[1] : 0,
            Dim > 2 ? subgrid.offsets[2] : 0,
            subgrid.extents[0],
            Dim > 1 ? subgrid.extents[1] : 1,
            Dim > 2 ? subgrid.extents[2] : 1,
            output_grid[0],
            Dim > 1 ? output_grid[1] : 1,
            Dim > 2 ? output_grid[2] : 1,
            output,
            const_cast<T *>(input));
    }
};

struct LegacyThreadSafeAccumulateWrappedSubgrid {
    template <typename T, std::size_t Dim>
    void operator()(
        T const *input, grid_specification<Dim> const &subgrid, T *output,
        std::array<std::size_t, Dim> const &output_grid) const {

        static_assert(Dim >= 1, "Dimension must be at least 1");
        static_assert(Dim <= 3, "Legacy add wrapped subgrid only supports up to dimension 3.");

        finufft::spreadinterp::add_wrapped_subgrid_thread_safe(
            subgrid.offsets[0],
            Dim > 1 ? subgrid.offsets[1] : 0,
            Dim > 2 ? subgrid.offsets[2] : 0,
            subgrid.extents[0],
            Dim > 1 ? subgrid.extents[1] : 1,
            Dim > 2 ? subgrid.extents[2] : 1,
            output_grid[0],
            Dim > 1 ? output_grid[1] : 1,
            Dim > 2 ? output_grid[2] : 1,
            output,
            const_cast<T *>(input));
    }
};

template <typename T, std::size_t Dim>
SynchronizedAccumulateFactory<T, Dim> get_legacy_locking_accumulator() {
    return make_lambda_synchronized_accumulate_factory<T, Dim>(
        [](T *output, std::array<std::size_t, Dim> const &sizes) {
            return SynchronizedAccumulateFunctor<T, Dim>(
                GlobalLockedSynchronizedAccumulate<T, Dim, LegacyAccumulateWrappedSubgrid>(
                    output, sizes));
        });
}

template <typename T, std::size_t Dim>
SynchronizedAccumulateFactory<T, Dim> get_legacy_singlethreaded_accumulator() {
    return make_lambda_synchronized_accumulate_factory<T, Dim>(
        [](T *output, std::array<std::size_t, Dim> const &sizes) {
            return SynchronizedAccumulateFunctor<T, Dim>(
                NonLockedSynchronizedAccumulate<T, Dim, LegacyAccumulateWrappedSubgrid>{
                    output, sizes});
        });
}

template <typename T, std::size_t Dim>
SynchronizedAccumulateFactory<T, Dim> get_legacy_atomic_accumulator() {
    return make_lambda_synchronized_accumulate_factory<T, Dim>(
        [](T *output, std::array<std::size_t, Dim> const &sizes) {
            return SynchronizedAccumulateFunctor<T, Dim>(
                NonLockedSynchronizedAccumulate<T, Dim, LegacyThreadSafeAccumulateWrappedSubgrid>{
                    output, sizes});
        });
}

} // namespace spreading
} // namespace finufft
