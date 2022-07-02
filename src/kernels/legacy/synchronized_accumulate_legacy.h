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
} // namespace spreadinterp
} // namespace finufft

namespace finufft {
namespace spreading {

struct NonSynchronizedAccumulateWrappedSubgrid {
    template <typename T>
    void operator()(
        T const *input, grid_specification<1> const &subgrid, T *output,
        std::array<std::size_t, 1> const &output_grid) const {
        finufft::spreadinterp::add_wrapped_subgrid(
            subgrid.offsets[0],
            0,
            0,
            subgrid.extents[0],
            1,
            1,
            output_grid[0],
            1,
            1,
            output,
            const_cast<T *>(input));
    }

    template <typename T>
    void operator()(
        T const *input, grid_specification<2> const &subgrid, T *output,
        std::array<std::size_t, 2> const &output_grid) const {

        finufft::spreadinterp::add_wrapped_subgrid(
            subgrid.offsets[0],
            subgrid.offsets[1],
            0,
            subgrid.extents[0],
            subgrid.extents[1],
            1,
            output_grid[0],
            output_grid[1],
            1,
            output,
            const_cast<T *>(input));
    }

    template <typename T>
    void operator()(
        T const *input, grid_specification<3> const &subgrid, T *output,
        std::array<std::size_t, 3> const &output_grid) const {

        finufft::spreadinterp::add_wrapped_subgrid(
            subgrid.offsets[0],
            subgrid.offsets[1],
            subgrid.offsets[2],
            subgrid.extents[0],
            subgrid.extents[1],
            subgrid.extents[2],
            output_grid[0],
            output_grid[1],
            output_grid[2],
            output,
            const_cast<T *>(input));
    }
};

template <typename T, std::size_t Dim>
SynchronizedAccumulateFactory<T, Dim> get_legacy_locking_accumulator() {
    return make_lambda_synchronized_accumulate_factory<T, Dim>(
        [](T *output, std::array<std::size_t, Dim> const &sizes) {
            return SynchronizedAccumulateFunctor<T, Dim>(
                GlobalLockedSynchronizedAccumulate<T, Dim, NonSynchronizedAccumulateWrappedSubgrid>(
                    output, sizes));
        });
}

} // namespace spreading
} // namespace finufft
