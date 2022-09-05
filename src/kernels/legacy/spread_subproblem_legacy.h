#pragma once

#include <cassert>

#include <finufft/defs.h>
#include <finufft_spread_opts.h>

#include "../spreading.h"

#include "spread_legacy.h"

/** @file This file contains implementation for spreading subproblem which
 * delegate to the current implementation.
 *
 */

// Forward declaration of reference implementations.
namespace finufft {
namespace spreadinterp {
void spread_subproblem_1d(
    BIGINT off1, BIGINT size1, float *du0, BIGINT M0, float *kx0, float *dd0,
    const finufft_spread_opts &opts);
void spread_subproblem_2d(
    BIGINT off1, BIGINT off2, BIGINT size1, BIGINT size2, float *du0, BIGINT M0, float *kx0,
    float *ky0, float *dd0, const finufft_spread_opts &opts);
void spread_subproblem_3d(
    BIGINT off1, BIGINT off2, BIGINT off3, BIGINT size1, BIGINT size2, BIGINT size3, float *du0,
    BIGINT M0, float *kx0, float *ky0, float *kz0, float *dd0, const finufft_spread_opts &opts);

void spread_subproblem_1d(
    BIGINT off1, BIGINT size1, double *du0, BIGINT M0, double *kx0, double *dd0,
    const finufft_spread_opts &opts);
void spread_subproblem_2d(
    BIGINT off1, BIGINT off2, BIGINT size1, BIGINT size2, double *du0, BIGINT M0, double *kx0,
    double *ky0, double *dd0, const finufft_spread_opts &opts);
void spread_subproblem_3d(
    BIGINT off1, BIGINT off2, BIGINT off3, BIGINT size1, BIGINT size2, BIGINT size3, double *du0,
    BIGINT M0, double *kx0, double *ky0, double *kz0, double *dd0, const finufft_spread_opts &opts);
} // namespace spreadinterp
} // namespace finufft

namespace finufft {
namespace spreading {

/** This functor dispatches to the current implementation of subproblem spreading.
 *
 * The spreading spreading subproblem corresponds to the non-periodized single-threaded
 * spreading problem.
 *
 */
struct SpreadSubproblemLegacy {
    template <typename T>
    void operator()(
        nu_point_collection<1, T const> const &input, subgrid_specification<1> const &grid,
        T *output, const kernel_specification &kernel) const {

        assert(grid.is_contiguous());

        auto opts = legacy::construct_opts_from_kernel(kernel, 1);

        // Note: current implementation not const-correct, so we have to cast the const
        // specifier away to call the function. It does not actually write to the input.
        finufft::spreadinterp::spread_subproblem_1d(
            grid.offsets[0],
            grid.extents[0],
            output,
            input.num_points,
            const_cast<T *>(input.coordinates[0]),
            const_cast<T *>(input.strengths),
            opts);
    }

    template <typename T>
    void operator()(
        nu_point_collection<2, T const> const &input, subgrid_specification<2> const &grid,
        T *output, const kernel_specification &kernel) const {

        assert(grid.is_contiguous());

        auto opts = legacy::construct_opts_from_kernel(kernel, 2);
        finufft::spreadinterp::spread_subproblem_2d(
            grid.offsets[0],
            grid.offsets[1],
            grid.extents[0],
            grid.extents[1],
            output,
            input.num_points,
            const_cast<T *>(input.coordinates[0]),
            const_cast<T *>(input.coordinates[1]),
            const_cast<T *>(input.strengths),
            opts);
    }

    template <typename T>
    void operator()(
        nu_point_collection<3, T const> const &input, subgrid_specification<3> const &grid,
        T *output, const kernel_specification &kernel) const {

        assert(grid.is_contiguous());

        auto opts = legacy::construct_opts_from_kernel(kernel, 3);
        finufft::spreadinterp::spread_subproblem_3d(
            grid.offsets[0],
            grid.offsets[1],
            grid.offsets[2],
            grid.extents[0],
            grid.extents[1],
            grid.extents[2],
            output,
            input.num_points,
            const_cast<T *>(input.coordinates[0]),
            const_cast<T *>(input.coordinates[1]),
            const_cast<T *>(input.coordinates[2]),
            const_cast<T *>(input.strengths),
            opts);
    }
};

/** Dispatches to the current implementation of the subproblem
 * through the `SpreadSubproblemFunctor` interface.
 *
 */
template <typename T, std::size_t Dim> struct SpreadSubproblemLegacyFunctor {
    kernel_specification kernel;

    std::size_t num_points_multiple() const { return 1; }
    std::array<std::size_t, Dim> extent_multiple() const {
        std::array<std::size_t, Dim> result;
        result.fill(1);
        return result;
    }
    std::array<KernelWriteSpec<T>, Dim> target_padding() const {
        std::array<KernelWriteSpec<T>, Dim> result;
        result.fill({static_cast<T>(0.5 * kernel.width), 0, kernel.width});
        return result;
    }

    void operator()(
        nu_point_collection<Dim, typename identity<T>::type const> const &input,
        grid_specification<Dim> const &grid, T *output) const {
        SpreadSubproblemLegacy{}(input, grid, output, kernel);
    }
};

} // namespace spreading
} // namespace finufft
