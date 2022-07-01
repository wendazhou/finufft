#pragma once

#include <finufft/defs.h>
#include <finufft_spread_opts.h>

#include "../../spreading.h"

/** @file This file contains implementation for spreading which
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

/** Construct legacy options from kernel specification.
 *
 * In order to call reference implementation, we construct a legacy options struct
 * from the bare kernel specification. Note that the reverse engineering is not unique,
 * and depends on the specific implementation of setup_spreader.
 *
 */
inline finufft_spread_opts construct_opts_from_kernel(const kernel_specification &kernel) {
    finufft_spread_opts opts;

    opts.nspread = kernel.width;
    opts.kerevalmeth = 1;
    opts.kerpad = 1;
    opts.ES_beta = kernel.es_beta;
    opts.ES_c = 4.0 / (kernel.width * kernel.width);
    opts.ES_halfwidth = (double)kernel.width / 2.0;
    opts.flags = 0;

    // Reverse engineer upsampling factor from design in `setup_spreader`
    // This is necessary to call reference implementation, despite the fact
    // that the kernel is fully specified through the beta, c, and width parameters.
    double beta_over_ns = kernel.es_beta / kernel.width;

    // Baseline value of beta_over_ns for upsampling factor of 2
    double baseline_beta_over_ns = 2.3;
    switch (kernel.width) {
    case 2:
        baseline_beta_over_ns = 2.2;
    case 3:
        baseline_beta_over_ns = 2.26;
    case 4:
        baseline_beta_over_ns = 2.38;
    }

    if (std::abs(beta_over_ns - baseline_beta_over_ns) < 1e-6) {
        // Special-cased for upsampling factor of 2
        opts.upsampfac = 2;
    } else {
        // General formula, round obtained value in order to produce exact match if necessary.
        double upsamp_factor = 0.5 / (1 - beta_over_ns / (0.97 * M_PI));
        opts.upsampfac = std::round(upsamp_factor * 1000) / 1000;
    }

    return opts;
}

/** This functor dispatches to the current implementation of subproblem spreading.
 *
 * The spreading spreading subproblem corresponds to the non-periodized single-threaded
 * spreading problem.
 *
 */
struct SpreadSubproblemLegacy {
    // Specifies that the number of points in the input must be a multiple of this value.
    // The caller is required to pad the input with points within the domain and zero strength.
    static const std::size_t num_points_multiple = 1;
    // Specifies that the size of the output subgrid must be a multiple of this value.
    // The caller is required to allocate enough memory for the output, and pass in a compatible
    // grid specification.
    static const std::size_t extent_multiple = 1;

    template <typename T>
    void operator()(
        nu_point_collection<1, T const> const &input, grid_specification<1> const &grid, T *output,
        const kernel_specification &kernel) const {

        auto opts = construct_opts_from_kernel(kernel);

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
        nu_point_collection<2, T const> const &input, grid_specification<2> const &grid, T *output,
        const kernel_specification &kernel) const {

        auto opts = construct_opts_from_kernel(kernel);
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
        nu_point_collection<3, T const> const &input, grid_specification<3> const &grid, T *output,
        const kernel_specification &kernel) const {

        auto opts = construct_opts_from_kernel(kernel);
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
struct SpreadSubproblemLegacyFunctor {
    kernel_specification kernel;

    std::size_t num_points_multiple() const { return 1; }
    ConstantArray<std::size_t> extent_multiple() const { return {1}; }
    ConstantArray<std::pair<double, double>> target_padding() const {
        return {{0.5 * kernel.width, 0.5 * kernel.width}};
    }

    template <typename T, std::size_t Dim>
    void operator()(
        nu_point_collection<Dim, typename identity<T>::type const> const &input,
        grid_specification<Dim> const &grid, T *output) const {
        SpreadSubproblemLegacy{}(input, grid, output, kernel);
    }
};

} // namespace spreading
} // namespace finufft
