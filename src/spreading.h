#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#include <iostream>

#include <omp.h>

#include <finufft/defs.h>
#include <finufft_spread_opts.h>

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

/** This structure collects the parameters of the kernel to use when spreading.
 *
 */
struct kernel_specification {
    double es_c;
    double es_beta;
    int width;
};

template <std::size_t Dim> struct grid_specification {
    std::array<std::int64_t, Dim> offsets;
    std::array<std::int64_t, Dim> extents;

    const std::int64_t num_elements() const {
        return std::reduce(extents.begin(), extents.end(), 1, std::multiplies<>());
    }
};

/** This structure represents a collection of non-uniform points, and their associated complex
 * strengths.
 *
 * @tparam Dim The dimension of the points.
 * @tparam PtrT The pointer type used to store the points and strengths. Typically *float or
 * *double, but may be a smart pointer type to manage memory ownership.
 *
 */
template <std::size_t Dim, typename PtrT> struct nu_point_collection {
    std::size_t num_points;
    std::array<PtrT, Dim> coordinates;
    PtrT strengths;

    /** Converts the structure by casting each array.
     *
     * Utility function to create a new `nu_point_collection` where each of the
     * arrays have been processed by the given function. This can be used to
     * easily convert between various pointer and smart pointer types if needed.
     *
     */
    template <typename Fn>
    nu_point_collection<Dim, std::invoke_result_t<Fn, PtrT>> cast(Fn &&fn) const {
        nu_point_collection<Dim, std::invoke_result_t<Fn, PtrT>> result;
        result.num_points = num_points;
        for (std::size_t i = 0; i < Dim; ++i) {
            result.coordinates[i] = fn(coordinates[i]);
        }
        result.strengths = fn(strengths);
        return result;
    }
};

/** Utility functor to get the underlying raw pointer from a `std::unique_ptr`.
 *
 */
struct UniqueArrayToPtr {
    template <typename T, typename Deleter>
    T *operator()(std::unique_ptr<T[], Deleter> const &ptr) const {
        return ptr.get();
    }
};

/** Utility functor to get the underlying (const) raw pointer from a `std::unique_ptr`.
 *
 */
struct UniqueArrayToConstPtr {
    template <typename T, typename Deleter>
    const T *operator()(std::unique_ptr<T[], Deleter> const &ptr) const {
        return ptr.get();
    }
};

/** This structure represents the output information of the spreading operation.
 *
 * It specifies a set of non-uniform points, by their coordinates and their
 * complex values. Additionally, it specifies an indirect index which allows
 * for the points to be sorted.
 *
 */
template <std::size_t Dim, typename T>
struct spread_problem_input : nu_point_collection<Dim, const T> {
    std::int64_t const *sorted_idx;
};

/** Computes subgrid offsets and extents large enough to contain all locations.
 *
 * We compute a rectangular subgrid specified as offsets and sizes which is large
 * enough to contain all non-uniform points, padded up to half the kernel width.
 *
 * @param M Number of points.
 * @param coordinates Array containing coordinates of non-uniform points.
 *      coordinates[i][j] contains the ith coordinate of the jth point.
 * @param ns Width of the kernel in pixels.
 *
 */
template <std::size_t Dim, typename T>
grid_specification<Dim>
compute_subgrid(std::size_t M, std::array<T const *, Dim> const &coordinates, int ns) {
    std::array<std::int64_t, Dim> offsets;
    std::array<std::int64_t, Dim> sizes;

    T ns_half = static_cast<T>(ns) / 2;

    for (int i = 0; i < Dim; ++i) {
        auto minmax = std::minmax_element(coordinates[i], coordinates[i] + M);
        auto min_val = *minmax.first;
        auto max_val = *minmax.second;

        offsets[i] = std::ceil(min_val - ns_half);
        sizes[i] = static_cast<std::size_t>(std::ceil(max_val - ns_half)) - offsets[i] + ns;
    }

    return {offsets, sizes};
}

// Utility function which rounds the given integer value to the next multiple.
template <typename T, typename U> T round_to_next_multiple(T v, U multiple) {
    return (v + multiple - 1) / multiple * multiple;
}

/** Utility deleter which deletes memory allocated with an aligned new operation.
 *
 */
template <typename T> struct AlignedDeleter {
    std::size_t alignment;

    void operator()(T *ptr) const noexcept {
        ::operator delete(ptr, std::align_val_t(this->alignment));
    }
};

template <typename T> struct AlignedDeleter<T[]> {
    std::size_t alignment;

    void operator()(T *ptr) const noexcept {
        ::operator delete[](ptr, std::align_val_t(this->alignment));
    }
};

template <typename T> using aligned_unique_ptr = std::unique_ptr<T, AlignedDeleter<T>>;

/** Allocates an array of the given size with specified alignment (in bytes).
 *
 * @param size Number of elements in the array.
 * @param alignment Alignment of the array in bytes. Must be a power of 2.
 */
template <typename T>
aligned_unique_ptr<T[]> allocate_aligned_array(std::size_t size, std::size_t alignment) {
    std::size_t size_bytes = size * sizeof(T);
    size_bytes = round_to_next_multiple(size_bytes, alignment);
    size = size_bytes / sizeof(T);

    return aligned_unique_ptr<T[]>(
        new (std::align_val_t(alignment)) T[size], AlignedDeleter<T[]>{alignment});
}

template <std::size_t Dim, typename T>
std::array<aligned_unique_ptr<T[]>, Dim>
allocate_aligned_arrays(std::size_t size, std::size_t alignment) {
    std::array<aligned_unique_ptr<T[]>, Dim> arrays;
    for (int i = 0; i < Dim; ++i) {
        arrays[i] = allocate_aligned_array<T>(size, alignment);
    }

    return std::move(arrays);
}

template <typename T> using aligned_unique_array = aligned_unique_ptr<T[]>;

/** Input for the spreading sub-operation.
 *
 * This struct is used to track the inputs to the contiguous spreading memory operation.
 * It captures the wrapped and rescaled coordinates, as well as the complex strengths.
 *
 */
template <std::size_t Dim, typename T>
struct SpreaderMemoryInput : nu_point_collection<Dim, aligned_unique_array<T>> {
    SpreaderMemoryInput(std::size_t num_points)
        : nu_point_collection<Dim, aligned_unique_array<T>>(
              {num_points,
               allocate_aligned_arrays<Dim, T>(num_points, 64),
               allocate_aligned_array<T>(2 * num_points, 64)}) {}
    SpreaderMemoryInput(SpreaderMemoryInput const &) = delete;
    SpreaderMemoryInput(SpreaderMemoryInput &&) = default;

    std::array<T const *, Dim> get_coordinates() const {
        std::array<T const *, Dim> result;
        for (int i = 0; i < Dim; ++i) {
            result[i] = this->coordinates[i].get();
        }
        return result;
    }
};

template <typename T> struct FoldRescalePi {
    T operator()(T x, T extent) const {
        if (x < -M_PI) {
            x += M_2PI;
        } else if (x >= M_PI) {
            x -= M_2PI;
        }

        return (x + M_PI) * extent * M_1_2PI;
    }
};

template <typename T> struct FoldRescaleIdentity {
    T operator()(T x, T extent) const {
        if (x < 0) {
            x += extent;
        } else if (x >= extent) {
            x -= extent;
        }

        return x;
    }
};

enum class FoldRescaleRange { Identity, Pi };

/** Generic implementation of spread-rescale with parametrizable rescaling.
 *
 * See gather_and_fold for general API details.
 *
 */
template <std::size_t Dim, typename T, typename IdxT, typename RescaleFn>
void gather_and_fold_impl(
    SpreaderMemoryInput<Dim, T> const &memory, nu_point_collection<Dim, T const *> const &input,
    std::array<T, Dim> const &extent, IdxT const *sort_indices, RescaleFn &&fold_rescale) {

    for (std::size_t i = 0; i < memory.num_points; ++i) {
        auto idx = sort_indices[i];

        for (int j = 0; j < Dim; ++j) {
            memory.coordinates[j][i] = fold_rescale(input.coordinates[j][idx], extent[j]);
        }

        memory.strengths[2 * i] = input.strengths[2 * idx];
        memory.strengths[2 * i + 1] = input.strengths[2 * idx + 1];
    }
}

/** Collect non-uniform points by index into contiguous array, and rescales coordinates.
 *
 * To prepare for the spreading operation, non-uniform points are collected according
 * to the indirect sorting array. Additionally, the coordinates are processed to a normalized
 * format.
 *
 * @param memory Output memory which will be filled with the gathered points.
 *               This parameter additionally implicitly specifies the number of points to gather.
 * @param input Input points to be gathered.
 * @param sizes Extent of the spreading target. Used to rescale the coordinates.
 * @param sort_indices Indirect sorting array.
 * @param rescale_range Whether the input is expected to be 2pi-periodic or extent-periodic.
 *
 */
template <std::size_t Dim, typename IdxT, typename T>
void gather_and_fold(
    SpreaderMemoryInput<Dim, T> const &memory, nu_point_collection<Dim, T const *> const &input,
    std::array<int64_t, Dim> const &sizes, IdxT const *sort_indices,
    FoldRescaleRange rescale_range) {

    std::array<T, Dim> sizes_floating;
    std::copy(sizes.begin(), sizes.end(), sizes_floating.begin());

    if (rescale_range == FoldRescaleRange::Pi) {
        gather_and_fold_impl(memory, input, sizes_floating, sort_indices, FoldRescalePi<T>{});
    } else {
        gather_and_fold_impl(memory, input, sizes_floating, sort_indices, FoldRescaleIdentity<T>{});
    }
}

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
    opts.ES_c = kernel.es_c;
    opts.ES_halfwidth = (double)kernel.width / 2.0;

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
struct SpreadSubproblemReference {
    // Specifies that the number of points in the input must be a multiple of this value.
    // The caller is required to pad the input with points within the domain and zero strength.
    static const std::size_t num_points_multiple = 1;
    // Specifies that the size of the output subgrid must be a multiple of this value.
    // The caller is required to allocate enough memory for the output, and pass in a compatible
    // grid specification.
    static const std::size_t extent_multiple = 1;

    template <typename T>
    void operator()(
        nu_point_collection<1, T const *> const &input, grid_specification<1> const &grid,
        T *output, const kernel_specification &kernel) const {

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
        nu_point_collection<2, T const *> const &input, grid_specification<2> const &grid,
        T *output, const kernel_specification &kernel) const {

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
        nu_point_collection<3, T const *> const &input, grid_specification<3> const &grid,
        T *output, const kernel_specification &kernel) const {

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

const static SpreadSubproblemReference spread_subproblem_reference;

template <typename T>
void add_wrapped_subgrid(
    T const *input, T *output, grid_specification<1> const &subgrid,
    std::array<std::int64_t, 1> const &output_grid) {
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
void add_wrapped_subgrid(
    T const *input, T *output, grid_specification<2> const &subgrid,
    std::array<std::int64_t, 2> const &output_grid) {

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
void add_wrapped_subgrid(
    T const *input, T *output, grid_specification<3> const &subgrid,
    std::array<std::int64_t, 3> const &output_grid) {

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

/** This structure represents strengths distributed on a regular grid.
 *
 */
template <std::size_t Dim, typename T> struct SubgridData {
    //! Array containing the strengths in complex interleaved format.
    aligned_unique_ptr<T[]> strengths;
    //! Description of the subgrid.
    grid_specification<Dim> grid;
};

/** Pad the input strengths and coordinates between the current size and the given desired total
 * size.
 *
 * Note that this function uses the existing allocation: the arrays in `points` must have been
 * allocated to support the desired total size.
 *
 * @param points The collection of non-uniform points to pad.
 * @param total_size The final size of the padded collection.
 * @param pad_coordinate The coordinate to use for padding.
 *
 */
template <std::size_t Dim, typename PtrT, typename T>
void pad_nu_point_collection(
    nu_point_collection<Dim, PtrT> &points, std::size_t total_size,
    std::array<T, Dim> const &pad_coordinate) {
    for (std::size_t i = points.num_points; i < total_size; ++i) {
        for (std::size_t j = 0; j < Dim; ++j) {
            points.coordinates[j][i] = pad_coordinate[j];
        }
        points.strengths[2 * i] = 0;
        points.strengths[2 * i + 1] = 0;
    }

    points.num_points = total_size;
}

template <std::size_t Dim, typename T, typename IdxT>
SubgridData<Dim, T> spread_block(
    IdxT const *sort_indices, std::array<std::int64_t, Dim> const &sizes, std::size_t num_points,
    std::array<T const *, Dim> const &coordinates, T const *strengths, T *output,
    const kernel_specification &kernel, FoldRescaleRange range) {

    // subproblem implementation being used (TODO: make parameter).
    auto spread_subproblem = spread_subproblem_reference;

    // round up to required number of points
    auto num_points_padded =
        round_to_next_multiple(num_points, spread_subproblem.num_points_multiple);

    SpreaderMemoryInput<Dim, T> memory(num_points_padded);
    auto memory_reference = memory.cast(UniqueArrayToConstPtr{});

    // Set number of points for now to the values to be provided
    // memory.num_points = num_points;
    gather_and_fold(memory, {num_points, coordinates, strengths}, sizes, sort_indices, range);

    // Compute subgrid for given set of points.
    auto subgrid = compute_subgrid<Dim, T>(num_points, memory_reference.coordinates, kernel.width);
    // Round up subgrid extent to required multiple for subproblem implementation.
    std::transform(
        subgrid.extents.begin(), subgrid.extents.end(), subgrid.extents.begin(), [&](auto x) {
            return round_to_next_multiple(x, spread_subproblem.num_points_multiple);
        });

    // Pad the input points to the required multiple, using a pad coordinate derived from the
    // subgrid. The pad coordinate is given by the leftmost valid coordinate in the subgrid.
    {
        std::array<T, Dim> pad_coordinate;
        std::transform(
            subgrid.offsets.begin(), subgrid.offsets.end(), pad_coordinate.begin(), [&](auto x) {
                return x + static_cast<T>(0.5 * kernel.width);
            });
        pad_nu_point_collection(memory, num_points_padded, pad_coordinate);
    }

    auto output_size = 2 * subgrid.num_elements();
    auto spread_weights = allocate_aligned_array<T>(output_size, 64);

    spread_subproblem_reference(memory_reference, subgrid, spread_weights.get(), kernel);
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

    if (nb * opts.max_subproblem_size < num_points) {  // ...or more subprobs to cap size
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

    std::mutex reduction_mutex;

#pragma omp parallel for schedule(dynamic, 1) // each is big
    for (int isub = 0; isub < nb; isub++) {   // Main loop through the subproblems
        std::size_t num_points_block =
            breaks[isub + 1] - breaks[isub]; // # NU pts in this subproblem
        SubgridData<Dim, T> block = spread_block(
            sort_indices + breaks[isub],
            sizes,
            num_points_block,
            coordinates,
            strengths,
            output,
            {opts.ES_c, opts.ES_beta, opts.nspread},
            opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity);

        {
            // Simple locked reduction strategy for now
            std::scoped_lock lock(reduction_mutex);
            add_wrapped_subgrid(block.strengths.get(), output, block.grid, sizes);
        }
    }
}

} // namespace spreading
} // namespace finufft
