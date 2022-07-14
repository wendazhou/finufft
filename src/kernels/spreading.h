#pragma once

/** @file
 *
 * This file contains the main interfaces for assembling the spreading operation.
 * The spreading operation is divided into three parts:
 * - gather and rescale
 * - subproblem
 * - accumulate
 * Optimized implementation of each of those sub-operations are implemented
 * in the `kernels` folder.
 *
 * In practice, these components are assembled in "spreading_default.h"
 *
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include <function2/function2.h>

#define _USE_MATH_DEFINES
#include <cmath>
#undef _USE_MATH_DEFINES

#include "../memory.h"

namespace finufft {
namespace spreading {

#ifdef __cpp_lib_type_identity
template <typename T> using identity = std::type_identity<T>;
#else
template <typename T> struct identity { using type = T; };
#endif

/** This structure collects the parameters of the kernel to use when spreading.
 *
 * @var es_beta The exponent of the beta term in the kernel.
 * @var width The width of the kernel in grid units.
 *
 */
struct kernel_specification {
    double es_beta;
    int width;
};

/** This structure represents a grid of points, potentially within a larger grid.
 *
 */
template <std::size_t Dim> struct grid_specification {
    std::array<std::int64_t, Dim> offsets;
    std::array<std::size_t, Dim> extents;

    const std::int64_t num_elements() const {
        return std::accumulate(
            extents.begin(),
            extents.end(),
            static_cast<std::size_t>(1),
            std::multiplies<std::size_t>());
    }
};

/** This structure represents a collection of non-uniform points, and their associated complex
 * strengths.
 *
 * @tparam Dim The dimension of the points.
 * @tparam PtrT The pointer type used to store the points and strengths. Typically *float or
 *    *double, but may be a smart pointer type to manage memory ownership.
 *
 * @var num_points The number of points in the collection.
 * @var coordinates An array representing the coordinates of the points in each dimension
 * @var strengths An array representing the strengths of the points in complex interleaved format
 *
 */
template <std::size_t Dim, typename T> struct nu_point_collection {
    std::size_t num_points;
    std::array<T *, Dim> coordinates;
    T *strengths;

    nu_point_collection() : num_points(0), coordinates(), strengths(nullptr) {}
    nu_point_collection(std::size_t num_points, std::array<T *, Dim> coordinates, T *strengths)
        : num_points(num_points), coordinates(coordinates), strengths(strengths) {}
    nu_point_collection(nu_point_collection<Dim, std::remove_const_t<T>> const &other)
        : num_points(other.num_points), coordinates(), strengths(other.strengths) {
        std::copy(other.coordinates.begin(), other.coordinates.end(), coordinates.begin());
    }
};

/** This structure captures the exact information required to deduce
 * the location being written to by a spreading operation, even in the
 * presence of potential floating point errors, in a single dimension.
 *
 * The locations written to, for a given point at coordinate x represented
 * in type T, is computed in the following fashion:
 * - let xi = (int64_t)ceil(x + offset), where all computations are done in floating point precision
 * T
 * - the functor will write to (at most) locations [xi - grid_left, xi + grid_right).
 *
 * Note that this constraint may be violated at individual points (in particular,
 * vectorized implementations may write to the left of unaligned points without declaring
 * it in the write specification), but it is guaranteed that this holds for the convex hull
 * of the intervals of the points. (In the vectorized case, as the array is implicitly aligned
 * by using the index 0 as the start of the dimension after offset, further aligning the
 * write-out index can never bring it below 0). See also compute_subgrid.
 *
 */
template <typename T> struct KernelWriteSpec {
    T offset;
    int grid_left;
    int grid_right;

    /// Maximum valid value in the array, give a grid offset and size.
    T max_valid_value(std::int64_t grid_offset, std::size_t grid_size) const {
        // Slightly imprecise below, this value should be:
        // max_{x \in T} x - offset \leq grid_size + grid_offset - grid_right - 1
        return static_cast<T>(static_cast<int64_t>(grid_size) + grid_offset - grid_right - 1) +
               offset;
    }

    /// Minimum valid value in the array, give a grid offset and size.
    T min_valid_value(std::int64_t grid_offset, std::size_t grid_size) const {
        // Give ourselves a little extra space to write to.
        // Technically, this value is given by:
        // min_{x \in T} x - offset > grid_offset + grid_left - 1
        return static_cast<T>(grid_offset + grid_left) + offset;
    }
};

/** Main type-erased holder for subproblem functor.
 *
 * This is the main type-erased holder for performing the spreading subproblem.
 * It also defines the concept for providing custom implementations of the inner
 * loop of the spreader.
 *
 */
template <typename T, std::size_t Dim> class SpreadSubproblemFunctor {
  public:
    /** Main concept for subproblem implementations.
     *
     * In addition to the main operation, the subproblem must specify
     * three additional values corresponding to the various amounts
     * of padding required.
     *
     */
    struct Concept {
        virtual ~Concept() = default;

        /** Specify the requirement for the number of points in the subproblem input.
         *
         * Callers must ensure that the number of points in the input is divisible by the
         * specified number. Points should be padded by points within the problem of strengths
         * 0 as necessary to satisfy this requirement.
         */
        virtual std::size_t num_points_multiple() const = 0;
        /** Specify the requirement for the extent of the target buffer in each dimension.
         *
         * Callers must ensure that the extent of the target buffer in each dimension is divisible
         * by the corresponding specified value. The target buffer should be enlarged to satisfy
         * this requirement.
         */
        virtual std::array<std::size_t, Dim> extent_multiple() const = 0;
        /** Specify the requirement for the padding of the target buffer.
         *
         * See the documentation of KernelWriteSpec to understand the exact
         * computation of location written to.
         *
         */
        virtual std::array<KernelWriteSpec<T>, Dim> target_padding() const = 0;

        /** Performs the given subproblem operation.
         *
         * @param input The input data to the subproblem.
         * @param grid The grid specification of the target buffer.
         * @param output[out] The target buffer to write to.
         *
         */
        virtual void operator()(
            nu_point_collection<Dim, T const> const &input, grid_specification<Dim> const &grid,
            T *output) const = 0;
    };

    template <typename Impl> class Model : public Concept {
      private:
        Impl impl_;

      public:
        Model(Impl &&impl) : impl_(std::move(impl)) {}

        virtual std::size_t num_points_multiple() const override {
            return impl_.num_points_multiple();
        };
        virtual std::array<std::size_t, Dim> extent_multiple() const override {
            return impl_.extent_multiple();
        };
        virtual std::array<KernelWriteSpec<T>, Dim> target_padding() const override {
            return impl_.target_padding();
        };
        virtual void operator()(
            nu_point_collection<Dim, T const> const &input, grid_specification<Dim> const &grid,
            T *output) const override {
            return impl_(input, grid, output);
        }
    };

  private:
    std::unique_ptr<Concept> impl_;

  public:
    template <typename Impl>
    SpreadSubproblemFunctor(Impl &&impl)
        : impl_(std::make_unique<Model<Impl>>(std::forward<Impl>(impl))) {}

    std::size_t num_points_multiple() const { return impl_->num_points_multiple(); }
    std::array<std::size_t, Dim> extent_multiple() const { return impl_->extent_multiple(); }
    std::array<KernelWriteSpec<T>, Dim> target_padding() const { return impl_->target_padding(); }
    void operator()(
        nu_point_collection<Dim, T const> const &input, grid_specification<Dim> const &grid,
        T *__restrict output) const {
        impl_->operator()(input, grid, output);
    }
};

/** This functor represents a strategy for accumulating data into a target buffer.
 *
 * This functor implements synchronized accumulation of the data into
 * a target buffer. As some synchronization strategies may require allocation
 * which depends on the size of the target buffer, this functor assumes that
 * the target buffer and its size have already been provided.
 *
 * In order to initialize a reduction for a new target buffer, see the
 * `SynchronizedAccumulateFactory` trait.
 *
 * The parameters of the function are expected as follows:
 * - T const* input: The input data to the accumulation.
 * - grid_specification<Dim> const &grid: The grid specification specifying the logical
 *      location of the input data with respect to the target buffer.
 *
 */
template <typename T, std::size_t Dim>
using SynchronizedAccumulateFunctor =
    fu2::unique_function<void(T const *, grid_specification<Dim> const &) const>;

/** Factory to initialize an accumulation strategy.
 *
 * This functor implements functionality to initialize a given synchronized
 * accumulation strategy into a target buffer of the given size.
 * This is used to implement the final step of the spreading process where
 * the data is accumulated into the target buffer.
 *
 * The paramaters of the function are expected as follows:
 * - T* target: The target buffer to accumulate into.
 * - std::array<std::size_t, Dim> const& size: The size of the target buffer in each dimension.
 *
 */
template <typename T, std::size_t Dim>
using SynchronizedAccumulateFactory = fu2::unique_function<SynchronizedAccumulateFunctor<T, Dim>(
    T *, std::array<std::size_t, Dim> const &) const>;

/** Enum representing the range of the input data.
 *
 * @var Identity The input range is the same as the output range
 * @var Pi The input range is (-pi, pi).
 *
 */
enum class FoldRescaleRange { Identity, Pi };

/** This functor represents an implementation to gather and rescale the data.
 *
 * The parameters of the function are expected as follows:
 * - nu_point_collection<Dim, T> output: the collection to which to write the collected points
 * - nu_point_collection<Dim, const T> input: the collection from which to collect points
 * - std::array<int64_t, Dim> sizes: the size of the output for rescaling
 * - int64_t const* sort_index: indirect index to gather points
 *
 */
template <typename T, std::size_t Dim>
using GatherRescaleFunctor = fu2::unique_function<void(
    nu_point_collection<Dim, T> const &, nu_point_collection<Dim, const T> const &,
    std::array<int64_t, Dim>, int64_t const *) const>;

/** This structure groups the necessary sub-components of a spread implementation.
 *
 */
template <typename T, std::size_t Dim> struct SpreadFunctorConfiguration {
    GatherRescaleFunctor<T, Dim> gather_rescale;
    SpreadSubproblemFunctor<T, Dim> spread_subproblem;
    SynchronizedAccumulateFactory<T, Dim> make_synchronized_accumulate;
};

/** This function represents a processor for the spreading problem.
 * The processor is responsible for handling the multithreading and coordination
 * of the spreading process. The components of the computation are provided
 * to the processor through the `SpreadFunctorConfiguration` structure.
 *
 * The parameters of the function are expected as follows:
 * - SpreadFunctorConfiguration<T, Dim>: the specific implementations of the computation
 * - nu_point_collection<Dim, const T> input: the collection of non-uniform points to process
 * - int64_t const* sort_index: indirect index to gather points
 * - std::array<int64_t, Dim> const& sizes: the size of the output
 * - T* output: the target buffer to write the data to
 *
 */
template <typename T, std::size_t Dim>
using SpreadProcessor = fu2::unique_function<void(
    SpreadFunctorConfiguration<T, Dim> const &, nu_point_collection<Dim, const T> const &,
    int64_t const *, std::array<int64_t, Dim> const &, T *) const>;

/** This function represents a strategy to obtain an indirect sort
 * of the given points according to their bin index. The parameters
 * are expected as follows:
 *
 * - int64_t* sort_index: an array of size num_points, which will be
 *      filled with a permutation representing the indirect sort at
 *      the end of the function.
 * - std::size_t num_points: The number of points to sort
 * - std::array<T const*, Dim> coordinates: The coordinates of the points to sort
 * - std::array<T, Dim> extents: The virtual size of the target after rescaling
 * - std::array<T, Dim> bin_sizes: The bin size in each coordinate (in units after rescaling)
 * - FoldRescaleRange input_range: The range of the input data.
 *
 */
template <typename T, std::size_t Dim>
using BinSortFunctor = fu2::unique_function<void(
    int64_t *, std::size_t, std::array<T const *, Dim> const &, std::array<T, Dim> const &,
    std::array<T, Dim> const &, FoldRescaleRange) const>;

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
 * enough to contain all non-uniform points, padded as per the specification.
 * The computed subgrid satisfies that in each dimension, the coordinate lies in:
 *    [offset + padding.first, offset + extent - padding.second - 1]
 *
 *
 * @param M Number of points.
 * @param coordinates Array containing coordinates of non-uniform points.
 *      coordinates[i][j] contains the ith coordinate of the jth point.
 * @param padding_left The required amount of padding on the left side of the subgrid.
 * @param padding_right The required amount of padding on the right side of the subgrid.
 *
 */
template <std::size_t Dim, typename T>
grid_specification<Dim> compute_subgrid(
    std::size_t M, std::array<T const *, Dim> const &coordinates,
    std::array<KernelWriteSpec<T>, Dim> const &padding) {
    std::array<std::int64_t, Dim> offsets;
    std::array<std::size_t, Dim> sizes;

    for (int i = 0; i < Dim; ++i) {
        auto minmax = std::minmax_element(coordinates[i], coordinates[i] + M);
        auto min_val = *minmax.first;
        auto max_val = *minmax.second;

        // Note: must cast to ensure computation same as during spreading.
        // At large values of min_val / max_val there can be significant floating point errors
        // (especially in single precision).
        offsets[i] =
            static_cast<int64_t>(std::ceil(min_val - padding[i].offset) - padding[i].grid_left);
        sizes[i] = static_cast<size_t>(std::ceil(max_val - padding[i].offset)) +
                   padding[i].grid_right - offsets[i];
    }

    return {offsets, sizes};
}

// Utility function which rounds the given integer value to the next multiple.
template <typename T, typename U> T round_to_next_multiple(T v, U multiple) {
    return (v + multiple - 1) / multiple * multiple;
}

/** Input for the spreading sub-operation.
 *
 * This struct is used to track the inputs to the contiguous spreading memory operation.
 * It captures the wrapped and rescaled coordinates, as well as the complex strengths.
 * In order to avoid unnecessary memory allocations, a single allocation is made for
 * the coordinates and the strengths.
 *
 */
template <std::size_t Dim, typename T> struct SpreaderMemoryInput : nu_point_collection<Dim, T> {
    aligned_unique_array<T> data_;

    SpreaderMemoryInput(std::size_t num_points)
        : data_(allocate_aligned_array<T>(
              (Dim + 2) * round_to_next_multiple(num_points, 64 / sizeof(T)), 64)) {
        auto num_points_multiple = round_to_next_multiple(num_points, 64 / sizeof(T));

        this->num_points = num_points;
        for (std::size_t i = 0; i < Dim; ++i) {
            this->coordinates[i] = data_.get() + i * num_points_multiple;
        }

        this->strengths = data_.get() + Dim * num_points_multiple;
    }
    SpreaderMemoryInput(SpreaderMemoryInput const &) = delete;
    SpreaderMemoryInput(SpreaderMemoryInput &&) = default;
};

/** Utility structure which mimics an array with constant values.
 *
 */
template <typename T> struct ConstantArray {
    T value;

    template <std::size_t N> operator std::array<T, N>() const {
        std::array<T, N> result;
        std::fill_n(result.begin(), N, value);
        return result;
    }

    T operator[](std::size_t i) const { return value; }
};

/** This structure represents strengths distributed on a regular grid.
 *
 */
template <std::size_t Dim, typename T> struct SubgridData {
    //! Array containing the strengths in complex interleaved format.
    aligned_unique_array<T> strengths;
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
template <std::size_t Dim, typename T>
void pad_nu_point_collection(
    nu_point_collection<Dim, T> &points, std::size_t total_size,
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

} // namespace spreading
} // namespace finufft
