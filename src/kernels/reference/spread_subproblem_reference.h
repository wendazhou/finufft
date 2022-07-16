#pragma once

/** @file
 *
 * This file contains a reference implementation of the spreading subproblem,
 * based on a plain C++ implementation with no intrinsics. The performance
 * of this implementation is highly dependent on the compiler and hardware.
 *
 * Optimized implementations are provided for specific architectures in
 * other folders.
 *
 */

#include <algorithm>
#include <stdexcept>

#include "../../precomputed_poly_kernel_data.h"
#include "../spreading.h"
#include <tcb/span.hpp>

namespace finufft {
namespace spreading {

/** Direct evaluation of the exp-sqrt kernel.
 *
 * Note that due to the use of transcendental functions, this evaluation
 * is fairly slow, and polynomial approximations are more performant.
 *
 */
template <typename T> inline T evaluate_es_kernel_direct(T x, T beta) {
    auto s = std::sqrt(static_cast<T>(1.0) - x * x);
    return std::exp(beta * s);
}

///@{

template <typename T, std::size_t Dim> struct WriteSeparableKernelImpl;

/** Compute values of a separable kernel in arbitrary dimension and accumulates to the output array.
 *
 * This function computes the values of a separable kernel given as the tensor product of
 * 1-dimension kernels, and accumulates the values into the given output array. The output array is
 * assumed to be in column-major strided format, as an interleaved complex array. The kernel is
 * assumed to be real-valued, and multiplied by a complex scalar given through `k_re` and `k_im`.
 *
 * The implementation is provided in all dimensions using a recursive strategy in column-major order
 * from slowest to fastest dimension. In order to facilitate the recursive implementation, we make
 * use of an auxiliary template structure `WriteSeparableKernelImpl`.
 *
 * @param output[in, out] The output array.
 * @param strides Strides of the output array in each dimension.
 * @param values Values of the 1-dimension kernels in each dimension.
 * @param width Width of the kernel. It is assumed to be the same in each dimension.
 * @param k_re Real part of the complex scalar coefficient.
 * @param k_im Imaginary part of the complex scalar coefficient.
 *
 */
template <typename T, std::size_t Dim>
void write_separable_kernel(
    T __restrict *output, tcb::span<std::size_t, Dim> strides, tcb::span<T const *, Dim> values,
    std::size_t width) {
    return WriteSeparableKernelImpl<T, Dim>{}(
        output, strides, values, width);
}

template <typename T, std::size_t Dim> struct WriteSeparableKernelImpl {
    void operator()(
        T *__restrict output, tcb::span<std::size_t, Dim> strides, tcb::span<T const *, Dim> values,
        std::size_t width, T k = T(1)) const {
        for (std::size_t i = 0; i < width; ++i) {
            // Set-up values for the current slice in the slowest dimension
            auto k_i = k * values[Dim - 1][i];

            // Dispatch to accumulate in current slice.
            WriteSeparableKernelImpl<T, Dim - 1>{}(
                output + 2 * i * strides[Dim - 1],
                strides.template subspan<0, Dim - 1>(),
                values.template subspan<0, Dim - 1>(),
                width,
                k_i);
        }
    }
};

template <typename T> struct WriteSeparableKernelImpl<T, 1> {
    void operator()(
        T *__restrict output, tcb::span<std::size_t, 1> strides, tcb::span<T const *, 1> values,
        std::size_t width) {

        // Base case of the recursion: 1-D kernel accumulation.
        for (std::size_t i = 0; i < 2 * width; ++i) {
            output[i] += values[0][i];
        }
    }

    void operator()(
        T *__restrict output, tcb::span<std::size_t, 1> strides, tcb::span<T const *, 1> values,
        std::size_t width, T k) {

        // Base case of the recursion: 1-D kernel accumulation.
        for (std::size_t i = 0; i < 2 * width; ++i) {
            output[i] = std::fma(k, values[0][i], output[i]);
        }
    }
};

///@}

/** Generic implementation of point-by-point subproblem parametrized by kernel evaluation.
 *
 * This function implements a generic version of the point-by-point spreading subproblem,
 * evaluating the kernel in each dimension and accumulating the tensor outer product of those.
 * It is parametrized by the kernel evaluation function, which may be implemented as either
 * direct evaluation of the exp-sqrt function, or through some polynomial approximation strategy
 * for better performance.
 *
 * The kernel evaluation is required to be a grid-batched evaluation, where
 * a single call to the function evaluates the kernel at a grid of points separated
 * by 1 / kernel_width. The offset of that grid with respect to the center of the segment
 * is specified by the input to the kernel evaluation, as a floating point number in the
 * range [-1, 1].
 *
 * @param input Non-uniform points to spread.
 * @param grid Grid specification.
 * @param output[out] Output array. Must no alias any other input pointer.
 * @param kernel Evaluation function of the kernel. Must be callable with signature
 *     `void (T*, T)`, and additionally have a member `width` indicating the width of
 *      the output of the kernel. When called, the function is guaranteed that its
 *      first argument is a pointer to a contiguous array of length `width`. Note that
 *      this width may be larger than `kernel_width` of the kernel to enable padding
 *      and related optimizations.
 * @param kernel_width The width of the kernel being evaluated.
 *
 */
template <std::size_t Dim, typename T, typename Fn>
void spread_subproblem_generic_with_kernel(
    nu_point_collection<Dim, T const> const &input, grid_specification<Dim> const &grid,
    T *__restrict output, Fn &&kernel, std::size_t kernel_width) {
    std::fill_n(output, 2 * grid.num_elements(), T(0));

    // Allocate according to the width requested by the kernel.
    // To allow for pre-multiplication by real and imaginary strengths,
    // we allocate 2x the width of the kernel for the first dimension.
    auto kernel_values_stride = round_to_next_multiple(kernel.width, 8);
    auto kernel_values = allocate_aligned_array<T>(kernel_values_stride * (Dim + 1), 64);
    std::fill_n(kernel_values.get(), kernel_values_stride * (Dim + 1), T(0));

    T ns2 = static_cast<T>(0.5 * kernel_width);

    // Pre-compute strides for each array dimension (column-major format).
    std::array<std::size_t, Dim> strides;
    strides[0] = 1;
    for (std::size_t dim = 1; dim < Dim; ++dim) {
        strides[dim] = strides[dim - 1] * grid.extents[dim - 1];
    }

    // Pre-compute pointers into each segment containing the computed kernel values.
    std::array<T const *, Dim> kernel_values_view;
    for (std::size_t dim = 0; dim < Dim; ++dim) {
        kernel_values_view[dim] = kernel_values.get() + kernel_values_stride * (dim + (dim > 0));
    }

    for (std::size_t i = 0; i < input.num_points; ++i) {
        std::size_t point_total_offset = 0;

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            // Compute kernel values in each dimension
            auto x = input.coordinates[dim][i];
            // Compute integer grid index
            auto x_i = static_cast<int64_t>(std::ceil(x - ns2));
            // Compute offset in subgrid
            auto x_f = x_i - x;
            auto z = 2 * x_f + (kernel_width - 1);
            kernel(const_cast<T*>(kernel_values_view[dim]), z);

            point_total_offset += (x_i - grid.offsets[dim]) * strides[dim];
        }

        {
            // Pre-multiply kernel values by strengths
            // Note: pre-multiply in reverse to avoid overwriting kernel values.
            auto ker_val_ptr = kernel_values.get();
            for (std::size_t j = kernel_width - 1; j != std::size_t(-1); --j) {
                auto ker_val_j = ker_val_ptr[j];
                ker_val_ptr[2 * j] = ker_val_j * input.strengths[2 * i];
                ker_val_ptr[2 * j + 1] = ker_val_j * input.strengths[2 * i + 1];
            }
        }

        // Write out product kernel to output array.
        write_separable_kernel<T, Dim>(
            output + 2 * point_total_offset,
            strides,
            kernel_values_view,
            kernel_width);
    }
}

/** Functor implementing direct evaluation of the kernel at grid points.
 *
 * This function evaluates the exp-sqrt kernel with the given beta and c parameters
 * on an integer grid of the given width.
 *
 */
template <typename T> struct KernelDirectReference {
    T es_beta;
    std::size_t width;

    void operator()(T *output, T x) const {
        T z = x / width;

        for (std::size_t i = 0; i < width; ++i) {
            T center = 2 * (i + static_cast<T>(0.5)) / width - 1;
            output[i] = evaluate_es_kernel_direct(center + z, es_beta);
        }
    }
};

/** Reference implementation for subproblem with direct evaluation of exponential-sqrt kernel.
 *
 * This functor provides a direct naive subproblem implementation, by evaluating the exp-sqrt
 * kernel for each point and accumulating into the output.
 *
 */
template <typename T, std::size_t Dim> struct SpreadSubproblemDirectReference {
    kernel_specification kernel_;

    void operator()(
        nu_point_collection<Dim, typename identity<T>::type const> const &input,
        grid_specification<Dim> const &grid, T *__restrict output) const {
        KernelDirectReference<T> kernel_fn{
            static_cast<T>(kernel_.es_beta), static_cast<std::size_t>(kernel_.width)};
        spread_subproblem_generic_with_kernel(
            input, grid, output, kernel_fn, static_cast<std::size_t>(kernel_.width));
    }

    std::size_t num_points_multiple() const { return 1; }
    std::array<std::size_t, Dim> extent_multiple() const {
        std::array<std::size_t, Dim> result;
        result.fill(1);
        return result;
    }
    std::array<KernelWriteSpec<T>, Dim> target_padding() const {
        std::array<KernelWriteSpec<T>, Dim> result;
        result.fill({static_cast<T>(0.5 * kernel_.width), 0, kernel_.width});
        return result;
    }
};

// @{

/** Implementation of Horner scheme for polynomial evaluation through a recursive strategy.
 * This structure implements a recursive for polynomial evaluation.
 *
 * @param x The value to evaluate the polynomial at.
 * @param coeffs The coefficients of the polynomial in reverse order. Note: must be array of length
 * `degree + 1`.
 *
 * @tparam T The type of the evaluation.
 * @tparam Arr The type of the array of coefficients. Must produce a result compatible
 *    with the `T` type when indexed.
 *
 */
template <typename T, std::size_t Degree> struct HornerPolynomialEvaluation {
    template <typename Arr> T operator()(T x, Arr const &coeffs) const {
        // Note: enforce FMA to ensure bit-exact compatibility with vectorized implementations.
        // This is particularly important for fp32 arithmetic as the difference can be rather
        // significant.
        return std::fma(x, HornerPolynomialEvaluation<T, Degree - 1>{}(x, coeffs), coeffs[Degree]);
        // return x * HornerPolynomialEvaluation<T, Degree - 1>{}(x, coeffs) + coeffs[Degree];
    }
};

template <typename T> struct HornerPolynomialEvaluation<T, 0> {
    template <typename Arr> T operator()(T x, Arr const &coeffs) const { return coeffs[0]; }
};

// @}

/** Helper structure to provide a strided view over an array.
 *
 */
template <typename T> struct StridedArray {
    T *data;
    std::size_t stride;

    T &operator[](std::size_t i) { return data[i * stride]; }
    T const &operator[](std::size_t i) const { return data[i * stride]; }
};

/** Fills target batch polynomial coefficients from given source coefficients.
 *
 * This function fills the target polynomial coefficients in reverse order from the source
 * coefficients presented in standard order. More precisely, it considers the following:
 * - source_coefficients, a source_width x degree + 1 array of coefficients in column-major order,
 *   representing a set of polynomials of given degree such that source_coefficients[i, d] denotes
 *   the coefficient of x^d of the i-th polynomial.
 * - target_coefficients, a target_width x degree + 1 array of coefficients in column-major order,
 *   representing a set of polynomials of given degree such that target_coefficients[i, d] denotes
 *   the coefficient of x^(degree - d) of the i-th polynomial. Additionally, `target_stride` may
 *   be used to specify a stride larger than `target_width`.
 *
 * The target coefficients are filled from the source coefficients, and if the width of the target
 * is larger than the width of the source, the coefficients in the target are padded with zeros.
 *
 * @param degree The degree of the polynomials to copy.
 * @param source_coefficients The source coefficients in column-major order.
 * @param source_width The width of the source coefficients.
 * @param target_coefficients The target coefficients in strided column-major order.
 * @param target_width The width of the target coefficients.
 * @param target_stride The stride of the target coefficients (between each column).
 *
 */
template <typename ItS, typename ItT>
void fill_polynomial_coefficients(
    std::size_t degree, ItS source_coefficients, std::size_t source_width, ItT target_coefficients,
    std::size_t target_width, std::size_t target_stride = -1) {

    if (target_stride == -1) {
        target_stride = target_width;
    }

    for (std::size_t i = 0; i < degree + 1; ++i) {
        auto target_coeffs_i = target_coefficients + (degree - i) * target_stride;
        std::copy(
            source_coefficients + i * source_width,
            source_coefficients + (i + 1) * source_width,
            target_coeffs_i);
        std::fill(target_coeffs_i + source_width, target_coeffs_i + target_width, 0);
    }
};

/** Structure for evaluating a batch of polynomials of the given degree.
 *
 * In order to accelerate evaluation of the kernel, we may make use of
 * a polynomial approximation. This structure provides an implementation
 * for polynomial evaluation through Horner's method in a generic setting.
 *
 * @tparam T Floating point type used for evaluation.
 * @tparam Width The number of polynomials to evaluate.
 * @tparam Degree The degree of each polynomial.
 *
 */
template <typename T, std::size_t Width, std::size_t Degree> struct PolynomialBatch {
    /** Array of coefficients for each of the polynomial to evaluate.
     *
     * The array contains the coefficients with the faster dimension corresponding
     * to the width, and the slower dimension corresponding to the degree. Additionally,
     * the coefficients are stored in reverse order in the degree dimension, such that the
     * coefficient of the highest degree monomial is stored first.
     *
     */
    aligned_unique_array<T> coefficients;
    static const std::size_t width = Width;

    PolynomialBatch() : coefficients(allocate_aligned_array<T>(Width * (Degree + 1), 64)) {}

    /** Create a polynomial from the given coefficients.
     *
     * This constructor gathers the weights from the given array of coefficients,
     * expressed in standard order with the degree being the slower dimension.
     * Additionally, this may be used to expand the number of polynomials evaluated
     * by padding with zero coefficients beyond the specified width, in order
     * to facilitate vectorization of the kernel.
     *
     * @param coefficients Array of coefficients of the polynomial. Must be of length `width *
     * (degree + 1)`, and contain the coefficients with the width being the faster dimension and the
     * degree being the slower dimension.
     * @param width The width of the polynomial provided in the coefficients array. Must be less or
     * equal to Width.
     *
     */
    template <typename U>
    PolynomialBatch(U const *coefficients, std::size_t width = Width) : PolynomialBatch() {
        fill_polynomial_coefficients(Degree, coefficients, width, this->coefficients.get(), Width);
    }

    // Need copy-constructor for compatibility with std::function
    PolynomialBatch(PolynomialBatch const &other) : PolynomialBatch() {
        std::copy(
            other.coefficients.get(),
            other.coefficients.get() + Width * (Degree + 1),
            coefficients.get());
    };

    PolynomialBatch(PolynomialBatch &&) noexcept = default;

    void operator()(T *__restrict output, T x) const {
        for (std::size_t i = 0; i < Width; ++i) {
            output[i] = HornerPolynomialEvaluation<T, Degree>{}(
                x, StridedArray<T const>{coefficients.get() + i, Width});
        }
    }
};

/** Reference implementation for subproblem with polynomial approximation.
 *
 * This functor provides an implementation for the spreading subproblem
 * based on a polynomial approximation.
 *
 * @var kernel_width_ The width of the original kernel. May be smaller than the width of the
 *                    polynomial approximation.
 * @var kernel_polynomial_ The polynomial approximation to use for the kernel.
 *
 */
template <typename T, std::size_t Width, std::size_t Degree>
struct SpreadSubproblemPolynomialReference {
    std::size_t kernel_width_;
    PolynomialBatch<T, Width, Degree> kernel_polynomial_;

    template <std::size_t Dim>
    void operator()(
        nu_point_collection<Dim, typename identity<T>::type const> const &input,
        grid_specification<Dim> const &grid, T *__restrict output) const {
        return spread_subproblem_generic_with_kernel(
            input, grid, output, kernel_polynomial_, kernel_width_);
    }

    std::size_t num_points_multiple() const { return 1; }
    ConstantArray<std::size_t> extent_multiple() const { return {1}; }
    ConstantArray<KernelWriteSpec<T>> target_padding() const {
        // We only need to pad to the original kernel width,
        // rather than to the width of the polynomial evaluation.
        return {{static_cast<T>(0.5 * kernel_width_), 0, static_cast<int>(kernel_width_)}};
    }
};

namespace detail {

/** Helper class to create a polynomial implementation of the spreading subproblem
 * based on a set of existing kernels.
 *
 * As the kernels may vary in their width and degree, we need to identify the correct
 * concrete implementation of PolynomialBatch, based on the runtime parameters of the kernel.
 * This is done through a linear search of the supported kernel configurations (i.e. width x
 * degree). If no configuration is found, an exception is thrown.
 *
 * In order to support the linear search at compile time, we implement it in a recursive
 * fashion based on a type list of the supported configurations.
 *
 */
template <typename... Configs> struct InstantiateFromList;

/** General case of the recursion.
 *
 * Compare the runtime configuration with the configuration of the first element
 * in the list. If they match, instantiate the polynomial implementation. Otherwise,
 * recurse to the next element in the list.
 *
 */
template <std::size_t Degree, std::size_t Width, typename... Configs>
struct InstantiateFromList<finufft::detail::poly_kernel_config<Degree, Width>, Configs...> {
    template <typename T, std::size_t Dim>
    SpreadSubproblemFunctor<T, Dim>
    make(finufft::detail::precomputed_poly_kernel_data const &data) const {
        if (data.width == Width && data.degree == Degree) {
            // Check if runtime request matches with compile-time configuration.
            // Note: polynomial width padded to next multiple of 4 for auto-vectorization.
            const std::size_t PadWidth = (Width + 3) / 4 * 4;
            return SpreadSubproblemPolynomialReference<T, PadWidth, Degree>{
                data.width, PolynomialBatch<T, PadWidth, Degree>(data.coefficients, data.width)};
        } else {
            return InstantiateFromList<Configs...>{}.template make<T, Dim>(data);
        }
    }
};

/** Base case of the recursion.
 *
 * No more configurations to compare, so we throw an exception.
 */
template <> struct InstantiateFromList<> {
    template <typename T, std::size_t Dim>
    SpreadSubproblemFunctor<T, Dim>
    make(finufft::detail::precomputed_poly_kernel_data const &) const {
        throw std::runtime_error("No suitable kernel found");
    }
};

/** Helper template to convert from a single tuple to variadic pack.
 * We simply use this to instantiate from the tuple of supported configurations
 * provided to us in the generated headers.
 *
 */
template <typename T> struct InstantiateFromTupleList;

template <typename... Configs>
struct InstantiateFromTupleList<std::tuple<Configs...>> : InstantiateFromList<Configs...> {
    using InstantiateFromList<Configs...>::make;
};
} // namespace detail

/** Instantiate a functor for the spreading subproblem based on a polynomial approximation.
 *
 * This function searches through the pre-generated polynomial approximations for one
 * that matches, and instantiates a functor accordingly. If no matches are found, an
 * exception is thrown.
 *
 * @tparam T The data type of the input and output data.
 * @tparam Dim The dimension of the problem.
 *
 * @param kernel Specification of the kernel to use.
 *
 */
template <typename T, std::size_t Dim>
SpreadSubproblemFunctor<T, Dim>
get_subproblem_polynomial_reference_functor(kernel_specification const &kernel) {
    // Search through list of pre-generated kernels for matching configuration.
    // To avoid floating-point issues, we match on a quantized version of the beta parameter.
    auto it = std::find_if(
        finufft::detail::precomputed_poly_kernel_data_table.begin(),
        finufft::detail::precomputed_poly_kernel_data_table.end(),
        [&kernel](finufft::detail::precomputed_poly_kernel_data const &entry) {
            return (entry.width == kernel.width) &&
                   (entry.beta_1000 == static_cast<std::size_t>(kernel.es_beta * 1000));
        });

    // Could not find matching kernel, throw an exception.
    if (it == finufft::detail::precomputed_poly_kernel_data_table.end()) {
        throw std::runtime_error("No precomputed polynomial kernel data found for given kernel.");
    }

    // Instantiate functor based on the found kernel.
    auto factory = detail::InstantiateFromTupleList<finufft::detail::poly_kernel_configs_t>{};
    return factory.template make<T, Dim>(*it);
}

// Explicit instantiation of the functor in common dimension and data types.
extern template SpreadSubproblemFunctor<float, 1>
get_subproblem_polynomial_reference_functor<float, 1>(kernel_specification const &);
extern template SpreadSubproblemFunctor<float, 2>
get_subproblem_polynomial_reference_functor<float, 2>(kernel_specification const &);
extern template SpreadSubproblemFunctor<float, 3>
get_subproblem_polynomial_reference_functor<float, 3>(kernel_specification const &);
extern template SpreadSubproblemFunctor<double, 1>
get_subproblem_polynomial_reference_functor<double, 1>(kernel_specification const &);
extern template SpreadSubproblemFunctor<double, 2>
get_subproblem_polynomial_reference_functor<double, 2>(kernel_specification const &);
extern template SpreadSubproblemFunctor<double, 3>
get_subproblem_polynomial_reference_functor<double, 3>(kernel_specification const &);

} // namespace spreading
} // namespace finufft
