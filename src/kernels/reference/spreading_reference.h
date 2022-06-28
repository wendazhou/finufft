#pragma once

#include "../../spreading.h"
#include "../span.hpp"

namespace finufft {
namespace spreading {

/** Direct evaluation of the exp-sqrt kernel.
 *
 * Note that due to the use of transcendental functions, this evaluation
 * is fairly slow, and polynomial approximations are more performant.
 *
 */
template <typename T> inline T evaluate_es_kernel_direct(T x, T es_beta, T es_c) {
    auto s = std::sqrt(static_cast<T>(1.0) - es_c * x * x);
    return std::exp(es_beta * s);
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
    T *output, tcb::span<std::size_t, Dim> strides, tcb::span<T const *, Dim> values,
    std::size_t width, T k_re, T k_im) {
    return WriteSeparableKernelImpl<T, Dim>{}(output, strides, values, width, k_re, k_im);
}

template <typename T, std::size_t Dim> struct WriteSeparableKernelImpl {
    void operator()(
        T *output, tcb::span<std::size_t, Dim> strides, tcb::span<T const *, Dim> values,
        std::size_t width, T k_re, T k_im) {
        for (std::size_t i = 0; i < width; ++i) {
            auto k_re_i = k_re * values[Dim - 1][i];
            auto k_im_i = k_im * values[Dim - 1][i];

            WriteSeparableKernelImpl<T, Dim - 1>{}(
                output + 2 * i * strides[Dim - 1],
                strides.template subspan<0, Dim - 1>(),
                values.template subspan<0, Dim - 1>(),
                width,
                k_re_i,
                k_im_i);
        }
    }
};

template <typename T> struct WriteSeparableKernelImpl<T, 1> {
    void operator()(
        T *output, tcb::span<std::size_t, 1> strides, tcb::span<T const *, 1> values,
        std::size_t width, T k_re, T k_im) {

        // Base case of the recursion: 1-D kernel accumulation.
        for (std::size_t i = 0; i < width; ++i) {
            output[2 * i * strides[0]] += k_re * values[0][i];
            output[2 * i * strides[0] + 1] += k_im * values[0][i];
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
 * @param input Non-uniform points to spread.
 * @param grid Grid specification.
 * @param output[out] Output array. Must no alias any other input pointer.
 * @param kernel Evaluation function of the kernel. Must be callable with signature
 *     `void (T*, T)`, and additionally have a member `width` indicating the width of
 *      the output of the kernel. When called, the function is guaranteed that its
 *      first argument is a pointer to a contiguous array of length `width`. Note that
 *      this width may be larger than the true width of the kernel to enable padding
 *      and related optimizations.
 * @param kernel_width The width of the kernel being evaluated.
 *
 */
template <std::size_t Dim, typename T, typename Fn>
void spread_subproblem_generic_with_kernel(
    nu_point_collection<Dim, T const *> const &input, grid_specification<Dim> const &grid,
    T *__restrict output, Fn &&kernel, std::size_t kernel_width) {
    std::fill_n(output, 2 * grid.num_elements(), T(0));

    // Allocate according to the width requested by the kernel.
    auto kernel_values_stride = round_to_next_multiple(kernel.width, 8);
    auto kernel_values = allocate_aligned_array<T>(kernel_values_stride * Dim, 64);
    std::fill_n(kernel_values.get(), kernel_values_stride * Dim, T(0));

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
        kernel_values_view[dim] = kernel_values.get() + kernel_values_stride * dim;
    }

    for (std::size_t i = 0; i < input.num_points; ++i) {
        std::size_t point_total_offset = 0;

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            // Compute kernel values in each dimension
            auto x = input.coordinates[dim][i];
            auto x_i = static_cast<int64_t>(std::ceil(x - ns2));
            auto x_f = x_i - x;

            kernel(kernel_values.get() + dim * kernel_values_stride, x_f);

            point_total_offset += (x_i - grid.offsets[dim]) * strides[dim];
        }

        // Write out product kernel to output array.
        write_separable_kernel<T, Dim>(
            output + 2 * point_total_offset,
            strides,
            kernel_values_view,
            kernel_width,
            input.strengths[2 * i],
            input.strengths[2 * i + 1]);
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
    T es_c;
    std::size_t width;

    void operator()(T *output, T x) const {
        for (std::size_t i = 0; i < width; ++i) {
            output[i] = evaluate_es_kernel_direct(x + static_cast<T>(i), es_beta, es_c);
        }
    }
};

/** Reference implementation for subproblem with direct evaluation of exponential-sqrt kernel.
 *
 * This functor provides a direct naive subproblem implementation, by evaluating the exp-sqrt
 * kernel for each point and accumulating into the output.
 *
 */
struct SpreadSubproblemDirectReference {
    static const std::size_t num_points_multiple = 1;
    static const std::size_t extent_multiple = 1;

    template <std::size_t Dim, typename T>
    void operator()(
        nu_point_collection<Dim, T const *> const &input, grid_specification<Dim> const &grid,
        T *output, const kernel_specification &kernel) const {
        KernelDirectReference<T> kernel_fn{
            static_cast<T>(kernel.es_beta),
            static_cast<T>(kernel.es_c),
            static_cast<std::size_t>(kernel.width)};
        spread_subproblem_generic_with_kernel(
            input, grid, output, kernel_fn, static_cast<std::size_t>(kernel.width));
    }
};

static const SpreadSubproblemDirectReference spread_subproblem_direct_reference;

// @{

/** Implementation of Horner scheme for polynomial evaluation through a recursive strategy.
 * This structure implements a recursive for polynomial evaluation.
 * 
 * @param x The value to evaluate the polynomial at.
 * @param coeffs The coefficients of the polynomial in reverse order. Note: must be array of length `degree + 1`.
 * 
 * @tparam T The type of the evaluation.
 * @tparam Arr The type of the array of coefficients. Must produce a result compatible
 *    with the `T` type when indexed.
 * 
 */
template <typename T, std::size_t Degree> struct HornerPolynomialEvaluation {
    template <typename Arr> T operator()(T x, Arr const &coeffs) const {
        // Note: will dispatch to efficient FMA implementation if available on most compilers.
        // Explictly control FMA contraction using #pragma STDC FP_CONTRACT if desired.
        return x * HornerPolynomialEvaluation<T, Degree - 1>{}(x, coeffs) + coeffs[Degree];
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

/** Structure for evaluating a set of polynomials of the given degree.
 *
 * In order to accelerate evaluation of the kernel, we may make use of
 * a polynomial approximation. This structure provides an implementation
 * for polynomial evaluation through Horner's method in a generic setting.
 *
 */
template <typename T, std::size_t Width, std::size_t Degree> struct KernelPolynomialReference {
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

    KernelPolynomialReference() : coefficients(allocate_aligned_array<T>(Width * (Degree + 1), 64)) {}

    /** Create a polynomial from the given coefficients.
     * 
     * This constructor gathers the weights from the given array of coefficients,
     * expressed in standard order with the degree being the slower dimension.
     * Additionally, this may be used to expand the number of polynomials evaluated
     * by padding with zero coefficients beyond the specified width, in order
     * to facilitate vectorization of the kernel.
     * 
     * @param coefficients Array of coefficients of the polynomial. Must be of length `width * (degree + 1)`,
     *     and contain the coefficients with the width being the faster dimension and the degree being the slower dimension.
     * @param width The width of the polynomial provided in the coefficients array. Must be less or equal to Width.
     * 
     */
    template<typename U>
    KernelPolynomialReference(U const* coefficients, std::size_t width = Width) : KernelPolynomialReference() {
        for(std::size_t i = 0; i < Degree + 1; ++i) {
            auto deg_coeffs = this->coefficients.get() + (Degree - i) * Width;
            std::copy(coefficients + i * width, coefficients + (i + 1) * width, deg_coeffs);
            std::fill(deg_coeffs + width, deg_coeffs + Width, static_cast<T>(0));
        }
    }

    // Need copy-constructor for compatibility with std::function
    KernelPolynomialReference(KernelPolynomialReference const &other)
        : KernelPolynomialReference() {
        std::copy(
            other.coefficients.get(),
            other.coefficients.get() + Width * (Degree + 1),
            coefficients.get());
    };

    KernelPolynomialReference(KernelPolynomialReference &&) noexcept = default;

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
 */
template <typename T, std::size_t Width, std::size_t Degree>
struct SpreadSubproblemPolynomialReference {
    static const std::size_t num_points_multiple = 1;
    static const std::size_t extent_multiple = 1;

    KernelPolynomialReference<T, Width, Degree> kernel_polynomial;

    template <std::size_t Dim>
    void operator()(
        nu_point_collection<Dim, T const *> const &input, grid_specification<Dim> const &grid,
        T *output, const kernel_specification &kernel) const {
        return spread_subproblem_generic_with_kernel(
            input, grid, output, kernel_polynomial, static_cast<std::size_t>(kernel.width));
    }
};

} // namespace spreading
} // namespace finufft
