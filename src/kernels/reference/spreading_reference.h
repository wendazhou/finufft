#pragma once

#include "../../spreading.h"
#include "../span.hpp"

namespace finufft {
namespace spreading {

template <typename T> inline T evaluate_es_kernel_direct(T x, T es_beta, T es_c, int kernel_width) {
    auto s = std::sqrt(static_cast<T>(1.0) - es_c * x * x);
    return std::exp(es_beta * s);
}

///@{

template <typename T, std::size_t Dim> struct WriteSeparableKernelImpl {
    void operator()(
        T *output, tcb::span<std::size_t, Dim> strides, tcb::span<T const *, Dim> values,
        std::size_t width, T k_re, T k_im) {
        for (std::size_t i = 0; i < width; ++i) {
            auto k_re_i = k_re * values[Dim - 1][i];
            auto k_im_i = k_im * values[Dim - 1][i];

            WriteSeparableKernelImpl<T, Dim - 1>{}(
                output + 2 * i * strides[Dim - 1],
                strides.template subspan<0, Dim-1>(),
                values.template subspan<0, Dim-1>(),
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
        for (std::size_t i = 0; i < width; ++i) {
            output[2 * i * strides[0]] += k_re * values[0][i];
            output[2 * i * strides[0] + 1] += k_im * values[0][i];
        }
    }
};

/** Compute values of a separable kernel in arbitrary dimension and accumulates to the output array.
 *
 * This function computes the values of a separable kernel given as the tensor product of
 * 1-dimension kernels, and accumulates the values into the given output array. The output array is
 * assumed to be in column-major strided format, as an interleaved complex array. The kernel is
 * assumed to be real-valued, and multiplied by a complex scalar given through `k_re` and `k_im`.
 * 
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

///@}

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

        std::fill_n(output, 2 * grid.num_elements(), T(0));

        auto ns2 = static_cast<T>(0.5 * kernel.width);
        T es_beta = static_cast<T>(kernel.es_beta);
        T es_c = static_cast<T>(kernel.es_c);

        auto kernel_values_stride = round_to_next_multiple(kernel.width, 8);
        auto kernel_values = allocate_aligned_array<T>(kernel_values_stride * Dim, 64);

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

                for (int i = 0; i < kernel.width; ++i) {
                    kernel_values[dim * kernel_values_stride + i] = evaluate_es_kernel_direct(
                        x_f + static_cast<T>(i), es_beta, es_c, kernel.width);
                }

                point_total_offset += (x_i - grid.offsets[dim]) * strides[dim];
            }

            // Write out product kernel to output array.
            write_separable_kernel<T, Dim>(
                output + 2 * point_total_offset,
                strides,
                kernel_values_view,
                kernel.width,
                input.strengths[2 * i],
                input.strengths[2 * i + 1]);
        }
    }
};

static const SpreadSubproblemDirectReference spread_subproblem_direct_reference;

} // namespace spreading
} // namespace finufft
