#pragma once

#include <cstddef>
#include "kernels/span.hpp"

namespace finufft {
namespace detail {

/** Structure recording information about pre-computed polynomial weights.
 * 
 * This structure records the information used to pre-compute a polynomial approximation
 * of the function exp(beta * sqrt(1 - z^2)), where z is in the range [-1, 1].
 * The approximation is computed as a piecewise polynomial, with the number of pieces
 * given by width.
 * 
 * The coefficients correspond to a double precision array of (degree + 1) * width elements,
 * storing the coefficients for each polynomial piece. The coefficients are stored with the
 * faster dimension being the piece, and the slower dimension being the degree.
 * 
 * @var degree The degree of the polynomial.
 * @var beta_1000 The truncated value of beta * 1000 corresponding to the fitted kernel.
 * @var width The width (number of points) of the kernel.
 * @var coefficients The coefficients of the polynomial.
 * 
 */
struct precomputed_poly_kernel_data {
    std::size_t degree;
    std::size_t beta_1000;
    std::size_t width;
    double const *coefficients;
};

/** List of all pre-computed polynomial kernels.
 * 
 */
extern const tcb::span<const precomputed_poly_kernel_data> precomputed_poly_kernel_data_table;

} // namespace detail
} // namespace finufft
