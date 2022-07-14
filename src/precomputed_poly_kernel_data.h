#pragma once

#include <cstddef>
#include <tuple>

#include <tcb/span.hpp>
#include "kernels/spreading.h"

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

/** Utility template to record configurations of polynomial kernels.
 *
 * @tparam Degree The degree of the polynomial
 * @tparam Width The width of the kernel
 *
 */
template <std::size_t Degree, std::size_t Width> struct poly_kernel_config {
    static const std::size_t degree = Degree;
    static const std::size_t width = Width;
};

// This generated file defines a type poly_kernel_configs_t
// This type is a tuple of poly_kernel_config<Degree, Width>,
// with one entry for each degree and width used in the precomputed kernels.
#include "precomputed_poly_kernel_configs.inl"

/** Searches through the list of pre-computed polynomial kernels for a matching configuration.
 * If it is found, returns a pointer to the corresponding precomputed kernel data.
 * Otherwise, returns a null pointer.
 * 
 * @param kernel The kernel specification to search for.
 * 
 */
precomputed_poly_kernel_data const *
search_precomputed_poly_kernel(finufft::spreading::kernel_specification const &kernel) noexcept;

} // namespace detail
} // namespace finufft
