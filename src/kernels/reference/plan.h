#pragma once

/** @file
 *
 * Main entry points for creating plans using the reference implementation
 *
 */

#include "../../plan.h"

namespace finufft {
namespace reference {

/** Create type-1 plan which implements an exact non-uniform Fourier transform
 * through O(nm) direct summation. Note that this is only intendend for testing
 * and not for production use. In particular, it is both a sub-optimal algorithm,
 * and it is not optimized.
 *
 */
template <typename T, std::size_t Dim>
Type1Plan<T, Dim> make_exact_type1_plan(Type1TransformConfiguration<Dim> const &configuration);

} // namespace legacy
} // namespace finufft
