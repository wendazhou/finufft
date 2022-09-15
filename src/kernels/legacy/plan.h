#pragma once

/** @file
 *
 * Main entry points for creating plans using the current implementation.
 *
 */

#include "../../plan.h"

namespace finufft {
namespace legacy {

/** Create type-1 plan from given configuration using current implementation.
 *
 */
template <typename T, std::size_t Dim>
Type1Plan<T, Dim> make_type1_plan(Type1TransformConfiguration<Dim> const &configuration);

} // namespace legacy
} // namespace finufft
