#pragma once

/** @file
 *
 * This file contains the main definition and hierarchy for the finufft plan structure.
 * Roughly speaking, a finufft plan is created for a particular problem (e.g. specifying
 * the number of input points, the tolerance, parameters of the kernel etc.), and it
 * contains all necessary information to execute the transform. In particular, it may
 * allocate required temporary memory, decide on the optimal kernels to execute etc.
 *
 */

#include <cstddef>
#include <tcb/span.hpp>

namespace finufft {

/** Main interface encapsulating a finufft plan for a type-1 transform.
 *
 *
 */
template <typename T, std::size_t Dim> class Type1PlanInterface {
  public:
    virtual ~FinufftTypedPlanInterface() = default;
    virtual std::size_t dim() const override final { return Dim; }
    virtual void execute(tcb::span<T const *, Dim> coordinates, T const *weights, T *result) = 0;
};

} // namespace finufft
