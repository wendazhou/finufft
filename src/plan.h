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

#include "memory.h"

#include "kernels/fft.h"
#include "kernels/sorting.h"
#include "kernels/spreading.h"

namespace finufft {

/** Main interface encapsulating a finufft plan for a type-1 transform.
 *
 *
 */
template <typename T, std::size_t Dim> class Type1PlanInterface {
  public:
    virtual ~Type1PlanInterface() = default;
    virtual std::size_t dim() const override final { return Dim; }
    virtual void execute(tcb::span<T const *, Dim> coordinates, T const *weights, T *result) = 0;
};

template <typename T, std::size_t Dim>
class KernelType1PlanInterface : public Type1PlanInterface<T, Dim> {
    finufft::aligned_unique_array<T> uniform_buffer_;
    finufft::spreading::SpreadBlockedFunctor<T, Dim> spread_blocked_;
    finufft::fft::PlannedFourierTransformation<T> fft_;

  public:
    virtual ~KernelType1PlanInterface() = default;
    virtual void
    execute(tcb::span<T const *, Dim> coordinates, T const *weights, T *result) override final {}
};

} // namespace finufft
