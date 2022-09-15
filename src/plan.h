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

#include <algorithm>
#include <cstddef>
#include <tcb/span.hpp>

#include "memory.h"

#include "kernels/fft.h"
#include "kernels/interpolation.h"
#include "kernels/sorting.h"
#include "kernels/spreading.h"

namespace finufft {

/** Main interface encapsulating a finufft plan for a type-1 transform.
 *
 *
 */
template <typename T, std::size_t Dim> class Type1Plan {
  public:
    struct Concept {
        virtual ~Concept() = default;
        virtual void operator()(
            std::size_t num_points, tcb::span<T const *const, Dim> coordinates, T const *weights,
            T *result) = 0;
    };

  private:
    template <typename Impl> struct Model : Concept {
        Impl impl_;

        Model(Impl &&impl) : impl_(std::move(impl)) {}

        void operator()(
            std::size_t num_points, tcb::span<T const *const, Dim> coordinates, T const *weights,
            T *result) override final {
            impl_(num_points, coordinates, weights, result);
        }
        std::unique_ptr<Concept> clone() const { return std::make_unique<Model<Impl>>(*this); }
    };

    std::unique_ptr<Concept> holder_;

  public:
    template <typename Impl>
    Type1Plan(Impl &&impl) : holder_(std::make_unique<Model<Impl>>(std::move(impl))) {}
    Type1Plan(Type1Plan const &other) : holder_(other.holder_->clone()) {}
    Type1Plan(Type1Plan &&) noexcept = default;
    Type1Plan &operator=(Type1Plan const &other) {
        holder_ = other.holder_->clone();
        return *this;
    }
    Type1Plan &operator=(Type1Plan &&) noexcept = default;

    constexpr std::size_t dim() const noexcept { return Dim; }
    void operator()(
        std::size_t num_points, tcb::span<const T *const, Dim> coordinates, T const *weights,
        T *result) {
        (*holder_)(num_points, coordinates, weights, result);
    }
};

template <typename T, std::size_t Dim>
Type1Plan<T, Dim> make_type1_plan(
    finufft::aligned_unique_array<T> uniform_buffer,
    finufft::spreading::SpreadFunctor<T, Dim> spread_blocked,
    finufft::fft::PlannedFourierTransformation<T> fft,
    finufft::interpolation::InterpolationFunctor<T, Dim> interpolate);

/** Configuration for creating a type-1 transform.
 */
template <std::size_t Dim> struct Type1TransformConfiguration {
    std::size_t max_num_points_;         //!< Maximum number of input points
    std::array<std::size_t, Dim> modes_; //!< Number of modes in each dimension
    double tolerance_ = 1e-6;            //!< Target tolerance for the transform
    double upsampling_factor_ = 2.0;     //!< Upsampling factor for the transform
    std::size_t max_threads_ = 0;        //!< Maximum number of threads to use
    int mode_ordering_ = 1;              //!< Ordering of the output modes
};

} // namespace finufft
