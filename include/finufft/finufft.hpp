#pragma once

#include "../finufft.h"
#include <complex>
#include <cstddef>

/** @file
 *
 * Object-oriented C++ interfaces for the FINUFFT library.
 *
 */

namespace finufft {

/** Type-1 nu-fft transform.
 * 
 * @fn nuft_plan_type1(std::array<int64_t, Dim> const &sizes, double tol, finufft_opts const* opts)
 * @brief Creates a new plan for planning a type-1 nuft transform.
 * @param sizes The number of target modes in each dimension
 * @param tol The tolerance for the nuft transform
 * @param[opt] opts Options for the nuft transform. If nullptr, the default options are used.
 * 
 * @fn set_points(int64_t n, std::array<T*, Dim> const* points)
 * @brief Sets the location of non-uniform input points.
 * @param n The number of points
 * @param points The location of the points, given as an array for each coordinate.
 * 
 * @fn execute(std::complex<T> const* strengths, std::complex<T>* output)
 * @brief executes the transform and writes the output
 * @param strengths The strength of each point
 * @param output The output of the transform
 *
 */
template <typename T, std::size_t Dim> struct nuft_plan_type1;

template <std::size_t Dim> struct nuft_plan_type1<float, Dim> {
    typedef finufftf_plan plan_t;
    plan_t plan_;

    nuft_plan_type1(std::array<int64_t, Dim> const &sizes, double tol, finufft_opts const *opts)
        : plan_() {
        finufftf_makeplan(
            1,
            sizes.size(),
            const_cast<int64_t *>(sizes.data()),
            0,
            1,
            tol,
            &plan_,
            const_cast<finufft_opts *>(opts));
    }

    ~nuft_plan_type1() noexcept { finufftf_destroy(plan_); }

    void set_points(int64_t num_points, std::array<float *, Dim> const &points) {
        finufftf_setpts(
            plan_,
            num_points,
            points[0],
            Dim > 1 ? points[1] : nullptr,
            Dim > 2 ? points[2] : nullptr,
            0,
            nullptr,
            nullptr,
            nullptr);
    }

    void execute(std::complex<float> const *strengths, std::complex<float> *output) {
        finufftf_execute(plan_, const_cast<std::complex<float> *>(strengths), output);
    }
};

template <std::size_t Dim> struct nuft_plan_type1<double, Dim> {
    typedef finufft_plan plan_t;
    plan_t plan_;

    nuft_plan_type1(std::array<int64_t, Dim> const &sizes, double tol, finufft_opts const *opts)
        : plan_() {
        finufft_makeplan(
            1,
            sizes.size(),
            const_cast<int64_t *>(sizes.data()),
            0,
            1,
            tol,
            &plan_,
            const_cast<finufft_opts *>(opts));
    }

    ~nuft_plan_type1() noexcept { finufft_destroy(plan_); }

    void set_points(int64_t num_points, std::array<double *, Dim> const &points) {
        finufft_setpts(
            plan_,
            num_points,
            points[0],
            Dim > 1 ? points[1] : nullptr,
            Dim > 2 ? points[2] : nullptr,
            0,
            nullptr,
            nullptr,
            nullptr);
    }

    void execute(std::complex<double> const *strengths, std::complex<double> *output) {
        finufft_execute(plan_, const_cast<std::complex<double> *>(strengths), output);
    }
};

} // namespace finufft
