#pragma once

/** @file
 * Utility: type-generic mathematical constants used in finufft.
 * This header mostyl polyfills the C++20 `std::numbers` header.
 *
 */

#if __has_include(<numbers>)
#include <numbers>
#else
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#define FINUFFT_UNDEF_USE_MATH_DEFINES
#endif
#include <cmath>
#ifdef FINUFFT_UNDEF_USE_MATH_DEFINES
#undef _USE_MATH_DEFINES
#undef FINUFFT_UNDEF_USE_MATH_DEFINES
#endif
#endif

namespace finufft {
namespace constants {
#ifdef __cpp_lib_math_constants
using std::numbers::pi_v;
#else
template <typename T> constexpr T pi_v = static_cast<T>(M_PI);
#endif
} // namespace constants
} // namespace finufft
