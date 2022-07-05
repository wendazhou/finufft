#pragma once

/** Implementations of utilities from <bit> for earlier C++ versions.
 *
 */

#if defined(__cpp_lib_int_pow2)
#include <bit>
#endif

#include <limits>
#include <type_traits>

namespace finufft {

inline std::size_t bit_ceil(std::size_t n) noexcept {
#ifdef __cpp_lib_int_pow2
    return std::bit_ceil(n);
#else
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    if (sizeof(std::size_t) == 8) {
        n |= n >> 32;
    }

    return n + 1;
#endif
}

template <typename T>
std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, bool>
has_single_bit(T x) noexcept {
#ifdef __cpp_lib_int_pow2
    return std::has_single_bit(x);
#else
    return x != 0 && (x & (x - 1)) == 0;
#endif
}

template <typename T>
std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, T>
bit_width(T x) noexcept {
#ifdef __cpp_lib_int_pow2
    return std::bit_width(x);
#else
    for(int i = 1; i < std::numeric_limits<T>::digits; i++) {
        if(x < (static_cast<T>(1) << i)) {
            return i;
        }
    }
    return std::numeric_limits<T>::digits;
#endif
}


} // namespace finufft
