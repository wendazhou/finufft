#pragma once

#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>

#ifdef __cpp_lib_int_pow2
#include <bit>
#endif

namespace finufft {
namespace detail {

inline std::size_t get_next_power_of_two(std::size_t n) noexcept {
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

template <typename T> bool has_single_bit(T x) noexcept {
#ifdef __cpp_lib_int_pow2
    return std::has_single_bit(x);
#else
    return x != 0 && (x & (x - 1)) == 0;
#endif
}

} // namespace detail

/** Array of mutexes.
 *
 * This class encapsulates a system to provide exclusion at a given
 * number of locations. The current implementation is based on a
 * a set of mutexes, but a more efficient implementation could
 * be based on `std::atomic_wait` using C++20 features.
 *
 */
class MutexArray {
    std::size_t size_;
    std::unique_ptr<std::mutex[]> mutexes_;

  public:
    // The underlying type of the mutex currently used by this array.
    typedef std::mutex mutex_type;

    static std::size_t compute_size(std::size_t num_elements) {
        auto max_mutexes = detail::get_next_power_of_two(8 * std::thread::hardware_concurrency());

        if (num_elements > max_mutexes) {
            return max_mutexes;
        } else {
            return detail::get_next_power_of_two(num_elements);
        }
    }

    MutexArray(std::size_t size)
        : size_(compute_size(size)), mutexes_(std::make_unique<std::mutex[]>(size_)) {}
    std::mutex &operator[](std::size_t i) const {
        // Multiply by a prime to avoid needless collisions due to aliasing
        // across small powers of 2
        return mutexes_[(i * 17) & (size_ - 1)];
    }
};

} // namespace finufft
