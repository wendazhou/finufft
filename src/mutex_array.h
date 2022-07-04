#pragma once

#include <algorithm>
#include <atomic>
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


#ifdef __cpp_lib_atomic_wait

/** Thin mutex implementation based on C++20 atomic_wait.
 * 
 */
class ThinMutex {
    std::atomic<int> value_;

  public:
    ThinMutex() : value_(0) {}
    ThinMutex(ThinMutex const &) = delete;
    ThinMutex(ThinMutex &&) = delete;

    void lock() noexcept {
        int c = 0;
        // Check if mutex is held
        if (!value_.compare_exchange_strong(c, 1)) {
            // Mutex is held, declare that there are multiple waiters
            // By storing 2.
            if (c != 2) {
                c = value_.exchange(2);
            }

            // Wait on value
            while (c != 0) {
                std::atomic_wait(&value_, 2);
                c = value_.exchange(2);
            }
        }
    }

    void unlock() noexcept {
        // Release mutex, and check for waiters
        if (--value_ != 1) {
            // If there are waiters, wake one up
            value_ = 0;
            std::atomic_notify_one(&value_);
        }
    }
};

typedef ThinMutex Mutex;

#else

typedef std::mutex Mutex;

#endif

/** Array of mutexes.
 *
 * This class encapsulates a system to provide exclusion at a given
 * number of locations. The current implementation is based on a
 * a set of mutexes, but a more efficient implementation could
 * be based on `std::atomic_wait` using C++20 features.
 *
 */
class MutexArray {
  public:
    // The underlying type of the mutex currently used by this array.
    typedef Mutex mutex_type;
  private:
    std::size_t size_;
    std::unique_ptr<mutex_type[]> mutexes_;

  public:

    static std::size_t compute_size(std::size_t num_elements) {
        auto max_mutexes = detail::get_next_power_of_two(8 * std::thread::hardware_concurrency());

        if (num_elements > max_mutexes) {
            return max_mutexes;
        } else {
            return detail::get_next_power_of_two(num_elements);
        }
    }

    MutexArray(std::size_t size)
        : size_(size), mutexes_(std::make_unique<mutex_type[]>(size_)) {}
    mutex_type &operator[](std::size_t i) const {
        return mutexes_[i];
    }
};

} // namespace finufft
