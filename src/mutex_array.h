#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include "bit.h"

namespace finufft {

#ifdef __cpp_lib_atomic_wait

/** Thin mutex implementation based on C++20 atomic_wait.
 *
 */
class ThinMutex {
    std::atomic<int> value_;

    static int cmpxchg(std::atomic<int> &v, int expected, int desired) {
        if (v.compare_exchange_strong(expected, desired)) {
            // We succeeded, return the value we swapped to
            return desired;
        }

        // We failed, return the current value, which has been loaded
        // into expected by the compare_exchange_strong call.
        return expected;
    }

  public:
    ThinMutex() : value_(0) {}
    ThinMutex(ThinMutex const &) = delete;
    ThinMutex(ThinMutex &&) = delete;

    void lock() noexcept {
        int c = 0;
        // Check if mutex is held
        if ((c = cmpxchg(value_, 0, 1)) != 0) {
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

typedef std::mutex Mutex;

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
        auto max_mutexes = bit_ceil(8 * std::thread::hardware_concurrency());

        if (num_elements > max_mutexes) {
            return max_mutexes;
        } else {
            return bit_ceil(num_elements);
        }
    }

    MutexArray(std::size_t size) : size_(size), mutexes_(std::make_unique<mutex_type[]>(size_)) {}
    mutex_type &operator[](std::size_t i) const { return mutexes_[i]; }
};

} // namespace finufft
