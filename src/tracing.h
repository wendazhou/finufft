#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace finufft {

class Timer;
class TimerRoot;
class ScopedTimerGuard;

/** Timer class.
 *
 * This class represents a timer which can be used to record a single event.
 * Note that timer creation may be expensive, so it is recommended to create
 * them ahead of time outside of the critical path.
 *
 */
class Timer {
  public:
    friend class TimerRoot;
    typedef std::chrono::high_resolution_clock::time_point time_point;
    typedef std::chrono::high_resolution_clock::duration duration;

  private:
    TimerRoot *root_;
    time_point start_;
    std::size_t index_;

    Timer(TimerRoot *root, std::size_t index) : root_(root), start_(), index_(index) {}

  public:
    Timer() : root_(nullptr), start_(), index_(0) {}
    Timer(Timer const &) = delete;
    Timer(Timer &&) = default;

    void start() noexcept;
    duration end() noexcept;

    /** Make a new timer with the given name.
     *
     * The newly created timer is derived from the current timer,
     * and its name will be prefixed with the current timer's name.
     *
     */
    Timer make_timer(std::string const &name);

    std::vector<std::tuple<std::string, Timer::duration>> report();
    std::string name() const;
};

/** RAII wrapper to start / stop timer on scope exit.
 *
 * It may be initialized from a nullptr, in which case no information
 * is recorded.
 *
 */
class ScopedTimerGuard {
    Timer *timer_;

  public:
    explicit ScopedTimerGuard(Timer *timer) : timer_(timer) {
        if (timer_) {
            timer_->start();
        }
    }
    explicit ScopedTimerGuard(Timer &timer) : ScopedTimerGuard(&timer) {}
    ~ScopedTimerGuard() noexcept {
        if (timer_) {
            timer_->end();
        }
    }

    ScopedTimerGuard(ScopedTimerGuard const &) = delete;
};

/** Timer root class.
 *
 * This class holds a collection of timers, and can be used
 * to create new timers. It aggregates elapsed time across all
 * the timers.
 *
 */
class TimerRoot {
    friend class Timer;

    std::mutex mutex_;
    std::string root_name_;
    std::vector<std::string> names_;
    std::vector<Timer::duration> durations_;

  private:
    void record(std::size_t idx, Timer::duration duration) noexcept;

    /// Create a timer without locking.
    /// Unsafe to call unless you have already aquired a lock on mutex_.
    Timer make_timer_unsafe(std::string name);
    /// Create a subtimer from the given name and parent index.
    Timer make_subtimer(std::string const &name, std::size_t parent_idx);

  public:
    TimerRoot(std::string name);

    /** Create a new timer with the given name.
     *
     * Note that the lifetime of the returned timer is tied to the lifetime
     * of the TimerRoot object creating it.
     *
     * @param name The name of the timer.
     */
    Timer make_timer(std::string name);

    /** Obtain a list of the accumulated time for each timer.
     *
     * Note that timers are aggregated by name (not instance),
     * and the returned list represents the total time across
     * all timers which share the name.
     *
     * @param prefix If non-empty, only timers with the given prefix will be included.
     *     Note that for conciseness, the prefix is not included in the returned names.
     *
     */
    std::vector<std::tuple<std::string, Timer::duration>> report(std::string const &prefix);
};

} // namespace finufft
