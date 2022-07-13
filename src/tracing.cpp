#include "tracing.h"

namespace finufft {

Timer Timer::make_timer(std::string const &name) {
    if (!root_) {
        return Timer();
    }

    return root_->make_subtimer(name, index_);
}

std::vector<std::tuple<std::string, Timer::duration>> Timer::report() {
    if (!root_) {
        return std::vector<std::tuple<std::string, Timer::duration>>();
    }

    return root_->report(this->name());
}

std::string Timer::name() const {
    std::lock_guard<std::mutex> lock(root_->mutex_);
    return root_->names_[index_];
}

TimerRoot::TimerRoot(std::string name) : root_name_(name) {
#ifdef FINUFFT_ENABLE_ITT
    itt_domain_ = __itt_domain_create(name.c_str());
#endif
}

void TimerRoot::record(std::size_t idx, Timer::duration duration) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    durations_[idx] += duration;
}

Timer TimerRoot::make_timer_unsafe(std::string name) {
    // Create ITT name before moving string.
#ifdef FINUFFT_ENABLE_ITT
    auto itt_name = __itt_string_handle_create(name.c_str());
    itt_names_.push_back(itt_name);
#endif

    names_.push_back(std::move(name));
    durations_.push_back(Timer::duration{});

    return Timer(
        this,
        names_.size() - 1
#ifdef FINUFFT_ENABLE_ITT
        ,
        itt_name
#endif
    );
}

Timer TimerRoot::make_timer(std::string name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return make_timer_unsafe(std::move(name));
}

Timer TimerRoot::make_subtimer(std::string const &name, std::size_t parent_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto parent_name = names_[parent_idx];
    auto new_name = parent_name + "/" + name;
    return make_timer_unsafe(std::move(new_name));
}

std::vector<std::tuple<std::string, Timer::duration>>
TimerRoot::report(std::string const &prefix = "") {
    std::vector<std::tuple<std::string, Timer::duration>> result;

    // Compute size of prefix to filter out
    // If prefix does not end in /, filter out an additional element.
    auto prefix_size = prefix.size();
    if ((prefix_size > 0) && (prefix[prefix.size() - 1] != '/')) {
        prefix_size += 1;
    }

    for (std::size_t i = 0; i < names_.size(); ++i) {
        if (names_[i].rfind(prefix, 0) != 0) {
            continue;
        }

        result.emplace_back(
            names_[i].substr(std::min(prefix_size, names_[i].size()), std::string::npos),
            durations_[i]);
    }

    return result;
}

} // namespace finufft
