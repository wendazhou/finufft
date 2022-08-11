/** @file Benchmarks for implementations of counting sort.
 *
 * This file contains benchmarks to explore various implementations of a counting
 * sort.
 *
 */

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <memory>
#include <numeric>

#include <benchmark/benchmark.h>

#include <ips4o/ips4o.hpp>

#include "../src/memory.h"
#include "../src/tracing.h"

#include "../test/spread_test_utils.h"

namespace {

struct CountingSortTimers {
    finufft::Timer count;
    finufft::Timer sort;

    CountingSortTimers(finufft::Timer const &timer)
        : count(timer.make_timer("count")), sort(timer.make_timer("sort")) {}
};

struct FreeDeleter {
    void operator()(void *ptr) { return std::free(ptr); }
};

std::size_t round_to_next_multiple(std::size_t n, std::size_t m) { return (n + m - 1) / m * m; }

template <typename T, typename U>
void compute_histogram(
    T const *data, std::size_t num_elements, U *__restrict histogram, std::size_t num_buckets) {
    auto mask = std::bit_ceil(num_buckets) - 1;

    for (std::size_t i = 0; i < num_elements; ++i) {
        auto bucket = data[i] & mask;
        ++histogram[bucket];
    }
}

template <std::size_t Unroll, typename T, typename U>
void compute_histogram_unroll(
    T const *data, std::size_t num_elements, U *histogram, std::size_t num_buckets) {
    auto mask = num_buckets - 1;

    auto histogram_local_holder = finufft::allocate_aligned_array<U>(num_buckets * Unroll, 64);
    auto histogram_local = histogram_local_holder.get();
    std::memset(histogram_local, 0, num_buckets * Unroll * sizeof(U));

    std::size_t i = 0;
    for (; i + Unroll - 1 < num_elements; i += Unroll) {
        for (std::size_t j = 0; j < Unroll; ++j) {
            auto bucket = data[i + j] & mask;
            ++histogram_local[j * num_buckets + bucket];
        }
    }

    for (; i < num_elements; ++i) {
        auto bucket = data[i] & mask;
        ++histogram_local[bucket];
    }

    for (std::size_t i = 0; i < num_buckets; ++i) {
        for (std::size_t j = 0; j < Unroll; ++j) {
            histogram[i] += histogram_local[j * num_buckets + i];
        }
    }
}

template <typename T>
void counting_sort_vanilla(
    T const *data, T *__restrict output, std::size_t num_elements, std::size_t num_bins,
    CountingSortTimers &timers) {

    auto counts_holder = finufft::allocate_aligned_array<std::size_t>(num_bins + 1, 64);

    auto counts = counts_holder.get();

    {
        finufft::ScopedTimerGuard guard(timers.count);

        std::memset(counts, 0, num_bins * sizeof(std::size_t));
        // Compute bin counts
        compute_histogram_unroll<4>(data, num_elements, counts, num_bins);

        // Compute offsets
        std::partial_sum(counts, counts + num_bins, counts);
    }

    {
        finufft::ScopedTimerGuard guard(timers.sort);

        // Write data back
        auto mask = num_bins - 1;
        for (std::size_t i = num_elements - 1; i != -1; --i) {
            auto d = data[i];
            auto bin_index = d & mask;
            auto offset = --counts[bin_index];
            output[offset] = d;
        }
    }
}

struct BlockedCountSortData {
    finufft::aligned_unique_array<std::size_t> holder_;

    std::size_t *const __restrict counts;
    std::size_t *const __restrict local_buffer_offset;
    std::size_t *const __restrict local_buffer;

    BlockedCountSortData(std::size_t num_bins, std::size_t block_size)
        : holder_(finufft::allocate_aligned_array<std::size_t>(
              2 * round_to_next_multiple(num_bins, 8) +
                  round_to_next_multiple(num_bins * block_size, 8),
              64)),
          counts(holder_.get()),
          local_buffer_offset(holder_.get() + round_to_next_multiple(num_bins, 8)),
          local_buffer(holder_.get() + 2 * round_to_next_multiple(num_bins, 8)) {}
};

template <typename T, std::size_t BlockSize>
void counting_sort_blocked(
    T const *data, T *__restrict output, std::size_t num_elements, std::size_t num_bins,
    CountingSortTimers &timers) {
    BlockedCountSortData data_holder(num_bins, BlockSize);

    auto counts = data_holder.counts;

    auto bin_ptr = data_holder.local_buffer_offset;

    auto local_bins = data_holder.local_buffer;

    auto mask = std::bit_ceil(num_bins) - 1;

    {
        finufft::ScopedTimerGuard guard(timers.count);

        std::memset(counts, 0, num_bins * sizeof(std::size_t));
        // Compute bin counts
        compute_histogram_unroll<4>(data, num_elements, counts, num_bins);

        // Compute offsets
        std::partial_sum(counts, counts + num_bins, counts);
    }

    {
        finufft::ScopedTimerGuard guard(timers.sort);

        std::fill(bin_ptr, bin_ptr + num_bins, BlockSize);
        // Write data back
        for (std::size_t i = num_elements - 1; i != -1; --i) {
            auto d = data[i];
            auto bin_index = d & mask;
            auto offset = --counts[bin_index];

            auto local_bin_offset = --bin_ptr[bin_index];
            auto local_offset = local_bin_offset + bin_index * BlockSize;
            local_bins[local_offset] = d;

            if (local_bin_offset == 0) {
                std::memcpy(output + offset, local_bins + local_offset, BlockSize * sizeof(T));
                bin_ptr[bin_index] = BlockSize;
            }
        }

        // Fixup remaining bins
        for (std::size_t i = 0; i < num_bins; ++i) {
            if (bin_ptr[i] != BlockSize) {
                std::memcpy(
                    output + counts[i],
                    local_bins + i * BlockSize + bin_ptr[i],
                    (BlockSize - bin_ptr[i]) * sizeof(T));
            }
        }
    }
}

template <std::size_t BlockSize> struct BlockedAlignedCountingSort {
    std::size_t max_output_size(std::size_t num_elements, std::size_t num_bins) const {
        return num_elements + num_bins * (BlockSize - 1);
    }

    template <typename T>
    void operator()(
        std::type_identity_t<T> const *data, T *__restrict output, std::size_t num_elements,
        std::size_t num_bins, CountingSortTimers &timers) const {

        BlockedCountSortData data_holder(num_bins, BlockSize);

        auto counts = data_holder.counts;

        auto bin_ptr = data_holder.local_buffer_offset;

        auto local_bins = data_holder.local_buffer;

        auto mask = std::bit_ceil(num_bins) - 1;

        {
            finufft::ScopedTimerGuard guard(timers.count);

            std::memset(counts, 0, num_bins * sizeof(std::size_t));
            // Compute bin counts
            compute_histogram_unroll<4>(data, num_elements, counts, num_bins);
            for (std::size_t i = 0; i < num_bins; ++i) {
                counts[i] = round_to_next_multiple(counts[i], BlockSize);
            }

            // Compute offsets
            std::partial_sum(counts, counts + num_bins, counts);
        }

        {
            finufft::ScopedTimerGuard guard(timers.sort);

            std::memset(bin_ptr, 0, num_bins * sizeof(std::size_t));

            // Write data back
            for (std::size_t i = 0; i < num_elements; ++i) {
                auto d = data[i];
                auto bin_index = d & mask;

                auto local_bin_offset = bin_ptr[bin_index]++;
                auto local_offset = local_bin_offset + bin_index * BlockSize;
                local_bins[local_offset] = d;

                if (local_bin_offset == BlockSize - 1) {
                    auto offset = counts[bin_index] - BlockSize;
                    counts[bin_index] = offset;
                    std::memcpy(output + offset, local_bins + local_offset, BlockSize * sizeof(T));
                    bin_ptr[bin_index] = 0;
                }
            }

            // Fixup remaining bins
            for (std::size_t i = 0; i < num_bins; ++i) {
                if (bin_ptr[i] != 0) {
                    std::memcpy(
                        output + counts[i] - BlockSize, local_bins + i * BlockSize, bin_ptr[i] * sizeof(T));
                }
            }
        }
    }
};

template <typename T>
void sort_ips4o(
    T const *data, T *__restrict output, std::size_t num_elements, std::size_t num_bins,
    CountingSortTimers &timers) {
    std::memcpy(output, data, num_elements * sizeof(T));
    ips4o::sort(output, output + num_elements);
}

template <typename Fn> void benchmark_counting_sort_aligned(benchmark::State &state, Fn &&fn) {
    std::size_t num_points = 1 << state.range(0);
    std::size_t num_bins = 256;

    auto num_points_output = fn.max_output_size(num_points, num_bins);

    auto data_holder = finufft::allocate_aligned_array<std::size_t>(num_points, 64);
    auto output_holder = finufft::allocate_aligned_array<std::size_t>(num_points_output, 64);

    auto data = data_holder.get();
    auto output = output_holder.get();

    finufft::testing::fill_random_uniform(data, num_points, 0);

    finufft::TimerRoot root("bench_sort_counting");
    CountingSortTimers timers(root.make_timer(""));

    for (auto _ : state) {
        fn(data, output, num_points, num_bins, timers);
        benchmark::DoNotOptimize(output[num_points - 1]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_points);
    state.SetBytesProcessed(state.iterations() * num_points * sizeof(std::size_t));

    auto timer_results = root.report("/");
    for (auto const &[name, time] : timer_results) {
        state.counters[name] = benchmark::Counter(
            std::chrono::duration<double>(time).count(), benchmark::Counter::kIsRate);
    }
}



template <typename Fn> struct NoAlignAdapter {
    Fn fn;

    std::size_t max_output_size(std::size_t num_elements, std::size_t num_bins) const {
        return num_elements;
    }

    template <typename T>
    void operator()(
        std::type_identity_t<T> const *data, T *__restrict output, std::size_t num_elements,
        std::size_t num_bins, CountingSortTimers &timers) const {
        fn(data, output, num_elements, num_bins, timers);
    }
};

template <typename Fn> void benchmark_counting_sort(benchmark::State &state, Fn &&fn) {
    benchmark_counting_sort_aligned(state, NoAlignAdapter<Fn>{std::forward<Fn>(fn)});
}

void bm_vanilla(benchmark::State &state) {
    benchmark_counting_sort(state, &counting_sort_vanilla<std::size_t>);
}

template <std::size_t BlockSize> void bm_blocked(benchmark::State &state) {
    benchmark_counting_sort(state, &counting_sort_blocked<std::size_t, BlockSize>);
}

template <std::size_t BlockSize> void bm_blocked_aligned(benchmark::State &state) {
    benchmark_counting_sort_aligned(state, BlockedAlignedCountingSort<BlockSize>{});
}

void bm_ips4o(benchmark::State &state) {
    benchmark_counting_sort(state, &sort_ips4o<std::size_t>);
}

} // namespace

BENCHMARK(bm_vanilla)->Arg(20)->Arg(22)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bm_blocked, 16)->Arg(20)->Arg(22)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bm_blocked, 64)->Arg(20)->Arg(22)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bm_blocked, 256)->Arg(20)->Arg(22)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(bm_blocked_aligned, 64)->Arg(20)->Arg(22)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_ips4o)->Arg(20)->Arg(22)->Unit(benchmark::kMillisecond);
