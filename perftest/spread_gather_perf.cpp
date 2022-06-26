#include <benchmark/benchmark.h>

#include "../test/spread_test_utils.h"

namespace {

int64_t num_bytes_read_gather_rescale(std::size_t dim, std::size_t num_points, std::size_t size) {
    // Total bytes read:
    //      2 * size * num_points for the strengths
    //      dim * size * num_points for the coordinates
    return (2 + dim) * num_points * size;
}

template <std::size_t Dim, typename T, typename Fn>
void gather_rescale_impl(benchmark::State &state, Fn &&fn) {
    std::size_t num_points = state.range(0);
    auto points = make_random_point_collection<Dim, T>(num_points, 0, {-1.0, 2.0});
    auto permutation = make_random_permutation(num_points, 1);

    finufft::spreading::SpreaderMemoryInput<Dim, T> output(num_points / 2);
    std::array<int64_t, Dim> sizes;
    std::fill(sizes.begin(), sizes.end(), 1);

    auto const &points_arg = points.cast([](auto &&x) { return static_cast<T const *>(x.get()); });

    for (auto _ : state) {
        fn(output,
           points_arg,
           sizes,
           permutation.data(),
           finufft::spreading::FoldRescaleRange::Identity);
    }

    state.SetBytesProcessed(
        state.iterations() * num_bytes_read_gather_rescale(Dim, num_points, sizeof(T)));
}

template <std::size_t Dim, typename T> void gather_rescale_reference(benchmark::State &state) {
    gather_rescale_impl<Dim, T>(state, &finufft::spreading::gather_and_fold<Dim, int64_t, T>);
}

} // namespace

BENCHMARK(gather_rescale_reference<1, double>)->RangeMultiplier(16)->Range(2 << 12, 2 << 18);
BENCHMARK(gather_rescale_reference<2, double>)->RangeMultiplier(16)->Range(2 << 12, 2 << 18);
BENCHMARK(gather_rescale_reference<3, double>)->RangeMultiplier(16)->Range(2 << 12, 2 << 18);

BENCHMARK(gather_rescale_reference<1, float>)->RangeMultiplier(16)->Range(2 << 12, 2 << 18);
BENCHMARK(gather_rescale_reference<2, float>)->RangeMultiplier(16)->Range(2 << 12, 2 << 18);
BENCHMARK(gather_rescale_reference<3, float>)->RangeMultiplier(16)->Range(2 << 12, 2 << 18);
