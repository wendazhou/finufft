#include "sort_bin_counting.h"

#include <libdivide.h>

#include <array>
#include <cmath>
#include <cstring>
#include <tcb/span.hpp>

#include "../../memory.h"
#include "../sorting.h"
#include "../spreading.h"
#include "gather_fold_reference.h"
#include "sort_bin_counting_impl.h"

#ifdef __cpp_lib_hardware_interference_size
#include <new>
using std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

namespace finufft {
namespace spreading {
namespace reference {

template <std::size_t Unroll, typename T, std::size_t Dim, typename FoldRescale>
struct ComputeBinIndex {
    IntBinInfo<T, Dim> info;
    FoldRescale fold_rescale;
    std::array<T, Dim> size_f;
    std::array<libdivide::divider<uint32_t>, Dim> dividers;

    static constexpr std::size_t unroll = Unroll;
    typedef std::uint32_t index_type;

    explicit ComputeBinIndex(IntBinInfo<T, Dim> const &info, FoldRescale const &fold_rescale)
        : info(info), fold_rescale(fold_rescale) {
        std::copy(info.size.begin(), info.size.end(), size_f.begin());
        for (std::size_t j = 0; j < Dim; ++j) {
            dividers[j] = libdivide::divider<uint32_t>(info.bin_size[j]);
        }

        // Basic error checking for valid 32-bit processing of input.
        if (info.num_bins_total() > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("Too many bins");
        }

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            if (info.size[dim] > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("Grid too large");
            }
        }
    }

    template <bool Partial, typename WriteTransformedCoordinate>
    void operator()(
        nu_point_collection<Dim, const T> const &input, std::size_t i, std::size_t limit,
        tcb::span<std::uint32_t, Unroll> bins, std::integral_constant<bool, Partial>,
        WriteTransformedCoordinate &&write_transformed_coordinate) const {
        std::fill(bins.begin(), bins.end(), 0);

        for (std::size_t j = 0; j < Dim; ++j) {
            for (std::size_t offset = 0; offset < (Partial ? limit : Unroll); ++offset) {
                auto x = input.coordinates[j][i + offset];
                x = fold_rescale(x, size_f[j]);

                auto x_c = std::ceil(x - info.offset[j]);
                std::uint32_t x_b =
                    static_cast<uint32_t>(x_c) - static_cast<uint32_t>(info.global_offset[j]);
                x_b /= dividers[j];

                bins[offset] += x_b * info.bin_index_stride[j];
                write_transformed_coordinate(j, offset, x);
            }
        }
    }

    template <bool Partial>
    void operator()(
        nu_point_collection<Dim, const T> const &input, std::size_t i, std::size_t limit,
        tcb::span<std::uint32_t, Unroll> bins,
        std::integral_constant<bool, Partial> partial) const {
        return (*this)(
            input, i, limit, bins, partial, [](std::size_t j, std::size_t offset, T x) {});
    }
};

template <typename T, std::size_t Dim>
void compute_histogram(
    nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram,
    IntBinInfo<T, Dim> const &info, FoldRescaleRange input_range) {
    if (input_range == FoldRescaleRange::Identity) {
        finufft::spreading::reference::detail::compute_histogram_impl(
            input, histogram,
            ComputeBinIndex<64 / sizeof(T), T, Dim, FoldRescaleIdentity<T>>(
                info, FoldRescaleIdentity<T>{}));
    } else {
        finufft::spreading::reference::detail::compute_histogram_impl(
            input, histogram,
            ComputeBinIndex<64 / sizeof(T), T, Dim, FoldRescalePi<T>>(info, FoldRescalePi<T>{}));
    }
}

template void compute_histogram(
    nu_point_collection<1, const float> const &input, tcb::span<std::size_t> histogram,
    IntBinInfo<float, 1> const &info, FoldRescaleRange input_range);

template <typename T, std::size_t Dim, typename BinIndexFunctor>
void move_points_by_histogram(
    tcb::span<std::size_t> histogram, nu_point_collection<Dim, const T> const &input,
    nu_point_collection<Dim, T> const &output, BinIndexFunctor const &compute_bin_index) {
    detail::move_points_by_histogram_impl(histogram, input, output, compute_bin_index);
}

namespace detail {
template <
    typename T, std::size_t Dim, typename BinIndexFunctor, typename WriteTransformedCoordinate>
void nu_point_counting_sort_direct_omp_impl(
    nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
    std::size_t *num_points_per_bin, IntBinInfo<T, Dim> const &info,
    BinIndexFunctor const &compute_bin_index,
    WriteTransformedCoordinate const &write_transformed_coordinate) {

    finufft::aligned_unique_array<std::size_t> histogram_alloc;
    auto histogram_stride = finufft::round_to_next_multiple(
        info.num_bins_total(), hardware_destructive_interference_size / sizeof(std::size_t));

#pragma omp parallel
    {
#pragma omp single
        histogram_alloc = finufft::allocate_aligned_array<std::size_t>(
            histogram_stride * omp_get_num_threads(), 64);

        auto histogram = tcb::span<std::size_t>(
            histogram_alloc.get() + omp_get_thread_num() * histogram_stride, info.num_bins_total());
        std::memset(histogram.data(), 0, histogram.size_bytes());

        auto points_per_thread = finufft::round_to_next_multiple(
            input.num_points / omp_get_num_threads(),
            hardware_destructive_interference_size / sizeof(T));
        auto thread_start = omp_get_thread_num() * points_per_thread;
        auto thread_length = thread_start < input.num_points
                                 ? std::min(points_per_thread, input.num_points - thread_start)
                                 : 0;
        auto input_thread = input.slice(thread_start, thread_length);

        if (input_thread.num_points > 0) {
            compute_histogram_impl(input_thread, histogram, compute_bin_index);
        }
#pragma omp barrier

#pragma omp single
        {
            std::size_t *histogram_global = histogram_alloc.get();

            // Process histograms
            std::size_t accumulator = 0;
            for (std::size_t i = 0; i < info.num_bins_total(); ++i) {
                std::size_t bin_count = 0;

                for (std::size_t j = 0; j < omp_get_num_threads(); ++j) {
                    auto bin_thread_count = histogram_global[j * histogram_stride + i];
                    accumulator += bin_thread_count;
                    bin_count += bin_thread_count;
                    histogram_global[j * histogram_stride + i] = accumulator;
                }

                num_points_per_bin[i] = bin_count;
            }

            assert(accumulator == input.num_points);
        }

        move_points_by_histogram_impl(
            histogram, input_thread, output, compute_bin_index, write_transformed_coordinate);
    }
}

} // namespace detail

#define COUNTING_SORT_SIGNATURE(NAME, T, Dim)                                                      \
    void NAME(                                                                                     \
        nu_point_collection<Dim, const T> const &input, FoldRescaleRange input_range,              \
        nu_point_collection<Dim, T> const &output, std::size_t *num_points_per_bin,                \
        IntBinInfo<T, Dim> const &info)

#define DEFINE_COUNTING_SORT_FROM_IMPL(NAME)                                                       \
    template <typename T, std::size_t Dim> COUNTING_SORT_SIGNATURE(NAME, T, Dim) {                 \
        const std::size_t unroll = 1;                                                              \
        auto write_transformed_coordinate =                                                        \
            detail::WriteTransformedCoordinateScalar<T, Dim, unroll>{};                            \
                                                                                                   \
        if (input_range == FoldRescaleRange::Identity) {                                           \
            detail::NAME##_impl(                                                                   \
                input, output, num_points_per_bin, info,                                           \
                ComputeBinIndex<unroll, T, Dim, FoldRescaleIdentity<T>>(                           \
                    info, FoldRescaleIdentity<T>{}),                                               \
                write_transformed_coordinate);                                                     \
        } else {                                                                                   \
            detail::NAME##_impl(                                                                   \
                input, output, num_points_per_bin, info,                                           \
                ComputeBinIndex<unroll, T, Dim, FoldRescalePi<T>>(info, FoldRescalePi<T>{}),       \
                write_transformed_coordinate);                                                     \
        }                                                                                          \
    }

DEFINE_COUNTING_SORT_FROM_IMPL(nu_point_counting_sort_direct_singlethreaded)
DEFINE_COUNTING_SORT_FROM_IMPL(nu_point_counting_sort_direct_omp)
DEFINE_COUNTING_SORT_FROM_IMPL(nu_point_counting_sort_blocked_singlethreaded)

#undef DEFINE_COUNTING_SORT_FROM_IMPL

#define INSTANTIATE(T, Dim)                                                                        \
    template COUNTING_SORT_SIGNATURE(nu_point_counting_sort_direct_singlethreaded, T, Dim);        \
    template COUNTING_SORT_SIGNATURE(nu_point_counting_sort_direct_omp, T, Dim);                   \
    template COUNTING_SORT_SIGNATURE(nu_point_counting_sort_blocked_singlethreaded, T, Dim);

INSTANTIATE(float, 1);
INSTANTIATE(float, 2);
INSTANTIATE(float, 3);

INSTANTIATE(double, 1);
INSTANTIATE(double, 2);
INSTANTIATE(double, 3);

#undef INSTANTIATE
#undef COUNTING_SORT_SIGNATURE

} // namespace reference
} // namespace spreading
} // namespace finufft
