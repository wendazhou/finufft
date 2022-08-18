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

namespace {
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

template <typename T, FoldRescaleRange input_range> struct FoldRescaleScalar;
template <typename T>
struct FoldRescaleScalar<T, FoldRescaleRange::Identity> : FoldRescaleIdentity<T> {};
template <typename T> struct FoldRescaleScalar<T, FoldRescaleRange::Pi> : FoldRescalePi<T> {};

template <typename T, std::size_t Dim> struct ComputeBinIndexScalar {
    template <FoldRescaleRange input_range>
    struct Impl : ComputeBinIndex<1, T, Dim, FoldRescaleScalar<T, input_range>> {
        Impl(IntBinInfo<T, Dim> const &info)
            : ComputeBinIndex<1, T, Dim, FoldRescaleScalar<T, input_range>>(
                  info, FoldRescaleScalar<T, input_range>{}) {}
    };
};

/** Generic implementation of the elements of a counting sort.
 * 
 * This functor captures all state (except histogram) associated with
 * the implementation of a counting sort.
 *
 */
template <
    typename T, std::size_t Dim, template <FoldRescaleRange> typename ComputeBinIndex,
    typename WriteTransformedCoordinate, typename MovePoints>
struct NuSortImplScalar {
    template <FoldRescaleRange input_range> struct Impl {
        [[no_unique_address]] ComputeBinIndex<input_range> compute_bin_index_;
        [[no_unique_address]] WriteTransformedCoordinate write_transformed_coordinate_;
        [[no_unique_address]] MovePoints move_points_;

        explicit Impl(IntBinInfo<T, Dim> const &info)
            : compute_bin_index_(info), write_transformed_coordinate_(), move_points_(info) {}

        void compute_histogram(
            nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram) {
            detail::compute_histogram_impl(input, histogram, compute_bin_index_);
        }

        void move_points_by_histogram(
            tcb::span<std::size_t> histogram, nu_point_collection<Dim, const T> const &input,
            nu_point_collection<Dim, T> const &output) {

            move_points_.initialize(input, histogram, output);
            detail::process_bin_function<T, Dim>(
                input, compute_bin_index_, move_points_, write_transformed_coordinate_);
        }
    };
};

/** Single-threaded counting sort implementation delegating to given histogram computation
 * and data movement implementations.
 * 
 */
template <typename T, std::size_t Dim, typename Impl> struct SortPointsSingleThreadedImpl {
    Impl impl_;
    finufft::aligned_unique_array<std::size_t> histogram_alloc_;
    tcb::span<std::size_t> histogram_;

    explicit SortPointsSingleThreadedImpl(IntBinInfo<T, Dim> const &info)
        : impl_(info),
          histogram_alloc_(finufft::allocate_aligned_array<std::size_t>(info.num_bins_total(), 64)),
          histogram_(histogram_alloc_.get(), info.num_bins_total()) {}

    void operator()(
        nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
        std::size_t *num_points_per_bin) {

        std::memset(histogram_.data(), 0, histogram_.size_bytes());
        impl_.compute_histogram(input, histogram_);

        std::memcpy(num_points_per_bin, histogram_.data(), histogram_.size_bytes());
        std::partial_sum(histogram_.begin(), histogram_.end(), histogram_.begin());

        impl_.move_points_by_histogram(histogram_, input, output);
    }
};

template <typename T, std::size_t Dim, template <FoldRescaleRange> typename Impl>
SortPointsPlannedFunctor<T, Dim> make_sort_functor_singlethreaded(
    FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info) {
    if (input_range == FoldRescaleRange::Identity) {
        return SortPointsSingleThreadedImpl<T, Dim, Impl<FoldRescaleRange::Identity>>(info);
    } else {
        return SortPointsSingleThreadedImpl<T, Dim, Impl<FoldRescaleRange::Pi>>(info);
    }
}

} // namespace

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_singlethreaded(
    FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info) {
    typedef NuSortImplScalar<
        T,
        Dim,
        ComputeBinIndexScalar<T, Dim>::template Impl,
        detail::WriteTransformedCoordinateScalar<T, Dim, 1>,
        detail::MovePointsDirect<T, Dim>>
        impl_type;

    return make_sort_functor_singlethreaded<T, Dim, impl_type::template Impl>(
        input_range, info);
}

template <typename T, std::size_t Dim>
SortPointsPlannedFunctor<T, Dim> make_sort_counting_blocked_singlethreaded(
    FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info) {
    typedef NuSortImplScalar<
        T,
        Dim,
        ComputeBinIndexScalar<T, Dim>::template Impl,
        detail::WriteTransformedCoordinateScalar<T, Dim, 1>,
        detail::MovePointsBlocked<T, Dim, 128>>
        impl_type;

    return make_sort_functor_singlethreaded<T, Dim, impl_type::template Impl>(
        input_range, info);
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

        // Compute slice to be processed by this thread
        auto points_per_thread = finufft::round_to_next_multiple(
            input.num_points / omp_get_num_threads(),
            hardware_destructive_interference_size / sizeof(T));
        auto thread_start = omp_get_thread_num() * points_per_thread;
        auto thread_length = thread_start < input.num_points
                                 ? std::min(points_per_thread, input.num_points - thread_start)
                                 : 0;
        auto input_thread = input.slice(thread_start, thread_length);

        // Compute histogram for all relevant slices
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

        // Move points to output
        move_points_by_histogram_impl(
            histogram, input_thread, output, compute_bin_index, write_transformed_coordinate);
    }
}

} // namespace detail

#define COUNTING_SORT_SIGNATURE(NAME, T, Dim)                                                      \
    void NAME(                                                                                     \
        nu_point_collection<Dim, const T> const &input,                                            \
        FoldRescaleRange input_range,                                                              \
        nu_point_collection<Dim, T> const &output,                                                 \
        std::size_t *num_points_per_bin,                                                           \
        IntBinInfo<T, Dim> const &info)

#define DEFINE_COUNTING_SORT_FROM_IMPL(NAME)                                                       \
    template <typename T, std::size_t Dim> COUNTING_SORT_SIGNATURE(NAME, T, Dim) {                 \
        const std::size_t unroll = 1;                                                              \
        using WriteTransformedCoordinate =                                                         \
            detail::WriteTransformedCoordinateScalar<T, Dim, unroll>;                              \
        auto write_transformed_coordinate =                                                        \
            detail::WriteTransformedCoordinateScalar<T, Dim, unroll>{};                            \
                                                                                                   \
        if (input_range == FoldRescaleRange::Identity) {                                           \
            using BinIndexFunctor = ComputeBinIndex<unroll, T, Dim, FoldRescaleIdentity<T>>;       \
            detail::DirectSinglethreadedImpl<T, Dim, BinIndexFunctor, WriteTransformedCoordinate>  \
                impl{                                                                              \
                    BinIndexFunctor{info, FoldRescaleIdentity<T>{}},                               \
                    write_transformed_coordinate};                                                 \
            detail::nu_point_counting_sort_singlethreaded_impl(                                    \
                input, output, num_points_per_bin, info.num_bins_total(), impl);                   \
        } else {                                                                                   \
            detail::NAME##_impl(                                                                   \
                input,                                                                             \
                output,                                                                            \
                num_points_per_bin,                                                                \
                info,                                                                              \
                ComputeBinIndex<unroll, T, Dim, FoldRescalePi<T>>(info, FoldRescalePi<T>{}),       \
                write_transformed_coordinate);                                                     \
        }                                                                                          \
    }

DEFINE_COUNTING_SORT_FROM_IMPL(nu_point_counting_sort_direct_omp)

#undef DEFINE_COUNTING_SORT_FROM_IMPL

template <typename T, std::size_t Dim>
COUNTING_SORT_SIGNATURE(nu_point_counting_sort_direct_singlethreaded, T, Dim) {
    return make_sort_counting_direct_singlethreaded<T, Dim>(input_range, info)(
        input, output, num_points_per_bin);
}

template <typename T, std::size_t Dim>
COUNTING_SORT_SIGNATURE(nu_point_counting_sort_blocked_singlethreaded, T, Dim) {
    return make_sort_counting_blocked_singlethreaded<T, Dim>(input_range, info)(
        input, output, num_points_per_bin);
}

#define INSTANTIATE(T, Dim)                                                                        \
    template COUNTING_SORT_SIGNATURE(nu_point_counting_sort_direct_singlethreaded, T, Dim);        \
    template COUNTING_SORT_SIGNATURE(nu_point_counting_sort_direct_omp, T, Dim);                   \
    template COUNTING_SORT_SIGNATURE(nu_point_counting_sort_blocked_singlethreaded, T, Dim);       \
    template SortPointsPlannedFunctor<T, Dim> make_sort_counting_direct_singlethreaded(            \
        FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info);                      \
    template SortPointsPlannedFunctor<T, Dim> make_sort_counting_blocked_singlethreaded(           \
        FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info);

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
