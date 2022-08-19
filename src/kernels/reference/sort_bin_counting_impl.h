#pragma once

/** @file
 *
 * Header-based parametrized implementations for bin-sorting through counting sorts.
 * These implementations are provided in order to facilitate the implementation of
 * optimized kernels which may only replace specific inner-loop parts.
 *
 */

#include <algorithm>
#include <cstring>
#include <tuple>
#include <type_traits>

#include "../../memory.h"
#include "sort_bin_counting.h"

#ifdef __cpp_lib_hardware_interference_size
#include <new>
#endif

namespace finufft {
namespace spreading {
namespace reference {
namespace detail {

#ifdef __cpp_lib_hardware_interference_size
using hardware_destructive_interference_size = std::hardware_destructive_interference_size;
#else
constexpr std::size_t hardware_destructive_interference_size = 64;
#endif

template <std::size_t S, typename IndexType, typename T> struct BinIndexValue {
    alignas(64) IndexType bin_index[S];
    [[no_unique_address]] T value;
};

/** Generic blocked loop over non-uniform points with bin index computation.
 *
 * This function implements a generic partially unrolled loop over non-uniform points,
 * computing the bin index for each point and process it in some fashion.
 * Additionally, it may implement functionality to interleave the computation of the bin
 * index and its processing in order to avoid dependencies that are too tight.
 *
 * This function is abstracted into three parts of the function:
 * - process_bin_index
 *      Computes the bin index for the given set of points, and additionally
 *      stores additional information based on the transformed (folded) coordinates.
 *      This argument should be a callable object which takes the following arguments:
 *      - nu_point_collection: the input collection of non-uniform points
 *      - i: the index of the point in the set to process
 *      - limit: the number of points to process starting from i. May be ignored if
 *          the partial argument is set to false.
 *      - bin_index: a reference to an aligned array of IndexType, which will be filled
 *          with the bin index for the given set of points.
 *      - partial: a boolean flag indicating whether the loop is partial or not.
 *
 */
template <
    typename T, std::size_t Dim, typename BinIndexFunctor, typename ProcessBinFunctor,
    typename WriteTransformedCoordinate>
void process_bin_function(
    nu_point_collection<Dim, const T> const &input, BinIndexFunctor &&compute_bin_index,
    ProcessBinFunctor &&process_bin_index,
    WriteTransformedCoordinate &&write_transformed_coordinate) {

    constexpr std::size_t unroll = std::remove_reference_t<BinIndexFunctor>::unroll;
    typedef typename std::remove_reference_t<BinIndexFunctor>::index_type index_type;
    typedef typename std::remove_reference_t<WriteTransformedCoordinate>::value_type value_type;

    BinIndexValue<unroll, index_type, value_type> bin_index_value;

    auto run_loop = [&](std::size_t i, std::size_t n, auto partial) {
        compute_bin_index(
            input,
            i,
            n,
            bin_index_value.bin_index,
            partial,
            [&](std::size_t d, std::size_t j, auto const &x) {
                write_transformed_coordinate(bin_index_value.value, d, j, x);
            });

        process_bin_index(input, i, n, bin_index_value, partial);
    };

    std::size_t i = 0;
    for (; i < input.num_points - unroll + 1; i += unroll) {
        run_loop(i, unroll, std::integral_constant<bool, false>{});
    }

    {
        // loop tail
        if (i < input.num_points) {
            run_loop(i, input.num_points - i, std::integral_constant<bool, true>{});
        }
    }
}

struct NoOpWriteTransformedCoordinate {
    // Do not store any value in the transformed coordinate.
    typedef std::tuple<> value_type;

    template <typename T>
    void operator()(value_type &v, std::size_t d, std::size_t j, T &&t) const {}
};

/** Computes bin histogram using given bin index computation.
 *
 * This provides a reference implementation based on a partially unrolled
 * bin index computation (in order to support vectorized bin index computations).
 *
 * @param input Input array of non-uniform points
 * @param histogram Output array of bin histograms
 * @param compute_bin_index Function to compute bin index for each point
 *
 */
template <typename T, std::size_t Dim, typename BinIndexFunctor>
void compute_histogram_impl(
    nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram,
    BinIndexFunctor const &compute_bin_index) {

    auto scatter_histogram = [&](nu_point_collection<Dim, const T> const &input,
                                 std::size_t i,
                                 std::size_t limit,
                                 auto const &bin_index_value,
                                 auto partial) {
        for (std::size_t j = 0; j < limit; ++j) {
            ++histogram[bin_index_value.bin_index[j]];
        }
    };

    process_bin_function(
        input, compute_bin_index, scatter_histogram, NoOpWriteTransformedCoordinate{});
}

/** Generic implementation of the elements of a counting sort.
 *
 * This functor captures all state (except histogram) associated with
 * the implementation of a counting sort. It decomposes the implementation
 * into three parametrized implementations:
 * - ComputeBinIndex: computes the bin index for each point in the input,
 *    (in a potentially batched fashion).
 * - WriteTransformedCoordinate: adapter for bin index computation to extract
 *    the transformed (i.e. fold-rescaled) coordinate from the bin index computation.
 * - MovePoints: moves the points in the input to their final location in the output,
 *    based on the computed histogram during the first phase.
 *
 * Note that this class is parametrized by an inner template parameter
 * corresponding to the range of the input point.
 *
 */
template <
    typename T, std::size_t Dim, template <FoldRescaleRange> typename ComputeBinIndex,
    typename WriteTransformedCoordinate, typename MovePoints>
struct NuSortImpl {
    template <FoldRescaleRange input_range> struct Impl {
        [[no_unique_address]] ComputeBinIndex<input_range> compute_bin_index_;
        [[no_unique_address]] WriteTransformedCoordinate write_transformed_coordinate_;
        [[no_unique_address]] MovePoints move_points_;

        explicit Impl(IntBinInfo<T, Dim> const &info)
            : compute_bin_index_(info), write_transformed_coordinate_(), move_points_(info) {}

        void compute_histogram(
            nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram) {
            compute_histogram_impl(input, histogram, compute_bin_index_);
        }

        void move_points_by_histogram(
            tcb::span<std::size_t> histogram, nu_point_collection<Dim, const T> const &input,
            nu_point_collection<Dim, T> const &output) {

            move_points_.initialize(input, histogram, output);
            process_bin_function<T, Dim>(
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

/** Multi-threaded implementation of a bin-sorting harness based on OMP parallelization.
 * 
 * This class implements a generic parametrized bin-sort through a counting sort strategy,
 * parametrized by the histogram counting and data movement implementations.
 * The parallelization strategy is based on a partitioning of the array into slices
 * between threads, which are separately scanned. The process then joins the histogram
 * data from different slices together, and computes global target directions.
 * Finally, each thread moves the points in its slice to their final location in the output.
 * 
 */
template <typename T, std::size_t Dim, typename Impl> struct SortPointsOmpImpl {
    std::vector<Impl> impl_;
    finufft::aligned_unique_array<std::size_t> histogram_alloc_;
    std::size_t num_bins_;

    explicit SortPointsOmpImpl(IntBinInfo<T, Dim> const &info) : num_bins_(info.num_bins_total()) {
        auto max_threads = omp_get_max_threads();
        impl_.reserve(max_threads);
        for (std::size_t i = 0; i < max_threads; ++i) {
            impl_.emplace_back(info);
        }

        auto bins_rounded = finufft::round_to_next_multiple(
            num_bins_, hardware_destructive_interference_size / sizeof(std::size_t));
        histogram_alloc_ =
            finufft::allocate_aligned_array<std::size_t>(bins_rounded * max_threads, 64);
    }

    void operator()(
        nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
        std::size_t *num_points_per_bin) {

        auto histogram_stride = finufft::round_to_next_multiple(
            num_bins_, hardware_destructive_interference_size / sizeof(std::size_t));

#pragma omp parallel num_threads(impl_.size())
        {
            // get local histogram
            auto histogram = tcb::span<std::size_t>(
                histogram_alloc_.get() + omp_get_thread_num() * histogram_stride, num_bins_);
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

            auto &impl = impl_[omp_get_thread_num()];

            // Compute histogram for all relevant slices
            if (input_thread.num_points > 0) {
                impl.compute_histogram(input_thread, histogram);
            }
#pragma omp barrier

#pragma omp single
            {
                std::size_t *histogram_global = histogram_alloc_.get();

                // Process histograms
                std::size_t accumulator = 0;
                for (std::size_t i = 0; i < num_bins_; ++i) {
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
            impl.move_points_by_histogram(histogram, input_thread, output);
        }
    }
};


template <
    typename T, std::size_t Dim, template <typename, std::size_t, typename> typename Sort,
    template <FoldRescaleRange> typename Impl>
SortPointsPlannedFunctor<T, Dim>
make_sort_functor(FoldRescaleRange const &input_range, IntBinInfo<T, Dim> const &info) {
    if (input_range == FoldRescaleRange::Identity) {
        return Sort<T, Dim, Impl<FoldRescaleRange::Identity>>(info);
    } else {
        return Sort<T, Dim, Impl<FoldRescaleRange::Pi>>(info);
    }
}

template <typename T, std::size_t Dim> struct MovePointsDirect {
    nu_point_collection<Dim, const T> input_;
    nu_point_collection<Dim, T> output_;
    tcb::span<std::size_t> histogram_;

    explicit MovePointsDirect(IntBinInfo<T, Dim> const &info) {}

    void initialize(
        nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram,
        nu_point_collection<Dim, T> const &output) {
        input_ = input;
        output_ = output;
        histogram_ = histogram;
    }

    template <typename BinIndexValue, bool Final>
    void operator()(
        // TODO: remove this argument
        nu_point_collection<Dim, const T> const &, std::size_t i, std::size_t limit,
        BinIndexValue const &bin_index_value, std::integral_constant<bool, Final>) {
        for (std::size_t j = 0; j < limit; ++j) {
            auto b = bin_index_value.bin_index[j];
            auto output_index = --histogram_[b];

            for (std::size_t d = 0; d < Dim; ++d) {
                output_.coordinates[d][output_index] = bin_index_value.value[d][j];
            }
            std::memcpy(
                output_.strengths + 2 * output_index,
                input_.strengths + 2 * (i + j),
                2 * sizeof(T));
        }
    }
};

/** Generic implementation for single-threaded sort.
 *
 */
template <typename T, std::size_t Dim, typename Impl>
void nu_point_counting_sort_singlethreaded_impl(
    nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
    std::size_t *num_points_per_bin, std::size_t num_bins, Impl &impl) {

    auto histogram_alloc = allocate_aligned_array<std::size_t>(num_bins, 64);
    auto histogram = tcb::span<std::size_t>(histogram_alloc.get(), num_bins);
    std::memset(histogram.data(), 0, histogram.size_bytes());

    impl.compute_histogram(input, histogram);

    std::copy(histogram.begin(), histogram.end(), num_points_per_bin);
    std::partial_sum(histogram.begin(), histogram.end(), histogram.begin());

    impl.move_points_by_histogram(histogram, input, output);
}


/** Reorders points into sorted order by using the given set of partial histogram sums.
 *
 * This implementation uses a blocking strategy in order to more efficiently move
 * data which is not in cache.
 *
 */
template <typename T, std::size_t Dim, std::size_t BlockSize> struct MovePointsBlocked {

    /** The local buffer is built by taking views into the buffer array.
     *
     * The buffer is separated into segments of size (2 + Dim) * BlockSize,
     * representing the non-uniform points in the given bin.
     *
     * Each block is represented in a structure of array format,
     * with the coordinates being represented in contiguous arrays of lengeth BLockSize,
     * followed by the strengths.
     *
     * The local buffer is filled from the back towards the front of the array.
     * To keep track of the buffer position, the `block_ptr_` member is used
     * to track an offset from the back of the buffer.
     * That is, `block_ptr_[i]` points one-past the next insertion point in the buffer
     * for block `i`.
     * Additionally, the initial value of `block_ptr_` is kept in `block_size_`.
     * This is only relevant for the first block, which may be a partial block
     * in order to bring the offsets into 64 byte alignment to ensure
     * optimal copy.
     *
     */
    finufft::aligned_unique_array<T> buffer_;
    std::vector<std::uint32_t> block_ptr_;
    std::vector<std::uint32_t> block_size_;

    std::size_t bin_buffer_stride;

    nu_point_collection<Dim, const T> input_;
    nu_point_collection<Dim, T> output_;
    tcb::span<std::size_t> histogram_;

    MovePointsBlocked(IntBinInfo<T, Dim> const &info)
        : buffer_(finufft::allocate_aligned_array<T>(
              info.num_bins_total() * BlockSize * (Dim + 2), 64)),
          block_ptr_(info.num_bins_total()), block_size_(info.num_bins_total()),
          bin_buffer_stride(BlockSize * (Dim + 2)) {}

    void initialize(
        nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram,
        nu_point_collection<Dim, T> const &output) {
        input_ = input;
        histogram_ = histogram;
        output_ = output;

        auto AlignElements = (64 / sizeof(T));

        for (std::size_t i = 0; i < histogram_.size(); ++i) {
            // set the initial block size to the remainder of the histogram size.
            // This ensures that after processing a full block of this size,
            // the remainder of the blocks are aligned.
            auto remainder = histogram_[i] % AlignElements;
            block_size_[i] = BlockSize - (remainder ? (AlignElements - remainder) : 0);
            block_ptr_[i] = block_size_[i];
        }
    }

    template <typename BinIndexValue, bool Final>
    void operator()(
        // TODO: refactor to remove unused first parameter
        nu_point_collection<Dim, const T> const &, std::size_t i, std::size_t limit,
        BinIndexValue const &bin_index_value, std::integral_constant<bool, Final>) {

        // Get buffer pointer, mark with restrict to help with optimizations.
        T *__restrict buffer = buffer_.get();

        for (std::size_t j = 0; j < limit; ++j) {
            auto bin_index = bin_index_value.bin_index[j];
            auto local_bin_offset = --block_ptr_[bin_index];

            auto local_buffer = bin_index * bin_buffer_stride;

            // Copy point data into corresponding local buffer.
            for (std::size_t d = 0; d < Dim; ++d) {
                buffer[local_buffer + d * BlockSize + local_bin_offset] =
                    bin_index_value.value[d][j];
            }
            std::memcpy(
                buffer + Dim * BlockSize + 2 * local_bin_offset,
                input_.strengths + 2 * (i + j),
                2 * sizeof(T));

            if (local_bin_offset == 0) {
                // Buffer for current bin is full, trigger copy into output.
                auto this_block_size = block_size_[bin_index];

                // trigger copy
                auto offset = histogram_[bin_index] - this_block_size;
                histogram_[bin_index] = offset;

                for (std::size_t d = 0; d < Dim; ++d) {
                    std::memcpy(
                        output_.coordinates[d] + offset,
                        buffer + local_buffer + d * BlockSize,
                        this_block_size * sizeof(T));
                }
                std::memcpy(
                    output_.strengths + 2 * offset,
                    buffer + local_buffer + Dim * BlockSize,
                    2 * this_block_size * sizeof(T));

                block_ptr_[bin_index] = BlockSize;
                block_size_[bin_index] = BlockSize;
            }

            if (Final) {
                finalize_bins();
            }
        }
    }

    void finalize_bins() {
        // Loop over all local blocks to flush remaining data into main buffer.
        for (std::size_t bin_index = 0; bin_index < block_ptr_.size(); ++bin_index) {
            auto this_block_size = block_size_[bin_index];
            auto local_offset = block_ptr_[bin_index];

            if (local_offset != this_block_size) {
                auto partial_block_size = (this_block_size - local_offset);
                auto offset = histogram_[bin_index] - partial_block_size;

                for (std::size_t d = 0; d < Dim; ++d) {
                    std::memcpy(
                        output_.coordinates[d] + offset,
                        buffer_.get() + bin_index * bin_buffer_stride + d * BlockSize +
                            local_offset,
                        partial_block_size * sizeof(T));
                }
                std::memcpy(
                    output_.strengths + 2 * offset,
                    buffer_.get() + bin_index * bin_buffer_stride + Dim * BlockSize +
                        2 * local_offset,
                    2 * partial_block_size * sizeof(T));
            }
        }
    }
};


} // namespace detail
} // namespace reference
} // namespace spreading
} // namespace finufft
