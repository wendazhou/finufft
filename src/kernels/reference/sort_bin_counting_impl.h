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

namespace finufft {
namespace spreading {
namespace reference {
namespace detail {

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
    nu_point_collection<Dim, const T> const &input, BinIndexFunctor const &compute_bin_index,
    ProcessBinFunctor&& process_bin_index,
    WriteTransformedCoordinate const &write_transformed_coordinate) {

    constexpr std::size_t unroll = BinIndexFunctor::unroll;
    typedef typename BinIndexFunctor::index_type index_type;
    typedef typename WriteTransformedCoordinate::value_type value_type;

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

template <typename T, std::size_t Dim, std::size_t Unroll> struct WriteTransformedCoordinateScalar {
    typedef std::array<std::array<T, Unroll>, Dim> value_type;

    void operator()(value_type &v, std::size_t d, std::size_t j, T const &t) const { v[d][j] = t; }
};

/** Reorders points into sorted order by using the given set of partial histogram sums.
 *
 */
template <
    typename T, std::size_t Dim, typename BinIndexFunctor, typename WriteTransformedCoordinate>
void move_points_by_histogram_impl(
    tcb::span<std::size_t> histogram, nu_point_collection<Dim, const T> const &input,
    nu_point_collection<Dim, T> const &output, BinIndexFunctor const &compute_bin_index,
    WriteTransformedCoordinate const &write_transformed_coordinate) {
    auto move_points = [&](nu_point_collection<Dim, const T> const &input,
                           std::size_t i,
                           std::size_t limit,
                           auto const &bin_index_value,
                           auto partial) {
        for (std::size_t j = 0; j < limit; ++j) {
            auto b = bin_index_value.bin_index[j];
            auto output_index = --histogram[b];

            for (std::size_t d = 0; d < Dim; ++d) {
                output.coordinates[d][output_index] = bin_index_value.value[d][j];
            }

            output.strengths[2 * output_index] = input.strengths[2 * (i + j)];
            output.strengths[2 * output_index + 1] = input.strengths[2 * (i + j) + 1];
        }
    };

    process_bin_function<T, Dim>(
        input, compute_bin_index, move_points, write_transformed_coordinate);
}

/** Parametrized single-threaded counting sort with direct data movement.
 * 
 * This function provides a generic implementation of a counting sort, based
 * on the given index computation. The function is provided here to enable
 * optimizations by adapting the `BinIndexFunctor` (and the associated `WriteTransformCoordinate`)
 * parameters.
 * 
 * This function does not attempt to use any kind of multithreading or adapt the data movement,
 * and is thus only suitable for small problems.
 * 
 */
template <
    typename T, std::size_t Dim, typename BinIndexFunctor, typename WriteTransformedCoordinate>
void nu_point_counting_sort_direct_singlethreaded_impl(
    nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
    std::size_t *num_points_per_bin, IntBinInfo<T, Dim> const &info,
    BinIndexFunctor const &compute_bin_index,
    WriteTransformedCoordinate const &write_transformed_coordinate) {

    auto histogram_alloc = allocate_aligned_array<std::size_t>(info.num_bins_total(), 64);
    auto histogram = tcb::span<std::size_t>(histogram_alloc.get(), info.num_bins_total());
    std::memset(histogram.data(), 0, histogram.size_bytes());

    compute_histogram_impl(input, histogram, compute_bin_index);
    std::copy(histogram.begin(), histogram.end(), num_points_per_bin);

    std::partial_sum(histogram.begin(), histogram.end(), histogram.begin());

    move_points_by_histogram_impl(
        histogram, input, output, compute_bin_index, write_transformed_coordinate);
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

    nu_point_collection<Dim, T> const &output;
    tcb::span<std::size_t> histogram;

    MovePointsBlocked(tcb::span<std::size_t> histogram, nu_point_collection<Dim, T> const &output)
        : buffer_(finufft::allocate_aligned_array<T>(histogram.size() * BlockSize * (Dim + 2), 64)),
          block_ptr_(histogram.size()), block_size_(histogram.size()),
          bin_buffer_stride(BlockSize * (Dim + 2)), output(output), histogram(histogram) {

        auto AlignElements = (64 / sizeof(T));

        for (std::size_t i = 0; i < histogram.size(); ++i) {
            // set the initial block size to the remainder of the histogram size.
            // This ensures that after processing a full block of this size,
            // the remainder of the blocks are aligned.
            auto remainder = histogram[i] % AlignElements;
            block_size_[i] = BlockSize - (remainder ? (AlignElements - remainder) : 0);
            block_ptr_[i] = block_size_[i];
        }
    }

    template <typename BinIndexValue, bool Final>
    void operator()(
        nu_point_collection<Dim, const T> const &input, std::size_t i, std::size_t limit,
        BinIndexValue const &bin_index_value, std::integral_constant<bool, Final>) {

        for (std::size_t j = 0; j < limit; ++j) {
            auto bin_index = bin_index_value.bin_index[j];

            auto local_bin_offset = --block_ptr_[bin_index];

            auto local_buffer = bin_index * bin_buffer_stride;

            for (std::size_t d = 0; d < Dim; ++d) {
                buffer_[local_buffer + d * BlockSize + local_bin_offset] =
                    bin_index_value.value[d][j];
            }
            buffer_[local_buffer + Dim * BlockSize + 2 * local_bin_offset] =
                input.strengths[2 * (i + j)];
            buffer_[local_buffer + Dim * BlockSize + 2 * local_bin_offset + 1] =
                input.strengths[2 * (i + j) + 1];

            if (local_bin_offset == 0) {
                auto this_block_size = block_size_[bin_index];

                // trigger copy
                auto offset = histogram[bin_index] - this_block_size;
                histogram[bin_index] = offset;

                for (std::size_t d = 0; d < Dim; ++d) {
                    std::memcpy(
                        output.coordinates[d] + offset,
                        buffer_.get() + local_buffer + d * BlockSize,
                        this_block_size * sizeof(T));
                }
                std::memcpy(
                    output.strengths + 2 * offset,
                    buffer_.get() + local_buffer + Dim * BlockSize,
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
                auto offset = histogram[bin_index] - partial_block_size;

                for (std::size_t d = 0; d < Dim; ++d) {
                    std::memcpy(
                        output.coordinates[d] + offset,
                        buffer_.get() + bin_index * bin_buffer_stride + d * BlockSize + local_offset,
                        partial_block_size * sizeof(T));
                }
                std::memcpy(
                    output.strengths + 2 * offset,
                    buffer_.get() + bin_index * bin_buffer_stride + Dim * BlockSize +
                        2 * local_offset,
                    2 * partial_block_size * sizeof(T));
            }
        }
    }
};

template <
    typename T, std::size_t Dim, typename BinIndexFunctor, typename WriteTransformedCoordinate>
void move_points_by_histogram_impl_blocked(
    tcb::span<std::size_t> histogram, nu_point_collection<Dim, const T> const &input,
    nu_point_collection<Dim, T> const &output, BinIndexFunctor const &compute_bin_index,
    WriteTransformedCoordinate const &write_transformed_coordinate) {

    MovePointsBlocked<T, Dim, 64> move_points(histogram, output);
    detail::process_bin_function<T, Dim>(
        input, compute_bin_index, std::move(move_points), write_transformed_coordinate);
}

template <
    typename T, std::size_t Dim, typename BinIndexFunctor, typename WriteTransformedCoordinate>
void nu_point_counting_sort_blocked_singlethreaded_impl(
    nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
    std::size_t *num_points_per_bin, IntBinInfo<T, Dim> const &info,
    BinIndexFunctor const &compute_bin_index,
    WriteTransformedCoordinate const &write_transformed_coordinate) {

    auto histogram_alloc = allocate_aligned_array<std::size_t>(info.num_bins_total(), 64);
    auto histogram = tcb::span<std::size_t>(histogram_alloc.get(), info.num_bins_total());
    std::memset(histogram.data(), 0, histogram.size_bytes());

    compute_histogram_impl(input, histogram, compute_bin_index);
    std::copy(histogram.begin(), histogram.end(), num_points_per_bin);

    std::partial_sum(histogram.begin(), histogram.end(), histogram.begin());

    move_points_by_histogram_impl_blocked(
        histogram, input, output, compute_bin_index, write_transformed_coordinate);
}

} // namespace detail
} // namespace reference
} // namespace spreading
} // namespace finufft
