#include "sort_bin_counting.h"
#include "sort_bin_counting_impl.h"

#include "../../memory.h"
#include "../sorting.h"
#include "../spreading.h"

#include <array>
#include <cmath>
#include <cstring>

#include <libdivide.h>
#include <tcb/span.hpp>

#include "gather_fold_reference.h"

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
            input,
            histogram,
            ComputeBinIndex<64 / sizeof(T), T, Dim, FoldRescaleIdentity<T>>(
                info, FoldRescaleIdentity<T>{}));
    } else {
        finufft::spreading::reference::detail::compute_histogram_impl(
            input,
            histogram,
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

template <typename T, std::size_t Dim>
void nu_point_counting_sort_direct_singlethreaded(
    nu_point_collection<Dim, const T> const &input, FoldRescaleRange input_range,
    nu_point_collection<Dim, T> const &output, std::size_t *num_points_per_bin,
    IntBinInfo<T, Dim> const &info) {

    const std::size_t unroll = 1;
    auto write_transformed_coordinate = detail::WriteTransformedCoordinateScalar<T, Dim, unroll>{};

    if (input_range == FoldRescaleRange::Identity) {
        detail::nu_point_counting_sort_direct_singlethreaded_impl(
            input,
            output,
            num_points_per_bin,
            info,
            ComputeBinIndex<unroll, T, Dim, FoldRescaleIdentity<T>>(info, FoldRescaleIdentity<T>{}),
            write_transformed_coordinate);
    } else {
        detail::nu_point_counting_sort_direct_singlethreaded_impl(
            input,
            output,
            num_points_per_bin,
            info,
            ComputeBinIndex<unroll, T, Dim, FoldRescalePi<T>>(info, FoldRescalePi<T>{}),
            write_transformed_coordinate);
    }
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
            block_size_[i] = BlockSize - (AlignElements - ((histogram[i] + 1) % AlignElements - 1));
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
                histogram[bin_index] = offset;

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

    detail::compute_histogram_impl(input, histogram, compute_bin_index);
    std::copy(histogram.begin(), histogram.end(), num_points_per_bin);

    std::partial_sum(histogram.begin(), histogram.end(), histogram.begin());

    move_points_by_histogram_impl_blocked(
        histogram, input, output, compute_bin_index, write_transformed_coordinate);
}

template <typename T, std::size_t Dim>
void nu_point_counting_sort_blocked_singlethreaded(
    nu_point_collection<Dim, const T> const &input, FoldRescaleRange input_range,
    nu_point_collection<Dim, T> const &output, std::size_t *num_points_per_bin,
    IntBinInfo<T, Dim> const &info) {

    const std::size_t unroll = 1;
    auto write_transformed_coordinate = detail::WriteTransformedCoordinateScalar<T, Dim, unroll>{};

    if (input_range == FoldRescaleRange::Identity) {
        nu_point_counting_sort_blocked_singlethreaded_impl(
            input,
            output,
            num_points_per_bin,
            info,
            ComputeBinIndex<unroll, T, Dim, FoldRescaleIdentity<T>>(info, FoldRescaleIdentity<T>{}),
            write_transformed_coordinate);
    } else {
        nu_point_counting_sort_blocked_singlethreaded_impl(
            input,
            output,
            num_points_per_bin,
            info,
            ComputeBinIndex<unroll, T, Dim, FoldRescalePi<T>>(info, FoldRescalePi<T>{}),
            write_transformed_coordinate);
    }
}

#define INSTANTIATE(T, Dim)                                                                        \
    template void nu_point_counting_sort_direct_singlethreaded<T, Dim>(                            \
        nu_point_collection<Dim, const T> const &input,                                            \
        FoldRescaleRange input_range,                                                              \
        nu_point_collection<Dim, T> const &output,                                                 \
        std::size_t *num_points_per_bin,                                                           \
        IntBinInfo<T, Dim> const &info);                                                           \
    template void nu_point_counting_sort_blocked_singlethreaded<T, Dim>(                           \
        nu_point_collection<Dim, const T> const &input,                                            \
        FoldRescaleRange input_range,                                                              \
        nu_point_collection<Dim, T> const &output,                                                 \
        std::size_t *num_points_per_bin,                                                           \
        IntBinInfo<T, Dim> const &info);

INSTANTIATE(float, 1);
INSTANTIATE(float, 2);
INSTANTIATE(float, 3);

INSTANTIATE(double, 1);
INSTANTIATE(double, 2);
INSTANTIATE(double, 3);

#undef INSTANTIATE

} // namespace reference
} // namespace spreading
} // namespace finufft
