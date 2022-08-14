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

        #pragma gcc unroll Dim
        for (std::size_t j = 0; j < Dim; ++j) {
            #pragma gcc unroll 4
            #pragma gcc ivdep
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

template <typename T, std::size_t Dim, typename BinIndexFunctor>
void compute_histogram_impl(
    nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram,
    BinIndexFunctor const &compute_bin_index) {
    constexpr std::size_t unroll = BinIndexFunctor::unroll;
    std::array<typename BinIndexFunctor::index_type, unroll> bin_index;

    std::size_t i = 0;
    for (; i < input.num_points - unroll + 1; i += unroll) {
        compute_bin_index(input, i, unroll, bin_index, std::integral_constant<bool, false>{});
        for (std::size_t j = 0; j < BinIndexFunctor::unroll; ++j) {
            ++histogram[bin_index[j]];
        }
    }

    {
        auto remainder = input.num_points - i;
        compute_bin_index(input, i, remainder, bin_index, std::integral_constant<bool, true>{});
        for (std::size_t j = 0; j < remainder; ++j) {
            ++histogram[bin_index[j]];
        }
    }
}

template <typename T, std::size_t Dim>
void compute_histogram(
    nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram,
    IntBinInfo<T, Dim> const &info, FoldRescaleRange input_range) {
    if (input_range == FoldRescaleRange::Identity) {
        compute_histogram_impl(
            input,
            histogram,
            ComputeBinIndex<64 / sizeof(T), T, Dim, FoldRescaleIdentity<T>>(
                info, FoldRescaleIdentity<T>{}));
    } else {
        compute_histogram_impl(
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

    const std::size_t unroll = BinIndexFunctor::unroll;

    std::array<T, unroll * Dim> transformed_coordinates;
    std::array<T *, Dim> transformed_coordinates_ptr;
    for (std::size_t j = 0; j < Dim; ++j) {
        transformed_coordinates_ptr[j] = transformed_coordinates.data() + j * unroll;
    }
    auto write_transformed_coordinate = [&](std::size_t j, std::size_t offset, T x) {
        transformed_coordinates_ptr[j][offset] = x;
    };

    std::array<typename BinIndexFunctor::index_type, unroll> bin_index;

    // unrolled main component
    std::size_t i = 0;
    for (; i < input.num_points - unroll + 1; i += unroll) {
        compute_bin_index(
            input,
            i,
            unroll,
            bin_index,
            std::integral_constant<bool, false>{},
            write_transformed_coordinate);

        for (std::size_t j = 0; j < unroll; ++j) {
            auto b = bin_index[j];
            auto output_index = --histogram[b];

            for (std::size_t k = 0; k < Dim; ++k) {
                output.coordinates[k][output_index] = transformed_coordinates_ptr[k][j];
            }
            output.strengths[2 * output_index] = input.strengths[2 * (i + j)];
            output.strengths[2 * output_index + 1] = input.strengths[2 * (i + j) + 1];
        }
    }

    // tail component
    {
        compute_bin_index(
            input,
            i,
            input.num_points - i,
            bin_index,
            std::integral_constant<bool, true>{},
            write_transformed_coordinate);

        for (std::size_t j = 0; j < input.num_points - i; ++j) {
            auto b = bin_index[j];
            auto output_index = --histogram[b];

            for (std::size_t k = 0; k < Dim; ++k) {
                output.coordinates[k][output_index] = transformed_coordinates_ptr[k][j];
            }
            output.strengths[2 * output_index] = input.strengths[2 * (i + j)];
            output.strengths[2 * output_index + 1] = input.strengths[2 * (i + j) + 1];
        }
    }
}

template <typename T, std::size_t Dim, typename BinIndexFunctor>
void nu_point_counting_sort_direct_singlethreaded_impl(
    nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
    std::size_t *num_points_per_bin, IntBinInfo<T, Dim> const &info,
    BinIndexFunctor const &compute_bin_index) {

    auto histogram_alloc = allocate_aligned_array<std::size_t>(info.num_bins_total(), 64);
    auto histogram = tcb::span<std::size_t>(histogram_alloc.get(), info.num_bins_total());
    std::memset(histogram.data(), 0, histogram.size_bytes());

    compute_histogram_impl(input, histogram, compute_bin_index);
    std::copy(histogram.begin(), histogram.end(), num_points_per_bin);

    std::partial_sum(histogram.begin(), histogram.end(), histogram.begin());

    move_points_by_histogram(histogram, input, output, compute_bin_index);
}

template <typename T, std::size_t Dim>
void nu_point_counting_sort_direct_singlethreaded(
    nu_point_collection<Dim, const T> const &input, FoldRescaleRange input_range,
    nu_point_collection<Dim, T> const &output, std::size_t *num_points_per_bin,
    IntBinInfo<T, Dim> const &info) {
    if (input_range == FoldRescaleRange::Identity) {
        nu_point_counting_sort_direct_singlethreaded_impl(
            input,
            output,
            num_points_per_bin,
            info,
            ComputeBinIndex<4, T, Dim, FoldRescaleIdentity<T>>(
                info, FoldRescaleIdentity<T>{}));
    } else {
        nu_point_counting_sort_direct_singlethreaded_impl(
            input,
            output,
            num_points_per_bin,
            info,
            ComputeBinIndex<4, T, Dim, FoldRescalePi<T>>(info, FoldRescalePi<T>{}));
    }
}

#define INSTANTIATE(T, Dim)                                                                        \
    template void nu_point_counting_sort_direct_singlethreaded<T, Dim>(                            \
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

template <typename T, std::size_t Dim>
void block_aligned_counting_sort(
    nu_point_collection<Dim, const T> const &input, nu_point_collection<Dim, T> const &output,
    IntBinInfo<T, Dim> const &info, FoldRescaleRange input_range) {

    auto histogram = allocate_aligned_array<std::size_t>(info.num_bins_total(), 64);
    auto histogram_span = tcb::span<std::size_t>(histogram.get(), info.num_bins_total());

    compute_histogram(input, histogram_span, info, input_range);

    std::partial_sum(histogram_span.begin(), histogram_span.end(), histogram_span.begin());
}

} // namespace reference
} // namespace spreading
} // namespace finufft
