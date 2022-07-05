#include <array>
#include <limits>
#include <stdexcept>

#include "../../bit.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE                                                                         \
    "spread_bin_sort_hwy_impl.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "fold_rescale.inl.h"

HWY_BEFORE_NAMESPACE();
namespace finufft {
namespace spreading {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T, std::size_t Dim> struct ComputeBinIndex;

template <std::size_t Dim> struct ComputeBinIndex<float, Dim> {
    int64_t *index;
    std::array<float const *, Dim> coordinates;
    std::array<float, Dim> extents;
    std::array<float, Dim> bin_scaling;
    std::array<std::size_t, Dim> stride;
    std::size_t num_point_bits;

    ComputeBinIndex(
        std::size_t num_points, int64_t *index, std::array<float const *, Dim> coordinates,
        std::array<float, Dim> const &extents, std::array<float, Dim> const &bin_sizes)
        : coordinates(coordinates), extents(extents), num_point_bits(bit_width(num_points)) {
        std::array<std::size_t, Dim> num_bins;
        for (std::size_t i = 0; i < Dim; ++i) {
            num_bins[i] = std::size_t(extents[i] / bin_sizes[i]) + 1;
            bin_scaling[i] = float(1. / bin_sizes[i]);
        }

        stride[0] = 1;
        for (size_t i = 0; i < Dim; ++i) {
            stride[i] = stride[i - 1] * num_bins[i - 1];
        }

        size_t bins_total = stride[Dim - 1] * num_bins[Dim - 1];

        if (bit_width(bins_total) + num_point_bits > std::numeric_limits<int64_t>::digits) {
            throw std::runtime_error("Too many bins for the given number of points");
        }
    }

    void operator()(std::size_t i) {
        hn::ScalableTag<float> d;
        hn::ScalableTag<uint32_t> di;
        hn::ScalableTag<uint64_t> di64;
        hn::ScalableTag<int32_t> di_s;

        FoldRescalePi<float> fold_rescale;

        auto bin_index_even = hn::Zero<uint64_t>(di64);
        auto bin_index_odd = hn::Zero<uint64_t>(di64);

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            auto folded = fold_rescale(hn::LoadU(d, coordinates[dim] + i), extents[dim], d);
            auto bin_floating = hn::Mul(folded, hn::Set(d, bin_scaling[dim]));
            auto bin_int = hn::BitCast(di, hn::ConvertTo(di_s, bin_floating));

            auto bin_index_even_dim = hn::MulEven(bin_int, hn::Set(di, stride[dim]));
            auto bin_index_odd_dim =
                hn::MulEven(hn::Reverse2(di, bin_int), hn::Set(di, stride[dim]));

            bin_index_even = hn::Add(bin_index_even, bin_index_even_dim);
            bin_index_odd = hn::Add(bin_index_odd, bin_index_odd_dim);
        }

        hn::StoreInterleaved2(
            bin_index_even, bin_index_odd, di64, reinterpret_cast<uint64_t *>(index) + i);
    }
};

template <typename T, std::size_t Dim, typename FoldRescale>
void compute_bin_index_impl(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const &extents, std::array<T, Dim> const &bin_sizes,
    FoldRescale &&fold_rescale) {

    ComputeBinIndex<T, Dim> compute_bin_index(num_points, index, coordinates, extents, bin_sizes);

    hn::ScalableTag<T> d;
    auto n = hn::Lanes(d);
    for (std::size_t i = 0; i + n < num_points; i += n) {
        compute_bin_index(i);
    }
}

template void compute_bin_index_impl<float, 1>(
    int64_t *index, std::size_t num_points, std::array<float const *, 1> const &coordinates,
    std::array<float, 1> const &extents, std::array<float, 1> const &bin_sizes,
    FoldRescalePi<float> &&fold_rescale);

template void compute_bin_index_impl<float, 2>(
    int64_t *index, std::size_t num_points, std::array<float const *, 2> const &coordinates,
    std::array<float, 2> const &extents, std::array<float, 2> const &bin_sizes,
    FoldRescalePi<float> &&fold_rescale);

template void compute_bin_index_impl<float, 3>(
    int64_t *index, std::size_t num_points, std::array<float const *, 3> const &coordinates,
    std::array<float, 3> const &extents, std::array<float, 3> const &bin_sizes,
    FoldRescalePi<float> &&fold_rescale);

} // namespace HWY_NAMESPACE
} // namespace spreading
} // namespace finufft
