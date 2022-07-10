#include "spread_bin_sort_int.h"

#include <libdivide.h>

namespace finufft {
namespace spreading {
namespace reference {

namespace {
template <typename T, std::size_t Dim, typename FoldRescale> struct ComputeAndPackSingle {
    IntBinInfo<T, Dim> const &info;
    FoldRescale fold_rescale;
    std::array<T, Dim> extents_f;
    std::array<libdivide::divider<uint32_t>, Dim> dividers;

    ComputeAndPackSingle(IntBinInfo<T, Dim> const &info, FoldRescale fold_rescale)
        : info(info), fold_rescale(fold_rescale) {
        std::copy(info.size.begin(), info.size.end(), extents_f.begin());
        for (std::size_t j = 0; j < Dim; ++j) {
            dividers[j] = libdivide::divider<uint32_t>(info.bin_size[j]);
        }
    }

    PointBin<T, Dim>
    operator()(nu_point_collection<Dim, const T> const &input, std::size_t i) const {
        PointBin<T, Dim> p;
        p.bin = 0;

        for (std::size_t j = 0; j < Dim; ++j) {
            auto coord = fold_rescale(input.coordinates[j][i], extents_f[j]);

            // Pack basic data
            p.coordinates[j] = coord;
            p.strength[0] = input.strengths[2 * i];
            p.strength[1] = input.strengths[2 * i + 1];

            // Compute bin index
            auto coord_grid_left = std::ceil(coord - info.offset[j]);
            std::uint32_t coord_grid = std::uint32_t(coord_grid_left - info.global_offset[j]);
            std::size_t bin_index_j = coord_grid / dividers[j];
            p.bin += bin_index_j * info.bin_index_stride[j];
        }

        return p;
    }
};

/** Computes bin index, and collects folded coordinates into
 * given buffer.
 *
 * @param[out] points_with_bin The buffer to write the folded coordinates to, of length
 * `num_points`
 * @param coordinates The points to fold and bin
 * @param fold_rescale Fold-rescale functor to use
 * @param info Bin information
 *
 */
template <typename T, std::size_t Dim, typename FoldRescale>
void compute_bins_and_pack_impl(
    PointBin<T, Dim> *points_with_bin, nu_point_collection<Dim, const T> input,
    FoldRescale &&fold_rescale, IntBinInfo<T, Dim> const &info) {

    ComputeAndPackSingle<T, Dim, FoldRescale> compute_and_pack(info, fold_rescale);

#pragma omp parallel for
    for (std::size_t i = 0; i < input.num_points; ++i) {
        points_with_bin[i] = compute_and_pack(input, i);
    }
}
} // namespace

template <typename T, std::size_t Dim>
void compute_bins_and_pack(
    nu_point_collection<Dim, const T> input, FoldRescaleRange range, IntBinInfo<T, Dim> const &info,
    PointBin<T, Dim> *output) {
    if (range == FoldRescaleRange::Pi) {
        compute_bins_and_pack_impl(output, input, FoldRescalePi<T>{}, info);
    } else {
        compute_bins_and_pack_impl(output, input, FoldRescaleIdentity<T>{}, info);
    }
}

#define INSTANTIATE(T, Dim)                                                                        \
    template void compute_bins_and_pack<T, Dim>(                                                   \
        nu_point_collection<Dim, const T> input,                                                   \
        FoldRescaleRange range,                                                                    \
        IntBinInfo<T, Dim> const &info,                                                            \
        PointBin<T, Dim> *output);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

#undef INSTANTIATE

template <typename T, std::size_t Dim>
void unpack_bins_to_points(
    PointBin<T, Dim> const *input, nu_point_collection<Dim, T> const &output, uint32_t *bin_index) {

#pragma omp parallel for
    for (std::size_t i = 0; i < output.num_points; ++i) {
        auto const &p = input[i];

        for (std::size_t j = 0; j < Dim; ++j) {
            output.coordinates[j][i] = p.coordinates[j];
        }

        output.strengths[2 * i] = p.strength[0];
        output.strengths[2 * i + 1] = p.strength[1];
        bin_index[i] = p.bin;
    }
}

#define INSTANTIATE(T, Dim)                                                                        \
    template void unpack_bins_to_points<T, Dim>(                                                   \
        PointBin<T, Dim> const *input,                                                             \
        nu_point_collection<Dim, T> const &output,                                                 \
        uint32_t *bin_index);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

#undef INSTANTIATE

} // namespace reference
} // namespace spreading
} // namespace finufft
