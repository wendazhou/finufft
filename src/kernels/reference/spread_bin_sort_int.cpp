#include "spread_bin_sort_int.h"

#include <ips4o/ips4o.hpp>
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

template <typename T, std::size_t Dim>
void unpack_sorted_bins_to_points(
    PointBin<T, Dim> const *input, nu_point_collection<Dim, T> const &output,
    std::size_t *bin_count) {
#pragma omp parallel
    {
        std::size_t i = omp_get_thread_num() * output.num_points / omp_get_num_threads();
        std::size_t final = (omp_get_thread_num() + 1) * output.num_points / omp_get_num_threads();

        // Adjust initial to be the first point in the thread's bin
        if (i > 0) {
            while ((input[i].bin == input[i - 1].bin) && i < final) {
                i++;
            }
        }

        while (i < final) {
            auto current_bin = input[i].bin;
            std::size_t current_bin_count = 0;

            while ((i < output.num_points) && (input[i].bin == current_bin)) {
                auto const &p = input[i];

                for (std::size_t j = 0; j < Dim; ++j) {
                    output.coordinates[j][i] = p.coordinates[j];
                }

                output.strengths[2 * i] = p.strength[0];
                output.strengths[2 * i + 1] = p.strength[1];

                ++current_bin_count;
                ++i;
            }

            bin_count[current_bin] = current_bin_count;
        }
    }
}

namespace {

/** Sorting functor using ips4o to sort packed bins.
 *
 */
template <typename T, std::size_t Dim> struct Ips4oSortFunctor {
    ComputeAndPackBinsFunctor<T, Dim> pack_;
    UnpackBinsFunctor<T, Dim> unpack_;
    SortPackedTimers timers_;

    void operator()(
        nu_point_collection<Dim, const T> const &points, FoldRescaleRange range,
        nu_point_collection<Dim, T> const &output, std::size_t *bin_counts,
        IntBinInfo<T, Dim> const &info) const {
        SortPackedTimers timers(timers_);
        auto packed = finufft::allocate_aligned_array<PointBin<T, Dim>>(points.num_points, 64);

        // Compute bins
        {
            finufft::ScopedTimerGuard guard(timers.pack);
            pack_(points, range, info, packed.get());
        }

        {
            finufft::ScopedTimerGuard guard(timers.sort);
            ips4o::parallel::sort(packed.get(), packed.get() + points.num_points);
        }

        // Unpack to output.
        {
            finufft::ScopedTimerGuard guard(timers.unpack);
            unpack_(packed.get(), output, bin_counts);
        }
    }
};
} // namespace

template <typename T, std::size_t Dim>
SortPointsFunctor<T, Dim> make_ips4o_sort_functor(
    ComputeAndPackBinsFunctor<T, Dim> &&pack, UnpackBinsFunctor<T, Dim> &&unpack,
    SortPackedTimers const &timers) {
    return Ips4oSortFunctor<T, Dim>{std::move(pack), std::move(unpack), timers};
}

template <typename T, std::size_t Dim>
SortPointsFunctor<T, Dim> get_sort_functor(SortPackedTimers const &timers) {
    return make_ips4o_sort_functor<T, Dim>(
        &compute_bins_and_pack<T, Dim>, &unpack_sorted_bins_to_points<T, Dim>, timers);
}

#define INSTANTIATE(T, Dim)                                                                        \
    template void unpack_bins_to_points<T, Dim>(                                                   \
        PointBin<T, Dim> const *input,                                                             \
        nu_point_collection<Dim, T> const &output,                                                 \
        uint32_t *bin_index);                                                                      \
    template void unpack_sorted_bins_to_points(                                                    \
        PointBin<T, Dim> const *input,                                                             \
        nu_point_collection<Dim, T> const &output,                                                 \
        std::size_t *bin_count);                                                                   \
    template SortPointsFunctor<T, Dim> make_ips4o_sort_functor(                                    \
        ComputeAndPackBinsFunctor<T, Dim> &&pack,                                                  \
        UnpackBinsFunctor<T, Dim> &&unpack,                                                        \
        SortPackedTimers const &timers);                                                           \
    template SortPointsFunctor<T, Dim> get_sort_functor(SortPackedTimers const &timers);

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
