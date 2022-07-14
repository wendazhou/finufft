#include <limits>
#include <stdexcept>

/** @file
 *
 * Implementation of gather-fold in highway for all vectorization types.
 * TODO: implement the function for double precision inputs.
 *
 */

#include "../spreading.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "gather_fold_hwy_impl.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "fold_rescale.inl.h"

HWY_BEFORE_NAMESPACE();
namespace finufft {
namespace spreading {
namespace highway {
namespace HWY_NAMESPACE {

namespace hn = ::hwy::HWY_NAMESPACE;
template <typename T, std::size_t Dim> struct GatherFoldImplLoop;

template <std::size_t Dim, typename RescaleFn>
void gather_and_fold_impl_hwy(
    nu_point_collection<Dim, float> const &memory,
    nu_point_collection<Dim, float const> const &input, std::array<float, Dim> const &extent,
    int64_t const *sort_indices, RescaleFn &&fold_rescale) {

    if (input.num_points > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("input.num_points > std::numeric_limits<int32_t>::max()");
    }

    hn::ScalableTag<float> d;
    hn::ScalableTag<int32_t> di;
    hn::ScalableTag<int64_t> di64;

    const auto N = hn::Lanes(d);

    std::size_t i = 0;

    for (; i + N < memory.num_points; i += N) {
        auto idx_l = hn::LoadU(di64, sort_indices + i);
        auto idx_h = hn::LoadU(di64, sort_indices + i + N / 2);

        // Bitcast 64bit -> 32bit OK for positive signed integers.
        // Then concat lower halves (i.e. even lanes).
        auto idx = hn::ConcatEven(di, hn::BitCast(di, idx_h), hn::BitCast(di, idx_l));

        for (std::size_t dim = 0; dim < Dim; ++dim) {
            auto v = hn::GatherIndex(d, input.coordinates[dim], idx);
            v = fold_rescale(v, extent[dim], d);
            hn::Store(v, d, memory.coordinates[dim] + i);
        }

        auto strengths1 =
            hn::GatherIndex(di64, reinterpret_cast<int64_t const *>(input.strengths), idx_l);
        auto strengths2 =
            hn::GatherIndex(di64, reinterpret_cast<int64_t const *>(input.strengths), idx_h);
        hn::Store(strengths1, di64, reinterpret_cast<int64_t *>(memory.strengths) + i);
        hn::Store(strengths2, di64, reinterpret_cast<int64_t *>(memory.strengths) + i + N / 2);
    }

    for (; i < memory.num_points; ++i) {
        auto idx = sort_indices[i];

        for (int j = 0; j < Dim; ++j) {
            memory.coordinates[j][i] = fold_rescale(input.coordinates[j][idx], extent[j]);
        }

        memory.strengths[2 * i] = input.strengths[2 * idx];
        memory.strengths[2 * i + 1] = input.strengths[2 * idx + 1];
    }
}

template <typename T, std::size_t Dim>
void gather_and_fold_hwy(
    nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
    std::array<int64_t, Dim> const &sizes, int64_t const *sort_indices,
    FoldRescaleRange rescale_range) {

    std::array<T, Dim> sizes_floating;
    std::copy(sizes.begin(), sizes.end(), sizes_floating.begin());

    if (rescale_range == FoldRescaleRange::Pi) {
        gather_and_fold_impl_hwy(memory, input, sizes_floating, sort_indices, FoldRescalePi<T>{});
    } else {
        gather_and_fold_impl_hwy(
            memory, input, sizes_floating, sort_indices, FoldRescaleIdentity<T>{});
    }
}

#define INSTANTIATE_GATHER_AND_FOLD(type, dim)                                                     \
    void gather_and_fold_hwy_##type##_##dim(                                                       \
        nu_point_collection<dim, type> const &memory,                                              \
        nu_point_collection<dim, type const> const &input,                                         \
        std::array<int64_t, dim> const &sizes,                                                     \
        int64_t const *sort_indices,                                                               \
        FoldRescaleRange rescale_range) {                                                          \
        gather_and_fold_hwy<type, dim>(memory, input, sizes, sort_indices, rescale_range);         \
    }

INSTANTIATE_GATHER_AND_FOLD(float, 1)
INSTANTIATE_GATHER_AND_FOLD(float, 2)
INSTANTIATE_GATHER_AND_FOLD(float, 3)

#undef INSTANTIATE_GATHER_AND_FOLD

} // namespace HWY_NAMESPACE
} // namespace highway
} // namespace spreading
} // namespace finufft
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace finufft {
namespace spreading {
namespace highway {

template <typename T, std::size_t Dim>
void gather_and_fold_hwy(
    nu_point_collection<Dim, T> const &memory, nu_point_collection<Dim, T const> const &input,
    std::array<int64_t, Dim> const &sizes, int64_t const *sort_indices,
    FoldRescaleRange rescale_range);

#define EXPORT_GATHER_AND_FOLD(type, dim)                                                          \
    HWY_EXPORT(gather_and_fold_hwy_##type##_##dim);                                                \
    template <>                                                                                    \
    void gather_and_fold_hwy<type, dim>(                                                           \
        nu_point_collection<dim, type> const &memory,                                              \
        nu_point_collection<dim, type const> const &input,                                         \
        std::array<int64_t, dim> const &sizes,                                                     \
        int64_t const *sort_indices,                                                               \
        FoldRescaleRange rescale_range) {                                                          \
        HWY_DYNAMIC_DISPATCH(gather_and_fold_hwy_##type##_##dim)                                   \
        (memory, input, sizes, sort_indices, rescale_range);                                       \
    }

EXPORT_GATHER_AND_FOLD(float, 1)
EXPORT_GATHER_AND_FOLD(float, 2)
EXPORT_GATHER_AND_FOLD(float, 3)

} // namespace highway
} // namespace spreading
} // namespace finufft

#endif
