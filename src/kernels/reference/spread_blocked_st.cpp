#include "spread_blocked_st.h"

#include <array>
#include <vector>

#include "spread_blocked_impl.h"

namespace finufft {
namespace spreading {
namespace reference {

template <std::size_t Dim> struct SupergridSpecification {
    std::array<std::size_t, Dim> size;
    std::array<std::int64_t, Dim> offset;
};

namespace {

template <typename T, std::size_t Dim, typename It>
void fill_bin_grids(
    It out, IntGridBinInfo<T, Dim> const &info, tcb::span<const std::size_t, Dim> extent_multiple,
    tcb::span<const std::size_t, Dim> supergrid_size) {

    std::array<std::size_t, Dim> grid_strides;
    {
        std::size_t stride = 1;
        for (std::size_t i = 0; i < Dim; ++i) {
            grid_strides[i] = stride;
            stride *= supergrid_size[i];
        }
    }

    std::array<std::size_t, Dim> grid_extents;
    for (std::size_t i = 0; i < Dim; ++i) {
        grid_extents[i] = round_to_next_multiple(info.grid_size[i], extent_multiple[i]);
    }

    auto grid_fn = [&](auto... idxs) -> subgrid_specification<Dim> {
        std::array<std::size_t, Dim> idx{idxs...};
        subgrid_specification<Dim> grid;

        for (std::size_t i = 0; i < Dim; ++i) {
            grid.offsets[i] = info.global_offset[i] + idx[i] * info.bin_size[i];
        }

        std::copy(grid_extents.begin(), grid_extents.end(), grid.extents.begin());
        std::copy(grid_strides.begin(), grid_strides.end(), grid.strides.begin());

        return grid;
    };

    generate_cartesian_grid(out, grid_extents, std::move(grid_fn));
}

template<typename T, std::size_t Dim>
SupergridSpecification<Dim> compute_supergrid_specification(
    IntGridBinInfo<T, Dim> const &bin_info, SpreadBlockedFunctor<T, Dim> spread_subproblem) {
    SupergridSpecification<Dim> spec;

    for (std::size_t i = 0; i < Dim; ++i) {
        auto bin_left = (bin_info.num_bins[i] - 1) * bin_info.bin_size[i];
        auto bin_right = bin_left = bin_info.grid_size[i];
        auto extent = round_to_next_multiple(bin_right, spread_subproblem.extent_multiple[i]);

        spec.size[i] = extent;
    }
}

} // namespace

template <typename T, std::size_t Dim> struct SingleThreadedSpreadBlockedImplementation {
    SpreadSubproblemFunctor<T, Dim> spread_subproblem_;
    IntGridBinInfo<T, Dim> bin_info_;
    std::vector<subgrid_specification<Dim>> bin_grids_;
    SupergridSpecification<Dim> output_spec_;

    SingleThreadedSpreadBlockedImplementation(
        SpreadSubproblemFunctor<T, Dim> spread_subproblem, IntGridBinInfo<T, Dim> bin_info)
        : spread_subproblem_(std::move(spread_subproblem)), bin_info_(bin_info),
          bin_grids_(bin_info_.num_bins_total()) {
        fill_bin_grids(
            bin_grids_.begin(), bin_info_, spread_subproblem_.extent_multiple(), output_spec_.size);
    }

    void operator()(
        nu_point_collection<Dim, const T> const &input, IntGridBinInfo<T, Dim> const &info,
        std::size_t const *bin_boundaries, T *output) const {}

    SupergridSpecification<Dim>
    get_output_specification(tcb::span<const std::size_t, Dim> extents) const noexcept {
        return output_spec_;
    }
};

} // namespace reference
} // namespace spreading
} // namespace finufft
