#include "spread_blocked_st.h"

#include <array>
#include <cstring>
#include <numeric>
#include <vector>

#include "spread_blocked_impl.h"

namespace finufft {
namespace spreading {
namespace reference {

template <std::size_t Dim> struct SupergridSpecification {
    std::array<std::size_t, Dim> size;
    std::array<std::int64_t, Dim> offset;
};

template <typename T, std::size_t Dim>
SupergridSpecification<Dim> compute_supergrid_specification(
    SpreadSubproblemFunctor<T, Dim> const &spread_subproblem,
    tcb::span<const std::size_t, Dim> target_size) {
    SupergridSpecification<Dim> spec;

    auto const &padding = spread_subproblem.target_padding();
    auto const &extent_multiple = spread_subproblem.extent_multiple();

    for (std::size_t i = 0; i < Dim; ++i) {
        auto offset = static_cast<std::int64_t>(-std::ceil(-padding[i].offset));
        auto extent_main = padding[i].grid_right + target_size[i];
        auto extent_total = extent_main + offset + padding[i].grid_left;

        spec.size[i] = round_to_next_multiple(extent_total, extent_multiple[i]);
        spec.offset[i] = offset;
    }

    return spec;
}

template <typename T, std::size_t Dim> struct SingleThreadedSpreadBlockedImplementation {
    SpreadSubproblemFunctor<T, Dim> spread_subproblem_;
    SupergridSpecification<Dim> output_spec_;
    LocalPointsBufferFactory<T, Dim> points_buffer_factory_;

    SingleThreadedSpreadBlockedImplementation(
        SpreadSubproblemFunctor<T, Dim> spread_subproblem,
        tcb::span<const std::size_t, Dim> target_size)
        : spread_subproblem_(std::move(spread_subproblem)),
          output_spec_(compute_supergrid_specification(spread_subproblem_, target_size)) {}

    void operator()(
        nu_point_collection<Dim, const T> const &input, IntGridBinInfo<T, Dim> const &bin_info,
        std::size_t const *bin_boundaries, T *output) const {

        std::size_t max_num_points = 0;
        // Determine amount of memory to allocate
        for (std::size_t i = 0; i < bin_info.num_bins_total(); ++i) {
            max_num_points = std::max(max_num_points, bin_boundaries[i + 1] - bin_boundaries[i]);
        }

        // Make grid specification
        subgrid_specification<Dim> grid;
        std::copy(output_spec_.size.begin(), output_spec_.size.end(), grid.extents.begin());
        std::copy(
            bin_info.global_offset.begin(), bin_info.global_offset.end(), grid.offsets.begin());
        {
            std::size_t stride = 1;
            for (std::size_t i = 0; i < Dim; ++i) {
                grid.strides[i] = stride;
                stride *= output_spec_.size[i];
            }
        }

        auto const &padding_info = spread_subproblem_.target_padding();

        // local input
        auto local_points_buffer = points_buffer_factory_(max_num_points);

        // Zero memory
        std::size_t total_size = std::accumulate(
            output_spec_.size.begin(),
            output_spec_.size.end(),
            std::size_t(1),
            std::multiplies<>());
        std::memset(output, 0, sizeof(T) * total_size);

        // Loop through blocks
        auto accumulate_block = [&](auto... idx_v) {
            std::array<std::size_t, Dim> idx{idx_v...};

            std::size_t bin_idx = std::inner_product(
                bin_info.bin_index_stride.begin(),
                bin_info.bin_index_stride.end(),
                idx.begin(),
                std::size_t(0));

            auto block_num_points = bin_boundaries[bin_idx + 1] - bin_boundaries[bin_idx];

            if (block_num_points == 0) {
                // Skip empty bins
                return;
            }

            auto local_points =
                local_points_buffer(input, bin_boundaries[bin_idx], block_num_points, grid);

            // Spread on subgrid
            spread_subproblem_(local_points, grid, output);
        };
    }

    SupergridSpecification<Dim>
    get_output_specification(tcb::span<const std::size_t, Dim> extents) const noexcept {
        return output_spec_;
    }
};

template struct SingleThreadedSpreadBlockedImplementation<float, 1>;

} // namespace reference
} // namespace spreading
} // namespace finufft
