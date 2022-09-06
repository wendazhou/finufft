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
    IntGridBinInfo<T, Dim> const &bin_info,
    SpreadSubproblemFunctor<T, Dim> const &spread_subproblem) {
    SupergridSpecification<Dim> spec;

    for (std::size_t i = 0; i < Dim; ++i) {
        auto bin_left = (bin_info.num_bins[i] - 1) * bin_info.bin_size[i];
        auto bin_right = bin_left + bin_info.grid_size[i];
        auto extent = round_to_next_multiple(bin_right, spread_subproblem.extent_multiple()[i]);

        spec.size[i] = extent;
        spec.offset[i] = bin_info.global_offset[i];
    }

    return spec;
}

template <typename T, std::size_t Dim> struct SingleThreadedSpreadBlockedImplementation {
    SpreadSubproblemFunctor<T, Dim> spread_subproblem_;
    IntGridBinInfo<T, Dim> bin_info_;
    SupergridSpecification<Dim> output_spec_;

    SingleThreadedSpreadBlockedImplementation(
        SpreadSubproblemFunctor<T, Dim> spread_subproblem, IntGridBinInfo<T, Dim> bin_info)
        : spread_subproblem_(std::move(spread_subproblem)), bin_info_(bin_info),
          output_spec_(compute_supergrid_specification(bin_info_, spread_subproblem_)) {}

    void operator()(
        nu_point_collection<Dim, const T> const &input, std::size_t const *bin_boundaries,
        T *output) const {

        std::size_t max_num_points = 0;
        // Determine amount of memory to allocate
        for (std::size_t i = 0; i < bin_info_.num_bins_total(); ++i) {
            max_num_points = std::max(max_num_points, bin_boundaries[i + 1] - bin_boundaries[i]);
        }

        // Make grid specification
        subgrid_specification<Dim> grid;
        std::copy(output_spec_.size.begin(), output_spec_.size.end(), grid.extents.begin());
        std::copy(
            bin_info_.global_offset.begin(), bin_info_.global_offset.end(), grid.offsets.begin());
        {
            std::size_t stride = 1;
            for (std::size_t i = 0; i < Dim; ++i) {
                grid.strides[i] = stride;
                stride *= output_spec_.size[i];
            }
        }

        auto const& padding_info = spread_subproblem_.target_padding();

        // local input
        auto local_points = finufft::spreading::SpreaderMemoryInput<Dim, T>(max_num_points);

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
                bin_info_.bin_index_stride.begin(),
                bin_info_.bin_index_stride.end(),
                idx.begin(),
                std::size_t(0));

            auto block_num_points = bin_boundaries[bin_idx + 1] - bin_boundaries[bin_idx];

            if (block_num_points == 0) {
                // Skip empty bins
                return;
            }

            {
                // Copy points to local memory
                local_points.num_points = block_num_points;

                // Copy points to local buffer
                std::memcpy(
                    local_points.strengths,
                    input.strengths + bin_boundaries[bin_idx],
                    block_num_points * sizeof(T) * 2);

                for (std::size_t dim = 0; dim < Dim; ++dim) {
                    std::memcpy(
                        local_points.coordinates[dim],
                        input.coordinates[dim] + bin_boundaries[bin_idx],
                        block_num_points * sizeof(T));
                }

                auto num_points_padded = finufft::round_to_next_multiple(
                    block_num_points, spread_subproblem_.num_points_multiple());

                // Pad the input points to the required multiple, using a pad coordinate derived
                // from the subgrid. The pad coordinate is given by the leftmost valid
                // coordinate in the subgrid.
                std::array<T, Dim> pad_coordinate;
                for (std::size_t i = 0; i < Dim; ++i) {
                    pad_coordinate[i] =
                        padding_info[i].min_valid_value(grid.offsets[i], grid.extents[i]);
                }
                finufft::spreading::pad_nu_point_collection(
                    local_points, num_points_padded, pad_coordinate);
            }

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
