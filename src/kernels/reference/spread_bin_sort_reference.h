#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace finufft {
namespace spreading {

/** Computes the bin each point belongs to.
 *
 */
template <typename T, std::size_t Dim>
void compute_bin_index(
    int64_t *index, std::size_t num_points, std::array<T const *, Dim> const &coordinates,
    std::array<T, Dim> const& extents, std::array<T, Dim> const &bin_sizes) {

    std::array<std::size_t, Dim> num_bins;
    std::array<T, Dim> bin_scaling;

    for (std::size_t i = 0; i < Dim; ++i) {
        num_bins[i] = static_cast<std::size_t>(extents[i] / bin_sizes[i]) + 1;
        bin_scaling[i] = static_cast<T>(1. / bin_sizes[i]);
    }

    std::memset(index, 0, sizeof(int64_t) * num_points);

    std::size_t stride = 1;

    for (std::size_t j = 0; j < Dim; ++j) {
        for (std::size_t i = 0; i < num_points; ++i) {
            std::size_t bin = static_cast<std::size_t>(coordinates[j][i] * bin_scaling[j]);
            index[i] += stride * bin;
        }

        stride *= num_bins[j];
    }
}
} // namespace spreading
} // namespace finufft
