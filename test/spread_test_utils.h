#pragma once

#include "../src/kernels/spreading.h"

#include <iterator>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_utils.h"

namespace {

template <std::size_t Dim, typename T>
finufft::spreading::SpreaderMemoryInput<Dim, T> make_random_point_collection(
    std::size_t num_points, uint32_t seed, std::array<std::pair<T, T>, Dim> const &range) {

    auto points = finufft::spreading::SpreaderMemoryInput<Dim, T>(num_points);

    for (std::size_t dim = 0; dim < Dim; ++dim) {
        if (range[dim].first > range[dim].second) {
            throw std::invalid_argument("range[dim].first > range[dim].second");
        }

        finufft::testing::fill_random_uniform(
            points.coordinates[dim],
            num_points,
            dim + seed,
            range[dim].first,
            range[dim].second);
    }

    finufft::testing::fill_random_uniform(
        points.strengths, 2 * num_points, Dim + seed, -1, 1);

    return points;
}

template <std::size_t Dim, typename T>
finufft::spreading::SpreaderMemoryInput<Dim, T>
make_random_point_collection(std::size_t num_points, uint32_t seed, std::pair<T, T> range) {
    std::array<std::pair<T, T>, Dim> range_array;
    std::fill(range_array.begin(), range_array.end(), range);
    return make_random_point_collection<Dim, T>(num_points, seed, range_array);
}

finufft::aligned_unique_array<int64_t> make_random_permutation(std::size_t n, int32_t seed) {
    auto permutation = finufft::allocate_aligned_array<int64_t>(n, 64);
    std::iota(permutation.get(), permutation.get() + n, 0);

    auto rng = std::minstd_rand(seed);
    std::shuffle(permutation.get(), permutation.get() + n, rng);
    return permutation;
}

/** Create data for the spread subproblem.
 *
 * This function creates a collection of random points with coordinates
 * lying in the required range as indicated by the grid and padding parameters.
 *
 * @param num_points The number of points to create
 * @param seed Random seed to use for generating points
 * @param grid The grid describing the location of the points
 * @param padding_left Required additional padding on the left
 * @param padding_right Required additional padding on the right
 *
 */
template <typename T, std::size_t Dim>
finufft::spreading::SpreaderMemoryInput<Dim, T> make_spread_subproblem_input(
    std::size_t num_points, uint32_t seed, finufft::spreading::grid_specification<Dim> const &grid,
    std::array<finufft::spreading::KernelWriteSpec<T>, Dim> const &padding) {
    std::array<std::pair<T, T>, Dim> range;

    for (std::size_t i = 0; i < Dim; ++i) {
        range[i].first = padding[i].min_valid_value(grid.offsets[i], grid.extents[i]);
        range[i].second = padding[i].max_valid_value(grid.offsets[i], grid.extents[i]);
    }

    return make_random_point_collection<Dim, T>(num_points, seed, range);
}

/** Sorts the collection of non-uniform points lexicographically.
 *
 */
template <typename T, std::size_t Dim>
void sort_point_collection(finufft::spreading::nu_point_collection<Dim, T> &points) {
    std::vector<std::size_t> permutation(points.num_points);
    std::iota(permutation.begin(), permutation.end(), 0);

    // Sort in reverse lexicographic order
    std::sort(permutation.begin(), permutation.end(), [&points](std::size_t i, std::size_t j) {
        for (std::size_t dim = Dim - 1; dim >= 0; --dim) {
            if (points.coordinates[dim][i] != points.coordinates[dim][j]) {
                return points.coordinates[dim][i] < points.coordinates[dim][j];
            }
        }
        return false;
    });

    for (std::size_t dim = 0; dim < Dim; ++dim) {
        std::vector<T> sorted_coordinates(points.num_points);
        for (std::size_t i = 0; i < points.num_points; ++i) {
            sorted_coordinates[i] = points.coordinates[dim][permutation[i]];
        }
        for (std::size_t i = 0; i < points.num_points; ++i) {
            points.coordinates[dim][i] = sorted_coordinates[i];
        }
    }

    std::vector<T> sorted_strengths(points.num_points * 2);
    for (std::size_t i = 0; i < points.num_points; ++i) {
        sorted_strengths[2 * i] = points.strengths[2 * permutation[i]];
        sorted_strengths[2 * i + 1] = points.strengths[2 * permutation[i] + 1];
    }
    for (std::size_t i = 0; i < points.num_points * 2; ++i) {
        points.strengths[i] = sorted_strengths[i];
    }
}

/** Utility function to compute specification (in particular beta) from width.
 *
 * Note that the choices here are matched from `setup_spreader`, and should
 * be kept in sync to ensure that pre-generated kernels can be found.
 */
inline finufft::spreading::kernel_specification
specification_from_width(int kernel_width, double upsampling_factor) {
    double beta_over_width;

    switch (kernel_width) {
    case 2:
        beta_over_width = 2.20;
        break;
    case 3:
        beta_over_width = 2.26;
        break;
    case 4:
        beta_over_width = 2.38;
        break;
    default:
        beta_over_width = 2.30;
        break;
    }

    if (upsampling_factor != 2.0) {
        beta_over_width = 0.97 * M_PI * (1.0 - 0.5 / upsampling_factor);
    }

    return finufft::spreading::kernel_specification{beta_over_width * kernel_width, kernel_width};
}

/** Computes threshold based on relative value of maximum absolute value in array.
 *
 */
template <typename U, typename It>
typename std::iterator_traits<It>::value_type
compute_max_relative_threshold(U tolerance, It first, It last) {
    typedef typename std::iterator_traits<It>::value_type T;
    auto r_min_max = std::minmax_element(first, last);
    auto r_max = std::max(std::abs(*r_min_max.first), std::abs(*r_min_max.second));
    return static_cast<T>(tolerance * r_max);
}

} // namespace
