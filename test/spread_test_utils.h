#pragma once

#include "../src/spreading.h"

#include <iterator>
#include <random>
#include <utility>
#include <vector>

namespace {

template <std::size_t Dim, typename T>
finufft::spreading::nu_point_collection<Dim, finufft::spreading::aligned_unique_array<T>>
make_random_point_collection(
    std::size_t num_points, uint32_t seed, std::array<std::pair<T, T>, Dim> const &range) {

    auto points =
        finufft::spreading::nu_point_collection<Dim, finufft::spreading::aligned_unique_array<T>>{
            num_points,
            finufft::spreading::allocate_aligned_arrays<Dim, T>(num_points, 64),
            finufft::spreading::allocate_aligned_array<T>(2 * num_points, 64)};

    std::minstd_rand rng(seed);

    for (std::size_t dim = 0; dim < Dim; ++dim) {
        std::uniform_real_distribution<T> uniform_dist(range[dim].first, range[dim].second);

        for (std::size_t i = 0; i < num_points; ++i) {
            points.coordinates[dim][i] = uniform_dist(rng);
        }
    }

    std::normal_distribution<T> normal_dist;

    for (std::size_t i = 0; i < num_points; ++i) {
        points.strengths[2 * i] = normal_dist(rng);
        points.strengths[2 * i + 1] = normal_dist(rng);
    }

    return points;
}

template <std::size_t Dim, typename T>
finufft::spreading::nu_point_collection<Dim, finufft::spreading::aligned_unique_array<T>>
make_random_point_collection(std::size_t num_points, uint32_t seed, std::pair<T, T> range) {
    std::array<std::pair<T, T>, Dim> range_array;
    std::fill(range_array.begin(), range_array.end(), range);
    return make_random_point_collection<Dim, T>(num_points, seed, range_array);
}

finufft::spreading::aligned_unique_array<int64_t>
make_random_permutation(std::size_t n, int32_t seed) {
    auto permutation = finufft::spreading::allocate_aligned_array<int64_t>(n, 64);
    std::iota(permutation.get(), permutation.get() + n, 0);

    auto rng = std::minstd_rand(seed);
    std::shuffle(permutation.get(), permutation.get() + n, rng);
    return permutation;
}


/** Computes threshold based on relative value of maximum absolute value in array.
 * 
 */
template <typename U, typename It>
typename std::iterator_traits<It>::value_type compute_max_relative_threshold(U tolerance, It first, It last) {
    typedef typename std::iterator_traits<It>::value_type T;
    auto r_min_max = std::minmax_element(first, last);
    auto r_max = std::max(std::abs(*r_min_max.first), std::abs(*r_min_max.second));
    return static_cast<T>(tolerance * r_max);
}


} // namespace
