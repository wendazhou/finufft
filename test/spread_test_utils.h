#pragma once

#include "../src/spreading.h"

#include <random>
#include <utility>
#include <vector>

namespace {

template <std::size_t Dim, typename T>
finufft::spreading::nu_point_collection<Dim, finufft::spreading::aligned_unique_array<T>>
make_random_point_collection(std::size_t num_points, uint32_t seed, std::pair<T, T> range) {

    auto points =
        finufft::spreading::nu_point_collection<Dim, finufft::spreading::aligned_unique_array<T>>{
            num_points,
            finufft::spreading::allocate_aligned_arrays<Dim, T>(num_points, 64),
            finufft::spreading::allocate_aligned_array<T>(2 * num_points, 64)};

    std::minstd_rand rng(seed);
    std::uniform_real_distribution<T> uniform_dist(range.first, range.second);

    for (std::size_t dim = 0; dim < Dim; ++dim) {
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

std::vector<int64_t> make_random_permutation(std::size_t n, int32_t seed) {
    std::vector<int64_t> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);

    auto rng = std::minstd_rand(seed);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    return permutation;
}

} // namespace
