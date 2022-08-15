#pragma once

/** @file
 *
 * Header-based parametrized implementations for bin-sorting through counting sorts.
 * These implementations are provided in order to facilitate the implementation of
 * optimized kernels which may only replace specific inner-loop parts.
 *
 */

#include <tuple>
#include <type_traits>

#include "sort_bin_counting.h"

namespace finufft {
namespace spreading {
namespace reference {
namespace detail {

template <std::size_t S, typename IndexType, typename T> struct BinIndexValue {
    alignas(64) IndexType bin_index[S];
    [[no_unique_address]] T value;
};

/** Generic blocked loop over non-uniform points with bin index computation.
 *
 * This function implements a generic partially unrolled loop over non-uniform points,
 * computing the bin index for each point and process it in some fashion.
 * Additionally, it may implement functionality to interleave the computation of the bin
 * index and its processing in order to avoid dependencies that are too tight.
 *
 * This function is abstracted into three parts of the function:
 * - process_bin_index
 *      Computes the bin index for the given set of points, and additionally
 *      stores additional information based on the transformed (folded) coordinates.
 *      This argument should be a callable object which takes the following arguments:
 *      - nu_point_collection: the input collection of non-uniform points
 *      - i: the index of the point in the set to process
 *      - limit: the number of points to process starting from i. May be ignored if
 *          the partial argument is set to false.
 *      - bin_index: a reference to an aligned array of IndexType, which will be filled
 *          with the bin index for the given set of points.
 *      - partial: a boolean flag indicating whether the loop is partial or not.
 *
 */
template <
    typename T, std::size_t Dim, typename BinIndexFunctor, typename ProcessBinFunctor,
    typename WriteTransformedCoordinate>
void process_bin_function(
    nu_point_collection<Dim, const T> const &input, BinIndexFunctor const &compute_bin_index,
    ProcessBinFunctor const &process_bin_index,
    WriteTransformedCoordinate const &write_transformed_coordinate) {

    constexpr std::size_t unroll = BinIndexFunctor::unroll;
    typedef typename BinIndexFunctor::index_type index_type;
    typedef typename WriteTransformedCoordinate::value_type value_type;

    BinIndexValue<unroll, index_type, value_type> bin_index_value;

    auto run_loop = [&](std::size_t i, std::size_t n, auto partial) {
        compute_bin_index(
            input,
            i,
            n,
            bin_index_value.bin_index,
            partial,
            [&](std::size_t d, std::size_t j, auto const &x) {
                write_transformed_coordinate(bin_index_value.value, d, j, x);
            });

        process_bin_index(input, i, n, bin_index_value, partial);
    };

    std::size_t i = 0;
    for (; i < input.num_points - unroll + 1; i += unroll) {
        run_loop(i, unroll, std::integral_constant<bool, false>{});
    }

    {
        // loop tail
        if (i < input.num_points) {
            run_loop(i, input.num_points - i, std::integral_constant<bool, true>{});
        }
    }
}

struct NoOpWriteTransformedCoordinate {
    // Do not store any value in the transformed coordinate.
    typedef std::tuple<> value_type;

    template <typename T>
    void operator()(value_type &v, std::size_t d, std::size_t j, T &&t) const {}
};

/** Computes bin histogram using given bin index computation.
 *
 * This provides a reference implementation based on a partially unrolled
 * bin index computation (in order to support vectorized bin index computations).
 *
 * @param input Input array of non-uniform points
 * @param histogram Output array of bin histograms
 * @param compute_bin_index Function to compute bin index for each point
 *
 */
template <typename T, std::size_t Dim, typename BinIndexFunctor>
void compute_histogram_impl(
    nu_point_collection<Dim, const T> const &input, tcb::span<std::size_t> histogram,
    BinIndexFunctor const &compute_bin_index) {

    auto scatter_histogram = [&](nu_point_collection<Dim, const T> const &input,
                                 std::size_t i,
                                 std::size_t limit,
                                 auto const &bin_index_value,
                                 auto partial) {
        for (std::size_t j = 0; j < limit; ++j) {
            ++histogram[bin_index_value.bin_index[j]];
        }
    };

    process_bin_function(
        input, compute_bin_index, scatter_histogram, NoOpWriteTransformedCoordinate{});
}

template <typename T, std::size_t Dim, std::size_t Unroll> struct WriteTransformedCoordinate {
    typedef std::array<std::array<T, Unroll>, Dim> value_type;

    void operator()(value_type &v, std::size_t d, std::size_t j, T const &t) const { v[d][j] = t; }
};

/** Reorders points into sorted order by using the given set of partial histogram sums.
 *
 */
template <typename T, std::size_t Dim, typename BinIndexFunctor>
void move_points_by_histogram_impl(
    tcb::span<std::size_t> histogram, nu_point_collection<Dim, const T> const &input,
    nu_point_collection<Dim, T> const &output, BinIndexFunctor const &compute_bin_index) {
    auto move_points = [&](nu_point_collection<Dim, const T> const &input,
                           std::size_t i,
                           std::size_t limit,
                           auto const &bin_index_value,
                           auto partial) {
        for (std::size_t j = 0; j < limit; ++j) {
            auto b = bin_index_value.bin_index[j];
            auto output_index = --histogram[b];

            for (std::size_t d = 0; d < Dim; ++d) {
                output.coordinates[d][output_index] = bin_index_value.value[d][j];
            }

            output.strengths[2 * output_index] = input.strengths[2 * (i + j)];
            output.strengths[2 * output_index + 1] = input.strengths[2 * (i + j) + 1];
        }
    };

    process_bin_function<T, Dim>(
        input,
        compute_bin_index,
        move_points,
        WriteTransformedCoordinate<T, Dim, BinIndexFunctor::unroll>{});
}

} // namespace detail
} // namespace reference
} // namespace spreading
} // namespace finufft
