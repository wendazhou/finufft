#pragma once

/** Common utilities for the reference implementation of a blocked spread. */

#include "../../tracing.h"
#include "../sorting.h"
#include "../spreading.h"

namespace finufft {
namespace spreading {
namespace reference {

namespace detail {
template <std::size_t Dim> struct GenerateCartesianGridImpl {
    template <typename It, typename IdxT, typename Fn>
    void operator()(It &first, tcb::span<const IdxT, Dim> extents, Fn &&fn) const {
        GenerateCartesianGridImpl<Dim - 1> next;
        for (IdxT i = 0; i < extents[Dim - 1]; ++i) {
            next(first, extents.template first<Dim - 1>(), [&](auto &&...args) -> decltype(auto) {
                return std::forward<Fn>(fn)(args..., i);
            });
        }
    }
};

template <> struct GenerateCartesianGridImpl<0> {
    template <typename It, typename IdxT, typename Fn>
    void operator()(It &first, tcb::span<const IdxT, 0> extents, Fn &&fn) const {
        *first = std::forward<Fn>(fn)();
        ++first;
    }
};
} // namespace detail

/** Assigns each element in a cartesian grid a value generated by the given function object `fn`.
 * 
 * The indices are generated in Fortran-contiguous order, with the first index changing
 * the fastest, and written in linear order to the output iterator `first`.
 * 
 * @param first The iterator to which to write the output.
 * @param extents The extents of the cartesian grid.
 * @param fn The function object to use to generate the values.
 * 
 * @tparam It The type of the output iterator.
 * @tparam IdxT The type of the indices.
 * @tparam fn The type of the generator object, must be callable as `fn(IdxT...)`.
 * 
 */
template <typename IdxT, std::size_t Dim, typename Fn, typename It>
void generate_cartesian_grid(It first, tcb::span<const IdxT, Dim> extents, Fn &&fn) {
    detail::GenerateCartesianGridImpl<Dim> impl;
    impl(first, extents, std::forward<Fn>(fn));
}

} // namespace reference
} // namespace spreading
} // namespace finufft
