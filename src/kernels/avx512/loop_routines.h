#pragma once

/** @file This file contains utility routines for working
 * with loops which require aligned memory.
 *
 */

#include <cassert>
#include <type_traits>

#include "../../memory.h"

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(likely)
#define FINUFFT_LIKELY [[likely]]
#endif

#if __has_cpp_attribute(unlikely)
#define FINUFFT_UNLIKELY [[unlikely]]
#endif

#endif

#ifndef FINUFFT_LIKELY
#define FINUFFT_LIKELY
#endif

#ifndef FINUFFT_UNLIKELY
#define FINUFFT_UNLIKELY
#endif

namespace finufft {
namespace spreading {
namespace avx512 {

/** Splits the given loop into a peel loop, a main loop and a remainder loop.
 *
 * @param loop_count The total number of iterations to perform in the loop.
 * @param loop_initial_missing The number of iterations which would bring the loop into alignment.
 * @param loop_alignment The alignment of the loop. This must be a power of 2, and the loop
 *      iteration index will be adjusted by this factor at each step. Additionally, we ensure
 *      that all calls to `fn` will have the loop parameter be a multiple of `loop_alignment`.
 * @param body The body of the loop, this must be a callable object with the signature
 *      `void(std::size_t, std::integral_constant<bool, Partial>, std::size_t)`, where
 *      the first parameter indicates the loop index to process, the second index indicates
 *      whether this is a partial iteration, and the third parameter indicates, when this
 *      is a partial iteration, a mask of the elements to process.
 *
 */
template <typename Fn>
void split_loop(
    std::size_t loop_count, std::size_t loop_initial_missing, std::size_t loop_alignment,
    Fn &&body) {
    std::size_t i = 0;

    assert(loop_initial_missing < loop_alignment);

    // Initial peel loop, brings the loop index into alignment
    if (loop_initial_missing > 0)
        FINUFFT_UNLIKELY {
            body(i, std::true_type{}, ((1 << loop_alignment) - 1) - ((1 << loop_initial_missing) - 1));
            loop_count += loop_initial_missing;
            i += loop_alignment;
        }

    // Main loop
    for (; i + loop_alignment - 1 < loop_count; i += loop_alignment) {
        body(i, std::false_type{}, 0);
    }

    // Remainder loop
    if (i < loop_count) {
        body(i, std::true_type{}, (1 << (loop_count - i)) - 1);
    }
}

namespace detail {
template <typename... Ts> struct AlignMultiplePointersPreviousImpl;

template <typename U, typename... Ts> struct AlignMultiplePointersPreviousImpl<U, Ts...> {
    std::size_t operator()(std::size_t alignment, U *&ptr, Ts *&...ptrs) const noexcept {
        auto new_ptr = align_pointer_previous(ptr, alignment);
        auto alignment_offset = (ptr - new_ptr);
        ptr = new_ptr;

        auto other_offset = AlignMultiplePointersPreviousImpl<Ts...>{}(alignment, ptrs...);
        assert(alignment_offset == other_offset);

        return alignment_offset;
    }
};

template <typename U> struct AlignMultiplePointersPreviousImpl<U> {
    std::size_t operator()(std::size_t alignment, U *&ptr) const noexcept {
        auto new_ptr = align_pointer_previous(ptr, alignment);
        auto alignment_offset = (ptr - new_ptr);
        ptr = new_ptr;

        return alignment_offset;
    }
};

} // namespace detail

/** Aligns multiple pointers to the given alignment, and returns
 * the offset in pointer elements by which the pointers were modified.
 * Note that this function requires that all pointers have the same
 * relative alignment.
 *
 * @param alignment The alignment to align the pointers to.
 * @param ptrs The pointers to align.
 *
 */
template <typename... T>
std::size_t align_multiple_pointers_previous(std::size_t alignment, T *&...ptrs) noexcept {
    return detail::AlignMultiplePointersPreviousImpl<T...>{}(alignment, ptrs...);
}

} // namespace avx512
} // namespace spreading
} // namespace finufft
