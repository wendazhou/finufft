#pragma once

/** @file
 *
 * Utilities for working with memory in finufft.
 * This file encapsulates functionality to work with memory,
 * and in particular allocate aligned memory and huge pages.
 *
 */

#include <cstdlib>
#include <memory>

namespace finufft {

#ifdef __cpp_aligned_new

/// @{

/** Allocator for aligned array memory using C++ standard new.
 *
 * From C++17 (feature test macro: __cpp_aligned_new), the standard
 * allocator supports specifying memory alignment in the new operation.
 * This allocator makes use of this feature to allocate aligned memory.
 *
 */
template <typename T> struct NewAlignedArrayAllocator {
    struct Deleter {
        std::size_t alignment;
        void operator()(T *ptr) const noexcept {
            ::operator delete[](ptr, std::align_val_t(this->alignment));
        }
    };

    std::unique_ptr<T[], Deleter>
    operator()(std::size_t num_elements, std::size_t alignment) const {
        std::size_t size_bytes = num_elements * sizeof(T);
        size_bytes = (size_bytes + alignment - 1) / alignment * alignment;
        num_elements = size_bytes / sizeof(T);

        auto deleter = Deleter{alignment};
        return std::unique_ptr<T[], Deleter>(
            new (std::align_val_t(alignment)) T[num_elements], deleter);
    }
};

/// @}

#endif

/** Simple deleter using cstdlib's free operation.
 *
 */
struct FreeDeleter {
    void operator()(void *ptr) const noexcept { std::free(ptr); }
};

namespace detail {
void *allocate_aligned_memory_hugepage_posix(std::size_t size_bytes, std::size_t alignment);
}

/** Allocator for aligned array memory using posix_memalign.
 *
 * This allocator uses posix_memalign to allocate aligned memory on linux systems.
 * Additionally, it attempts to allocate a huge page for allocations larger than
 * the standard page size (4 kiB).
 *
 */
template <typename T> struct PosixMemalignArrayAllocator {
    typedef FreeDeleter Deleter;

    std::unique_ptr<T[], Deleter>
    operator()(std::size_t num_elements, std::size_t alignment) const {
        std::size_t size_bytes = num_elements * sizeof(T);
        size_bytes = (size_bytes + alignment - 1) / alignment * alignment;
        auto ptr = detail::allocate_aligned_memory_hugepage_posix(size_bytes, alignment);
        return std::unique_ptr<T[], Deleter>(reinterpret_cast<T *>(ptr), Deleter{});
    }
};

#if __linux__
template <typename T> using DefaultAlignedAllocator = PosixMemalignArrayAllocator<T>;
#else
template <typename T> using DefaultAlignedAllocator = NewAlignedArrayAllocator<T>;
#endif

template <typename T>
using aligned_unique_array = std::unique_ptr<T[], typename DefaultAlignedAllocator<T>::Deleter>;

/** Allocates an array of the given size with specified alignment (in bytes).
 *
 * @param size Number of elements in the array.
 * @param alignment Alignment of the array in bytes. Must be a power of 2.
 */
template <typename T>
aligned_unique_array<T> allocate_aligned_array(std::size_t size, std::size_t alignment) {
    return DefaultAlignedAllocator<T>{}(size, alignment);
}

template <std::size_t Dim, typename T>
std::array<aligned_unique_array<T>, Dim>
allocate_aligned_arrays(std::size_t size, std::size_t alignment) {
    std::array<aligned_unique_array<T>, Dim> arrays;
    for (int i = 0; i < Dim; ++i) {
        arrays[i] = allocate_aligned_array<T>(size, alignment);
    }

    return std::move(arrays);
}

// Utility function which rounds the given integer value to the next multiple.
template <typename T, typename U> T round_to_next_multiple(T v, U multiple) {
    return (v + multiple - 1) / multiple * multiple;
}

/** Utility function which aligns the pointer to the previous pointer (i.e. the largest pointer
 *  which compares less than `ptr`) which is aligned to the given alignment.
 *
 * @param ptr The pointer to align
 * @param alignment The alignment in bytes
 *
 */
template <typename T> T *align_pointer_previous(T *ptr, std::size_t alignment) {
    return reinterpret_cast<T *>(reinterpret_cast<std::uintptr_t>(ptr) / alignment * alignment);
}

} // namespace finufft
