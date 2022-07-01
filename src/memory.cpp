#include "memory.h"

#include <cassert>
#include <stdexcept>
#include <stdlib.h>

#if __linux__

#include <sys/mman.h>

namespace finufft {
namespace detail {

/** This function allocates aligned memory using posix_memalign.
 *
 * In addition, for allocations larger than 4 kiB, it attempts to allocate a
 * huge page by aligning the memory to 2 MiB and advising for huge page.
 *
 */
void *allocate_aligned_memory_hugepage_posix(std::size_t size_bytes, std::size_t alignment) {
    if (size_bytes > 4 * 1024) {
        std::size_t TWO_MiB = 2 * 1024 * 1024;

        // Allocating data larger than a single page.
        // Attempt to allocate a huge page.
        // Align to at least 2MB boundary to attempt to get a huge page.
        alignment = std::max(alignment, TWO_MiB);
    }

    void *ptr = nullptr;
    int err = posix_memalign(&ptr, alignment, size_bytes);
    if (err == ENOMEM) {
        throw std::bad_alloc();
    }

    // Other possible failure is EINVAL which is a programming
    // error if the alignment is not a power of 2.
    assert(err == 0);

    if (size_bytes > 4 * 1024) {
        // If allocating data larger than a single standard page,
        // try to prompt kernel to allocate a huge page for us.
        madvise(ptr, size_bytes, MADV_HUGEPAGE);
    }

    return ptr;
}
} // namespace detail
} // namespace finufft

#endif
