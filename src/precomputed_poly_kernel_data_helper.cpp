#include "precomputed_poly_kernel_data.h"

#include <algorithm>

namespace finufft {
namespace detail {

precomputed_poly_kernel_data const *
search_precomputed_poly_kernel(finufft::spreading::kernel_specification const &kernel) noexcept {
    auto it = std::find_if(
        finufft::detail::precomputed_poly_kernel_data_table.begin(),
        finufft::detail::precomputed_poly_kernel_data_table.end(),
        [&kernel](finufft::detail::precomputed_poly_kernel_data const &entry) {
            return (entry.width == kernel.width) &&
                   (entry.beta_1000 == static_cast<std::size_t>(kernel.es_beta * 1000));
        });

    // Could not find matching kernel, throw an exception.
    if (it == finufft::detail::precomputed_poly_kernel_data_table.end()) {
        return nullptr;
    }
    return &*it;
}

} // namespace detail
} // namespace finufft