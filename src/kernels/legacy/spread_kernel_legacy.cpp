#include "spread_kernel.h"
#include "spread_legacy.h"

#include <finufft_spread_opts.h>

namespace finufft {

namespace common {
void onedim_fseries_kernel(int64_t nf, float *fwkerhalf, finufft_spread_opts opts);
void onedim_fseries_kernel(int64_t nf, double *fwkerhalf, finufft_spread_opts opts);
} // namespace common
namespace spreading {
namespace legacy {

void onedim_fseries_kernel(
    std::size_t num_frequencies, float *coeffs, kernel_specification const &kernel_spec) {
    finufft_spread_opts opts = construct_opts_from_kernel(kernel_spec);
    common::onedim_fseries_kernel(num_frequencies, coeffs, opts);
}
void onedim_fseries_kernel(
    std::size_t num_frequencies, double *coeffs, kernel_specification const &kernel_spec) {
    finufft_spread_opts opts = construct_opts_from_kernel(kernel_spec);
    common::onedim_fseries_kernel(num_frequencies, coeffs, opts);
}

} // namespace legacy
} // namespace spreading
} // namespace finufft
