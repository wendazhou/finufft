#include "spread_legacy.h"

#include <cstring>

#include "../reference/gather_fold_reference.h"
#include "spread_bin_sort_legacy.h"
#include "spread_subproblem_legacy.h"
#include "synchronized_accumulate_legacy.h"

namespace finufft {
namespace spreadinterp {

int spreadSortedOriginal(
    int64_t *sort_indices, int64_t N1, int64_t N2, int64_t N3, float *data_uniform, int64_t M,
    float *kx, float *ky, float *kz, float *data_nonuniform, finufft_spread_opts opts,
    int did_sort);
int spreadSortedOriginal(
    int64_t *sort_indices, int64_t N1, int64_t N2, int64_t N3, double *data_uniform, int64_t M,
    double *kx, double *ky, double *kz, double *data_nonuniform, finufft_spread_opts opts,
    int did_sort);

} // namespace spreadinterp
} // namespace finufft

namespace finufft {
namespace spreading {

template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_legacy(finufft_spread_opts const &opts) {
    GatherAndFoldReferenceFunctor gather_rescale{
        opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity};

    kernel_specification kernel_spec{opts.ES_beta, opts.nspread};
    SpreadSubproblemLegacyFunctor<T, Dim> spread_subproblem{kernel_spec};

    auto accumulate_subgrid_factory = (opts.nthreads > opts.atomic_threshold || opts.nthreads == 0)
                                          ? get_legacy_atomic_accumulator<T, Dim>()
                                          : get_legacy_locking_accumulator<T, Dim>();

    return SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_subproblem),
        std::move(accumulate_subgrid_factory),
    };
}

namespace legacy {

finufft_spread_opts
construct_opts_from_kernel(const kernel_specification &kernel, std::size_t dim) {
    finufft_spread_opts opts;

    std::memset(&opts, 0, sizeof(opts));

    opts.nspread = kernel.width;
    opts.kerevalmeth = 1;
    opts.kerpad = 1;
    opts.ES_beta = kernel.es_beta;
    opts.ES_c = 4.0 / (kernel.width * kernel.width);
    opts.ES_halfwidth = (double)kernel.width / 2.0;
    opts.flags = 0;

    opts.max_subproblem_size = (dim == 1) ? 10'000 : 100'000;
    opts.atomic_threshold = 10;

    // Reverse engineer upsampling factor from design in `setup_spreader`
    // This is necessary to call reference implementation, despite the fact
    // that the kernel is fully specified through the beta, c, and width parameters.
    double beta_over_ns = kernel.es_beta / kernel.width;

    // Baseline value of beta_over_ns for upsampling factor of 2
    double baseline_beta_over_ns = 2.3;
    switch (kernel.width) {
    case 2:
        baseline_beta_over_ns = 2.2;
    case 3:
        baseline_beta_over_ns = 2.26;
    case 4:
        baseline_beta_over_ns = 2.38;
    }

    if (std::abs(beta_over_ns - baseline_beta_over_ns) < 1e-6) {
        // Special-cased for upsampling factor of 2
        opts.upsampfac = 2;
    } else {
        // General formula, round obtained value in order to produce exact match if necessary.
        double upsamp_factor = 0.5 / (1 - beta_over_ns / (0.97 * M_PI));
        opts.upsampfac = std::round(upsamp_factor * 1000) / 1000;
    }

    // Use a single thread
    opts.nthreads = 1;

    return opts;
}

namespace {

/** Current implementation of spreading.
 *
 * This functor implements spreading as dispatched to the current finufft implementation.
 * It operates in two steps:
 * - a bin-sorting step, where points are sorted into bins according to their location in
 * the target bufer, and
 * - a spreading step, where points are spread into the target buffer (using intermediate
 * buffers as necessary).
 *
 */
template <typename T, std::size_t Dim> struct LegacySpreadFunctorImplementation {
    std::array<std::size_t, Dim> size_;
    std::array<T, Dim> size_f_;
    std::array<T, Dim> bin_size_;
    FoldRescaleRange input_range_;
    kernel_specification kernel_;
    finufft::Timer sort_timer_;
    finufft::Timer spread_timer_;

    LegacySpreadFunctorImplementation(
        kernel_specification const &kernel, FoldRescaleRange input_range,
        tcb::span<const std::size_t, Dim> size, finufft::Timer const &timer)
        : input_range_(input_range), kernel_(kernel), sort_timer_(timer.make_timer("sort")),
          spread_timer_(timer.make_timer("spread")) {
        std::copy(size.begin(), size.end(), size_.begin());
        std::copy(size.begin(), size.end(), size_f_.begin());
        bin_size_.fill(4);
        bin_size_[0] = 16;
    }

    void operator()(nu_point_collection<Dim, const T> points, T *output) const {
        auto sort_idx = finufft::allocate_aligned_array<int64_t>(points.num_points, 64);

        // Bin-sort points
        {
            finufft::Timer timer(sort_timer_);
            finufft::ScopedTimerGuard guard(timer);
            bin_sort_multithread_legacy<T, Dim>(
                sort_idx.get(),
                points.num_points,
                points.coordinates,
                size_f_,
                bin_size_,
                input_range_);
        }

        // Spread points
        {
            finufft::Timer timer(spread_timer_);
            finufft::ScopedTimerGuard guard(timer);

            auto opts = construct_opts_from_kernel(kernel_, Dim);
            opts.pirange = input_range_ == FoldRescaleRange::Pi;

            finufft::spreadinterp::spreadSortedOriginal(
                sort_idx.get(),
                size_[0],
                Dim > 1 ? size_[1] : 1,
                Dim > 2 ? size_[2] : 1,
                output,
                points.num_points,
                const_cast<T *>(points.coordinates[0]),
                Dim > 1 ? const_cast<T *>(points.coordinates[1]) : nullptr,
                Dim > 2 ? const_cast<T *>(points.coordinates[2]) : nullptr,
                const_cast<T *>(points.strengths),
                opts,
                1);
        }
    }
};

} // namespace

template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> make_spread_functor(
    kernel_specification const &kernel_spec, FoldRescaleRange input_range,
    tcb::span<const std::size_t, Dim> size, finufft::Timer const &timer) {
    return LegacySpreadFunctorImplementation<T, Dim>(kernel_spec, input_range, size, timer);
}

#define INSTANTIATE(T, Dim)                                                                        \
    template SpreadFunctor<T, Dim> make_spread_functor(                                            \
        kernel_specification const &kernel_spec,                                                   \
        FoldRescaleRange input_range,                                                              \
        tcb::span<const std::size_t, Dim>                                                          \
            size,                                                                                  \
        finufft::Timer const &timer);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

#undef INSTANTIATE

} // namespace legacy

template SpreadFunctorConfiguration<float, 1>
get_spread_configuration_legacy<float, 1>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<float, 2>
get_spread_configuration_legacy<float, 2>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<float, 3>
get_spread_configuration_legacy<float, 3>(finufft_spread_opts const &);

template SpreadFunctorConfiguration<double, 1>
get_spread_configuration_legacy<double, 1>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<double, 2>
get_spread_configuration_legacy<double, 2>(finufft_spread_opts const &);
template SpreadFunctorConfiguration<double, 3>
get_spread_configuration_legacy<double, 3>(finufft_spread_opts const &);

} // namespace spreading
} // namespace finufft
