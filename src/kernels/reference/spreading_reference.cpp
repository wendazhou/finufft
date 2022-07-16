#include "spread_reference.h"

#include "gather_fold_reference.h"
#include "spread_bin_sort_reference.h"
#include "spread_processor_reference.h"
#include "spread_subproblem_reference.h"
#include "synchronized_accumulate_reference.h"

#include "../legacy/synchronized_accumulate_legacy.h"

namespace finufft {
namespace spreading {

/** Get default configuration based on reference implementation.
 *
 */
template <typename T, std::size_t Dim>
SpreadFunctorConfiguration<T, Dim>
get_spread_configuration_reference(finufft_spread_opts const &opts) {
    GatherAndFoldReferenceFunctor gather_rescale{
        opts.pirange ? FoldRescaleRange::Pi : FoldRescaleRange::Identity};

    kernel_specification kernel_spec{opts.ES_beta, opts.nspread};
    auto spread_subproblem = get_subproblem_polynomial_reference_functor<T, Dim>(kernel_spec);

    // Use block locking accumulator.
    // Locking overhead should be minimal when number of threads is low, so
    // don't bother special casing a non-synchronized version for single-threaded problem.
    auto accumulate_subgrid_factory = get_reference_block_locking_accumulator<T, Dim>();

    return SpreadFunctorConfiguration<T, Dim>{
        std::move(gather_rescale),
        std::move(spread_subproblem),
        std::move(accumulate_subgrid_factory),
    };
}

namespace {
template <typename T, std::size_t Dim> struct IndirectOmpSpreadFunctor {
    BinSortFunctor<T, Dim> bin_sort_;
    SpreadFunctorConfiguration<T, Dim> config_;
    std::array<std::size_t, Dim> target_size_;
    std::array<T, Dim> bin_size_;
    FoldRescaleRange input_range_;
    finufft::Timer sort_timer_;
    finufft::Timer spread_timer_;

    IndirectOmpSpreadFunctor(
        BinSortFunctor<T, Dim> &&bin_sort, GatherRescaleFunctor<T, Dim> &&gather_rescale,
        SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,
        SynchronizedAccumulateFactory<T, Dim> &&accumulate_subgrid,
        tcb::span<const std::size_t, Dim> target_size, FoldRescaleRange range,
        finufft::Timer const &timer)
        : bin_sort_(std::move(bin_sort)),
          config_{
              std::move(gather_rescale),
              std::move(spread_subproblem),
              std::move(accumulate_subgrid)},
          input_range_(range), sort_timer_(timer.make_timer("sort")),
          spread_timer_(timer.make_timer("spread")) {
        std::copy(target_size.begin(), target_size.end(), target_size_.begin());
        bin_size_.fill(4);
        bin_size_[0] = 16;
    }

    void operator()(nu_point_collection<Dim, const T> points, T *output) const {
        auto sort_idx = finufft::allocate_aligned_array<int64_t>(points.num_points, 64);

        {
            finufft::Timer timer(sort_timer_);
            finufft::ScopedTimerGuard guard(timer);

            std::array<T, Dim> target_size_f;
            std::copy(target_size_.begin(), target_size_.end(), target_size_f.begin());
            bin_sort_(
                sort_idx.get(),
                points.num_points,
                points.coordinates,
                target_size_f,
                bin_size_,
                input_range_);
        }

        auto processor = OmpSpreadProcessor{0, 0};

        {
            finufft::Timer timer(spread_timer_);
            finufft::ScopedTimerGuard guard(timer);

            std::array<int64_t, Dim> target_size_i;
            std::copy(target_size_.begin(), target_size_.end(), target_size_i.begin());
            processor(config_, points, sort_idx.get(), target_size_i, output);
        }
    }
};
} // namespace

template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> make_indirect_omp_spread_functor(
    BinSortFunctor<T, Dim> &&bin_sort_functor, GatherRescaleFunctor<T, Dim> &&gather,
    SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,
    SynchronizedAccumulateFactory<T, Dim> &&accumulate_factory,
    tcb::span<const std::size_t, Dim> target_size, FoldRescaleRange input_range,
    finufft::Timer const &timer) {
    return IndirectOmpSpreadFunctor<T, Dim>(
        std::move(bin_sort_functor),
        std::move(gather),
        std::move(spread_subproblem),
        std::move(accumulate_factory),
        target_size,
        input_range,
        timer);
}

namespace reference {

template <typename T, std::size_t Dim>
SpreadFunctor<T, Dim> get_indirect_spread_functor(
    kernel_specification const &kernel_spec, tcb::span<const std::size_t, Dim> target_size,
    FoldRescaleRange input_range, finufft::Timer const &timer) {
    return make_indirect_omp_spread_functor<T, Dim>(
        get_bin_sort_functor<T, Dim>(),
        GatherAndFoldReferenceFunctor{input_range},
        get_subproblem_polynomial_reference_functor<T, Dim>(kernel_spec),
        get_reference_block_locking_accumulator<T, Dim>(),
        target_size,
        input_range,
        timer);
}

} // namespace reference

#define INSTANTIATE(T, Dim)                                                                        \
    template SpreadFunctorConfiguration<T, Dim> get_spread_configuration_reference(                \
        finufft_spread_opts const &opts);                                                          \
    template SpreadSubproblemFunctor<T, Dim> get_subproblem_polynomial_reference_functor<T, Dim>(  \
        kernel_specification const &);                                                             \
    template SpreadFunctor<T, Dim> make_indirect_omp_spread_functor(                               \
        BinSortFunctor<T, Dim> &&bin_sort_functor,                                                 \
        GatherRescaleFunctor<T, Dim> &&gather,                                                     \
        SpreadSubproblemFunctor<T, Dim> &&spread_subproblem,                                       \
        SynchronizedAccumulateFactory<T, Dim> &&accumulate_factory,                                \
        tcb::span<const std::size_t, Dim>                                                          \
            target_size,                                                                           \
        FoldRescaleRange input_range,                                                              \
        finufft::Timer const &timer);                                                              \
    template SpreadFunctor<T, Dim> reference::get_indirect_spread_functor(                         \
        kernel_specification const &kernel_spec,                                                   \
        tcb::span<const std::size_t, Dim>                                                          \
            target_size,                                                                           \
        FoldRescaleRange input_range,                                                              \
        finufft::Timer const &timer);

INSTANTIATE(float, 1);
INSTANTIATE(float, 2);
INSTANTIATE(float, 3);

INSTANTIATE(double, 1);
INSTANTIATE(double, 2);
INSTANTIATE(double, 3);

#undef INSTANTIATE

} // namespace spreading

} // namespace finufft
