#include "synchronized_accumulate_reference.h"

namespace finufft {
namespace spreading {

#define FINUFFT_DEFINE_ACCUMULATE_FACTORY_IMPLEMENTATIONS(fn, type, dim)                           \
    template SynchronizedAccumulateFactory<type, dim> fn<type, dim>();

#define FINUFFT_DEFINE_STANDARD_ACCUMULATE_FACTORY_IMPLEMENTATIONS(fn)                             \
    FINUFFT_DEFINE_ACCUMULATE_FACTORY_IMPLEMENTATIONS(fn, float, 1)                                \
    FINUFFT_DEFINE_ACCUMULATE_FACTORY_IMPLEMENTATIONS(fn, float, 2)                                \
    FINUFFT_DEFINE_ACCUMULATE_FACTORY_IMPLEMENTATIONS(fn, float, 3)                                \
    FINUFFT_DEFINE_ACCUMULATE_FACTORY_IMPLEMENTATIONS(fn, double, 1)                               \
    FINUFFT_DEFINE_ACCUMULATE_FACTORY_IMPLEMENTATIONS(fn, double, 2)                               \
    FINUFFT_DEFINE_ACCUMULATE_FACTORY_IMPLEMENTATIONS(fn, double, 3)

FINUFFT_DEFINE_STANDARD_ACCUMULATE_FACTORY_IMPLEMENTATIONS(get_reference_locking_accumulator)
FINUFFT_DEFINE_STANDARD_ACCUMULATE_FACTORY_IMPLEMENTATIONS(get_reference_block_locking_accumulator)
FINUFFT_DEFINE_STANDARD_ACCUMULATE_FACTORY_IMPLEMENTATIONS(get_reference_singlethreaded_accumulator)

#undef FINUFFT_DEFINE_STANDARD_ACCUMULATE_FACTORY_IMPLEMENTATIONS
#undef FINUFFT_DEFINE_ACCUMULATE_FACTORY_IMPLEMENTATIONS

} // namespace spreading
} // namespace finufft
