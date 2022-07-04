#include "synchronized_accumulate_reference.h"

namespace finufft {
namespace spreading {

template SynchronizedAccumulateFactory<float, 1> get_reference_locking_accumulator<float, 1>();
template SynchronizedAccumulateFactory<float, 2> get_reference_locking_accumulator<float, 2>();
template SynchronizedAccumulateFactory<float, 3> get_reference_locking_accumulator<float, 3>();

template SynchronizedAccumulateFactory<double, 1> get_reference_locking_accumulator<double, 1>();
template SynchronizedAccumulateFactory<double, 2> get_reference_locking_accumulator<double, 2>();
template SynchronizedAccumulateFactory<double, 3> get_reference_locking_accumulator<double, 3>();


template SynchronizedAccumulateFactory<float, 1> get_reference_block_locking_accumulator<float, 1>();
template SynchronizedAccumulateFactory<float, 2> get_reference_block_locking_accumulator<float, 2>();
template SynchronizedAccumulateFactory<float, 3> get_reference_block_locking_accumulator<float, 3>();

template SynchronizedAccumulateFactory<double, 1> get_reference_block_locking_accumulator<double, 1>();
template SynchronizedAccumulateFactory<double, 2> get_reference_block_locking_accumulator<double, 2>();
template SynchronizedAccumulateFactory<double, 3> get_reference_block_locking_accumulator<double, 3>();

}
} // namespace finufft
