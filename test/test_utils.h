#pragma once

#include <cstddef>
#include <cstdint>

namespace finufft {
namespace testing {

/** Fills the given array with uniform random numbers.
 * 
 * Implementation held separately in order to still enable optimizations even
 * when building for debug builds.
 * 
 */
void fill_random_uniform(float *data, std::size_t size, int32_t seed, float min, float max);
void fill_random_uniform(double *data, std::size_t size, int32_t seed, double min, double max);

// Pause or resume collection using Intel's ITT API if enabled.
void pause_collection();
void resume_collection();

} // namespace testing
} // namespace finufft
