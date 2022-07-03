#include "test_utils.h"

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

namespace finufft {
namespace testing {

void fill_random_uniform(float *data, std::size_t size, int32_t seed, float min, float max) {
    typedef r123::Philox4x32 RNG;

    RNG rng;
    RNG::ukey_type uk = {{}};
    RNG::ctr_type c = {{}};

    uk[0] = seed;

#pragma omp parallel for
    for (std::size_t i = 0; i < size / 4; ++i) {
        c[0] = i;
        auto r = rng(c, uk);

        for (std::size_t j = 0; j < 4; ++j) {
            data[4 * i + j] = r123::u01<float>(r[j]) * (max - min) + min;
        }
    }

    {
        std::size_t i = size / 4 * 4;
        c[0] = i;
        auto r = rng(c, uk);

        for (std::size_t j = 0; j + i < size; ++j) {
            data[i + j] = r123::u01<float>(r[j]) * (max - min) + min;
        }
    }
}

void fill_random_uniform(double *data, std::size_t size, int32_t seed, double min, double max) {
    typedef r123::Philox4x32 RNG;

    RNG rng;
    RNG::ukey_type uk = {{}};
    RNG::ctr_type c = {{}};

    uk[0] = seed;

    std::size_t i = 0;

#pragma omp parallel for
    for (std::size_t i = 0; i < size / 4; ++i) {
        c[0] = i;
        auto r = rng(c, uk);

        for (std::size_t j = 0; j < 4; ++j) {
            data[4 * i + j] = r123::u01<double>(r[j]) * (max - min) + min;
        }
    }

    {
        std::size_t i = size / 4 * 4;
        c[0] = i;
        auto r = rng(c, uk);

        for (std::size_t j = 0; j + i < size; ++j) {
            data[i + j] = r123::u01<double>(r[j]) * (max - min) + min;
        }
    }
}

} // namespace testing
} // namespace finufft
