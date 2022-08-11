#include "test_utils.h"

#include <ittnotify.h>

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

void fill_random_uniform(uint32_t *data, std::size_t elements, int seed) {
    typedef r123::Philox4x32 RNG;
    RNG rng;
    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};
    uk[0] = seed;

    std::size_t i = 0;

    for (; i + 3 < elements; i += 4) {
        c[0] = i;
        auto r = rng(c, uk);
        data[i] = r[0];
        data[i + 1] = r[1];
        data[i + 2] = r[2];
        data[i + 3] = r[3];
    }

    {
        c[0] = i;
        auto r = rng(c, uk);
        for (std::size_t j = 0; i < elements; ++i, ++j) {
            data[i] = r[j];
        }
    }
}

void fill_random_uniform(uint64_t *data, std::size_t elements, int seed) {
    typedef r123::Philox4x32 RNG;
    RNG rng;
    RNG::ctr_type c = {{}};
    RNG::ukey_type uk = {{}};
    uk[0] = seed;

    std::size_t i = 0;

    for (; i + 1 < elements; i += 2) {
        c[0] = i;
        auto r = rng(c, uk);
        data[i] = ((uint64_t)r[1] << 32) + r[0];
        data[i + 1] = ((uint64_t)r[3] << 32) + r[2];
    }

    {
        c[0] = i;
        auto r = rng(c, uk);
        for (std::size_t j = 0; i < elements; ++i, ++j) {
            data[i] = ((uint64_t)r[2 * j + 1] << 32) + r[2 * j];
        }
    }
}

void pause_collection() {
    __itt_pause();
}

void resume_collection() {
    __itt_resume();
}

} // namespace testing
} // namespace finufft
