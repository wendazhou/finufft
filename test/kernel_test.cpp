#include <gtest/gtest.h>
#include <vector>
#include "../src/ker_horner_avx2.h"

namespace {

#ifndef FLT
#define FLT float
#endif

template<int w>
inline void accumulate_kernel_vec_horner(float* out, const float x, float w_re, float w_im) {
    const int w_rounded = (w + 3) / 4 * 4;

    float ker[w_rounded];
    float z = 2 * x + w - 1.0;

#include "../src/ker_horner_allw_loop.c"

    for (int j = 0; j < w_rounded; ++j) {
        out[2 * j] += w_re * ker[j];
        out[2 * j + 1] += w_im * ker[j];
    }
}

#undef FLT


template<typename Fn>
std::vector<float> evaluate_kernel(int w, Fn&& fn) {
    const int w_rounded = (w + 3) / 4 * 4;
    std::vector<float> result(2 * w_rounded);

    float points[] = {-0.5, 0.0, 0.3};

    for (auto p : points) {
        fn(result.data(), p, 1.0, 2.0);
    }

    return result;
}

}

TEST(KernelTest, Kernel7HornerAvx2) {
    auto result_reference = evaluate_kernel(7, accumulate_kernel_vec_horner<7>);
    auto result_avx2 = evaluate_kernel(7, accumulate_kernel_vec_horner_7_avx2);

    for(int i = 0; i < result_reference.size(); ++i) {
        EXPECT_FLOAT_EQ(result_reference[i], result_avx2[i]);
    }
}
