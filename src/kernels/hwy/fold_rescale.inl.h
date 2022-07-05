#if defined(FINUFFT_KERNELS_HWY_FOLD_RESCALE_INL_H) == defined(HWY_TARGET_TOGGLE)
#ifdef FINUFFT_KERNELS_HWY_FOLD_RESCALE_INL_H
#undef FINUFFT_KERNELS_HWY_FOLD_RESCALE_INL_H
#else
#define FINUFFT_KERNELS_HWY_FOLD_RESCALE_INL_H
#endif

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#define FINUFFT_SET_MATH_DEFINES
#endif
#include <cmath>
#ifdef FINUFFT_SET_MATH_DEFINES
#undef _USE_MATH_DEFINES
#undef FINUFFT_SET_MATH_DEFINES
#endif

#include "hwy/highway.h"
#include "../reference/gather_fold_reference.h"

HWY_BEFORE_NAMESPACE();
namespace finufft {
namespace spreading {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T> struct FoldRescalePi : finufft::spreading::FoldRescalePi<T> {
    using finufft::spreading::FoldRescalePi<T>::operator();

    template <typename Vec, typename Tag> Vec operator()(Vec x, T extent, Tag d) const noexcept {
        T pi_v = M_PI;

        auto pi = hn::Set(d, pi_v);
        auto two_pi = hn::Set(d, 2 * pi_v);
        auto extent_over_2_pi = hn::Set(d, extent * 0.5 * M_1_PI);

        x = hn::IfThenElse(hn::Lt(x, hn::Neg(pi)), hn::Add(x, two_pi), x);
        x = hn::IfThenElse(hn::Gt(x, pi), hn::Sub(x, two_pi), x);

        return hn::Mul(hn::Add(x, pi), extent_over_2_pi);
    }
};

template <typename T> struct FoldRescaleIdentity : finufft::spreading::FoldRescaleIdentity<T> {
    using finufft::spreading::FoldRescaleIdentity<T>::operator();

    template <typename Vec, typename Tag> Vec operator()(Vec x, T extent, Tag d) const noexcept {
        auto extent_v = hn::Set(d, extent);

        x = hn::IfThenElse(hn::Lt(x, hn::Zero(d)), hn::Add(x, extent_v), x);
        x = hn::IfThenElse(hn::Gt(x, extent_v), hn::Sub(x, extent_v), x);

        return x;
    }
};

} // namespace HWY_NAMESPACE
} // namespace spreading
} // namespace finufft

#endif
