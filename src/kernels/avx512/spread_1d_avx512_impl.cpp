#include "spread_1d_avx512_impl.h"
#include "spread_avx512.h"

namespace finufft {
namespace spreading {
namespace avx512 {

template struct SpreadSubproblemPolyW8<7>;
template struct SpreadSubproblemPolyW8<8>;
template struct SpreadSubproblemPolyW8<9>;
template struct SpreadSubproblemPolyW8<10>;
template struct SpreadSubproblemPolyW8<11>;

template struct SpreadSubproblemPolyW4<4>;
template struct SpreadSubproblemPolyW4<5>;
template struct SpreadSubproblemPolyW4<6>;
template struct SpreadSubproblemPolyW4<7>;

template struct SpreadSubproblemPolyW8F64<7>;
template struct SpreadSubproblemPolyW8F64<8>;
template struct SpreadSubproblemPolyW8F64<9>;
template struct SpreadSubproblemPolyW8F64<10>;
template struct SpreadSubproblemPolyW8F64<11>;

} // namespace avx512

SpreadSubproblemFunctor<float, 1>
get_subproblem_polynomial_avx512_1d_fp32_functor(kernel_specification const &kernel) {
    auto it = finufft::detail::search_precomputed_poly_kernel(kernel);

    if (!it) {
        throw std::runtime_error("No precomputed polynomial kernel data found for given kernel.");
    }

    if (it->width > 8) {
        throw std::runtime_error("Precomputed polynomial kernel data for width > 8 not supported.");
    }

    if (it->width <= 4) {
        switch (it->degree) {
        case 4:
            return avx512::SpreadSubproblemPolyW4<4>(it->coefficients, it->width);
        case 5:
            return avx512::SpreadSubproblemPolyW4<5>(it->coefficients, it->width);
        case 6:
            return avx512::SpreadSubproblemPolyW4<6>(it->coefficients, it->width);
        case 7:
            return avx512::SpreadSubproblemPolyW4<7>(it->coefficients, it->width);
        }

        // Fall through? Valid but performance may be sub-par.
    }

    switch (it->degree) {
    case 7:
        return avx512::SpreadSubproblemPolyW8<7>(it->coefficients, it->width);
    case 8:
        return avx512::SpreadSubproblemPolyW8<8>(it->coefficients, it->width);
    case 9:
        return avx512::SpreadSubproblemPolyW8<9>(it->coefficients, it->width);
    case 10:
        return avx512::SpreadSubproblemPolyW8<10>(it->coefficients, it->width);
    case 11:
        return avx512::SpreadSubproblemPolyW8<11>(it->coefficients, it->width);
    default:
        throw std::runtime_error(
            "Precomputed polynomial kernel data for degree > 11 or degree < 7 not supported.");
    }
}

SpreadSubproblemFunctor<double, 1>
get_subproblem_polynomial_avx512_1d_fp64_functor(kernel_specification const &kernel) {
    auto it = finufft::detail::search_precomputed_poly_kernel(kernel);

    if (!it) {
        throw std::runtime_error("No precomputed polynomial kernel data found for given kernel.");
    }

    if (it->width > 8) {
        throw std::runtime_error("Precomputed polynomial kernel data for width > 8 not supported.");
    }

    switch (it->degree) {
    case 4:
        return avx512::SpreadSubproblemPolyW8F64<4>(it->coefficients, it->width);
    case 5:
        return avx512::SpreadSubproblemPolyW8F64<5>(it->coefficients, it->width);
    case 6:
        return avx512::SpreadSubproblemPolyW8F64<6>(it->coefficients, it->width);
    case 7:
        return avx512::SpreadSubproblemPolyW8F64<7>(it->coefficients, it->width);
    case 8:
        return avx512::SpreadSubproblemPolyW8F64<8>(it->coefficients, it->width);
    case 9:
        return avx512::SpreadSubproblemPolyW8F64<9>(it->coefficients, it->width);
    case 10:
        return avx512::SpreadSubproblemPolyW8F64<10>(it->coefficients, it->width);
    case 11:
        return avx512::SpreadSubproblemPolyW8F64<11>(it->coefficients, it->width);
    default:
        throw std::runtime_error(
            "Precomputed polynomial kernel data for degree > 11 or degree < 4 not supported.");
    }
}

} // namespace spreading
} // namespace finufft
