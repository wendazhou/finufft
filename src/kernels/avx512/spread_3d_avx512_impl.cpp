#include "spread_3d_avx512_impl.h"

namespace finufft {
namespace spreading {
namespace avx512 {

template struct SpreadSubproblemPoly3DW8<4>;
template struct SpreadSubproblemPoly3DW8<5>;
template struct SpreadSubproblemPoly3DW8<6>;
template struct SpreadSubproblemPoly3DW8<7>;
template struct SpreadSubproblemPoly3DW8<8>;
template struct SpreadSubproblemPoly3DW8<9>;
template struct SpreadSubproblemPoly3DW8<10>;
template struct SpreadSubproblemPoly3DW8<11>;

} // namespace avx512

SpreadSubproblemFunctor<float, 3>
get_subproblem_polynomial_avx512_3d_fp32_functor(kernel_specification const &kernel) {
    auto it = finufft::detail::search_precomputed_poly_kernel(kernel);

    if (!it) {
        throw std::runtime_error("No precomputed polynomial kernel data found for given kernel.");
    }

    if (it->width > 8) {
        throw std::runtime_error("Precomputed polynomial kernel data for width > 8 not supported.");
    }

    switch (it->degree) {
    case 4:
        return avx512::SpreadSubproblemPoly3DW8<4>(it->coefficients, it->width);
    case 5:
        return avx512::SpreadSubproblemPoly3DW8<5>(it->coefficients, it->width);
    case 6:
        return avx512::SpreadSubproblemPoly3DW8<6>(it->coefficients, it->width);
    case 7:
        return avx512::SpreadSubproblemPoly3DW8<7>(it->coefficients, it->width);
    case 8:
        return avx512::SpreadSubproblemPoly3DW8<8>(it->coefficients, it->width);
    case 9:
        return avx512::SpreadSubproblemPoly3DW8<9>(it->coefficients, it->width);
    case 10:
        return avx512::SpreadSubproblemPoly3DW8<10>(it->coefficients, it->width);
    case 11:
        return avx512::SpreadSubproblemPoly3DW8<11>(it->coefficients, it->width);
    default:
        throw std::runtime_error(
            "Precomputed polynomial kernel data for degree > 11 or degree < 4 not supported.");
    }
}

} // namespace spreading
} // namespace finufft
