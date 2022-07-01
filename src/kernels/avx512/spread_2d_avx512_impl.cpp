#include "spread_2d_avx512_impl.h"
#include "spread_axv512.h"

namespace finufft {
namespace spreading {
namespace avx512 {

template struct SpreadSubproblemPoly2DW8<4>;
template struct SpreadSubproblemPoly2DW8<5>;
template struct SpreadSubproblemPoly2DW8<6>;
template struct SpreadSubproblemPoly2DW8<7>;
template struct SpreadSubproblemPoly2DW8<8>;
template struct SpreadSubproblemPoly2DW8<9>;
template struct SpreadSubproblemPoly2DW8<10>;
template struct SpreadSubproblemPoly2DW8<11>;

template struct SpreadSubproblemPoly2DW8F64<4>;
template struct SpreadSubproblemPoly2DW8F64<5>;
template struct SpreadSubproblemPoly2DW8F64<6>;
template struct SpreadSubproblemPoly2DW8F64<7>;
template struct SpreadSubproblemPoly2DW8F64<8>;
template struct SpreadSubproblemPoly2DW8F64<9>;
template struct SpreadSubproblemPoly2DW8F64<10>;
template struct SpreadSubproblemPoly2DW8F64<11>;

} // namespace avx512

namespace {
finufft::detail::precomputed_poly_kernel_data const &
search_precomputed(kernel_specification const &kernel) {
    auto it = std::find_if(
        finufft::detail::precomputed_poly_kernel_data_table.begin(),
        finufft::detail::precomputed_poly_kernel_data_table.end(),
        [&kernel](finufft::detail::precomputed_poly_kernel_data const &entry) {
            return (entry.width == kernel.width) &&
                   (entry.beta_1000 == static_cast<std::size_t>(kernel.es_beta * 1000));
        });

    // Could not find matching kernel, throw an exception.
    if (it == finufft::detail::precomputed_poly_kernel_data_table.end()) {
        throw std::runtime_error("No precomputed polynomial kernel data found for given kernel.");
    }
    return *it;
}
} // namespace

SpreadSubproblemFunctor<float, 2>
get_subproblem_polynomial_avx512_2d_fp32_functor(kernel_specification const &kernel) {
    auto it = search_precomputed(kernel);

    if (it.width > 8) {
        throw std::runtime_error("Precomputed polynomial kernel data for width > 8 not supported.");
    }

    switch (it.degree) {
    case 4:
        return avx512::SpreadSubproblemPoly2DW8<4>(it.coefficients, it.width);
    case 5:
        return avx512::SpreadSubproblemPoly2DW8<5>(it.coefficients, it.width);
    case 6:
        return avx512::SpreadSubproblemPoly2DW8<6>(it.coefficients, it.width);
    case 7:
        return avx512::SpreadSubproblemPoly2DW8<7>(it.coefficients, it.width);
    case 8:
        return avx512::SpreadSubproblemPoly2DW8<8>(it.coefficients, it.width);
    case 9:
        return avx512::SpreadSubproblemPoly2DW8<9>(it.coefficients, it.width);
    case 10:
        return avx512::SpreadSubproblemPoly2DW8<10>(it.coefficients, it.width);
    case 11:
        return avx512::SpreadSubproblemPoly2DW8<11>(it.coefficients, it.width);
    default:
        throw std::runtime_error(
            "Precomputed polynomial kernel data for degree > 11 or degree < 4 not supported.");
    }
}

SpreadSubproblemFunctor<double, 2>
get_subproblem_polynomial_avx512_2d_fp64_functor(kernel_specification const &kernel) {
    auto it = search_precomputed(kernel);

    if (it.width > 8) {
        throw std::runtime_error("Precomputed polynomial kernel data for width > 8 not supported.");
    }

    switch (it.degree) {
    case 4:
        return avx512::SpreadSubproblemPoly2DW8F64<4>(it.coefficients, it.width);
    case 5:
        return avx512::SpreadSubproblemPoly2DW8F64<5>(it.coefficients, it.width);
    case 6:
        return avx512::SpreadSubproblemPoly2DW8F64<6>(it.coefficients, it.width);
    case 7:
        return avx512::SpreadSubproblemPoly2DW8F64<7>(it.coefficients, it.width);
    case 8:
        return avx512::SpreadSubproblemPoly2DW8F64<8>(it.coefficients, it.width);
    case 9:
        return avx512::SpreadSubproblemPoly2DW8F64<9>(it.coefficients, it.width);
    case 10:
        return avx512::SpreadSubproblemPoly2DW8F64<10>(it.coefficients, it.width);
    case 11:
        return avx512::SpreadSubproblemPoly2DW8F64<11>(it.coefficients, it.width);
    default:
        throw std::runtime_error(
            "Precomputed polynomial kernel data for degree > 11 or degree < 4 not supported.");
    }
}

} // namespace spreading
} // namespace finufft