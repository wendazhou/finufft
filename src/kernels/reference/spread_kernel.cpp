#include "spread_kernel.h"

#include "../../constants.h"

#include <cmath>
#include <complex>

namespace finufft {
namespace quadrature {
void legendre_compute_glr(int n, double x[], double w[]);
}

namespace spreading {
namespace reference {

namespace {

template <typename T> T evaluate_es_kernel(T x, kernel_specification const &kernel_spec) {
    T c = static_cast<T>(4.0 / (kernel_spec.width * kernel_spec.width));
    T dist = 1.0 - c * x * x;

    if (dist < 0) {
        return 0;
    }

    return std::exp(static_cast<T>(kernel_spec.es_beta) * std::sqrt(dist));
}

template <typename T>
void onedim_fseries_kernel_impl(
    std::size_t num_frequencies, T *coeffs, kernel_specification const &kernel_spec) {
    static constexpr int MAX_NQUAD = 100;

    T J2 = static_cast<T>(kernel_spec.width / 2.0); // J/2, half-width of ker z-support
    // # quadr nodes in z (from 0 to J/2; reflections will be added)...
    int q = (int)(2 + 3.0 * J2); // not sure why so large? cannot exceed MAX_NQUAD
    T f[MAX_NQUAD];
    double z[2 * MAX_NQUAD], w[2 * MAX_NQUAD];
    quadrature::legendre_compute_glr(2 * q, z, w); // only half the nodes used, eg on (0,1)
    std::complex<T> a[MAX_NQUAD];

    for (int n = 0; n < q; ++n) { // set up nodes z_n and vals f_n
        z[n] *= J2;               // rescale nodes
        f[n] = J2 * static_cast<T>(w[n]) *
               evaluate_es_kernel(static_cast<T>(z[n]), kernel_spec); // vals & quadr wei
        auto phase = 2 * finufft::constants::pi_v<T> * static_cast<T>(num_frequencies / 2 - z[n]) /
                     num_frequencies;
        a[n] = std::exp(std::complex<T>(0, phase)); // phase winding rates
    }

    static constexpr std::size_t chunk_size = 1 << 13;

    std::size_t nout = num_frequencies / 2 + 1; // how many values we're writing to
    std::size_t num_chunks = (nout + chunk_size - 1) / chunk_size;

    // We operate in chunks over the output array, in order
    // to strike a balance between performance, multithreading and floating point determinism.
    for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
        // Initialize local phase factors for this chunk
        std::complex<T> aj[MAX_NQUAD];

        std::transform(a, a + q, aj, [&](std::complex<T> const &a) {
            return std::pow(a, static_cast<T>(chunk * chunk_size));
        });

        std::size_t this_chunk_size = std::min(chunk_size, nout - chunk * chunk_size);
        for (std::size_t i = 0; i < this_chunk_size; ++i) {
            std::size_t j = chunk * chunk_size + i;
            T x = 0;
            for (std::size_t n = 0; n < q; ++n) {
                x += 2 * f[n] * std::real(aj[n]);
                aj[n] *= a[n]; // wind phases
            }
            coeffs[j] = x;
        }
    }
}
} // namespace

void onedim_fseries_kernel(
    std::size_t num_frequencies, float *coeffs, kernel_specification const &kernel_spec) {
    onedim_fseries_kernel_impl(num_frequencies, coeffs, kernel_spec);
}
void onedim_fseries_kernel(
    std::size_t num_frequencies, double *coeffs, kernel_specification const &kernel_spec) {
    onedim_fseries_kernel_impl(num_frequencies, coeffs, kernel_spec);
}

namespace {

template <typename T> struct InterpolationKernelFactoryImpl {
    kernel_specification kernel_spec_;

    InterpolationKernelFactoryImpl(kernel_specification const &kernel_spec)
        : kernel_spec_(kernel_spec) {}

    void operator()(T *coeffs, std::size_t num_frequencies) const noexcept {
        onedim_fseries_kernel(num_frequencies, coeffs, kernel_spec_);
    }
};
} // namespace

template <typename T>
interpolation::InterpolationKernelFactory<T>
make_interpolation_kernel_factory(kernel_specification const &kernel_spec) {
    return InterpolationKernelFactoryImpl<T>(kernel_spec);
}

kernel_specification get_default_kernel_specification(double tolerance, double upsampling_factor) {
    std::size_t width;

    if (upsampling_factor == 2.0) {
        // standard sigma (see SISC paper)
        width = std::ceil(-std::log10(tolerance / 10.0)); // 1 digit per power of 10
    } else {
        // custom sigma
        width = std::ceil(
            -log(tolerance) /
            (constants::pi_v<double> * std::sqrt(1.0 - 1.0 / upsampling_factor))); // formula, gam=1
    }
    width = std::max(std::size_t(2), width); // (we don't have ns=1 version yet)

    double beta_over_width;
    switch (width) {
    case 2:
        beta_over_width = 2.20;
        break;
    case 3:
        beta_over_width = 2.26;
        break;
    case 4:
        beta_over_width = 2.38;
        break;
    default:
        beta_over_width = 2.30;
    }

    if (upsampling_factor != 2.0) {
        double gamma = 0.97;
        beta_over_width = gamma * constants::pi_v<double> * (1.0 - 0.5 / upsampling_factor);
    }

    double beta = beta_over_width * width;
    return {beta, static_cast<int>(width)};
}

#define INSTANTIATE(T)                                                                             \
    template interpolation::InterpolationKernelFactory<T> make_interpolation_kernel_factory(       \
        kernel_specification const &kernel_spec);

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace reference
} // namespace spreading
} // namespace finufft