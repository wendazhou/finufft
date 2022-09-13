#include "../fft.h"

#include <fftw3.h>

/** @file
 *
 * Adapters to the FFTW library for planning Fourier transforms.
 * Note that we adapt many of the functions to a templated type-generic
 * interface.
 *
 */

namespace finufft {
namespace fft {
namespace {

template <typename T> struct fftw_plan_generic;
template <> struct fftw_plan_generic<float> { using type = ::fftwf_plan; };
template <> struct fftw_plan_generic<double> { using type = ::fftw_plan; };
template <typename T> using fftw_plan = typename fftw_plan_generic<T>::type;

template <typename T> struct FFFTWPlanDeleter;
template <> struct FFFTWPlanDeleter<float> {
    void operator()(fftw_plan<float> plan) const noexcept { fftwf_destroy_plan(plan); }
};
template <> struct FFFTWPlanDeleter<double> {
    void operator()(fftw_plan<double> plan) const noexcept { fftw_destroy_plan(plan); }
};

template <typename T> using FFTWPlanHolder = std::shared_ptr<std::remove_pointer_t<fftw_plan<T>>>;

template <typename T> struct fftw_iodim64_generic;
template <> struct fftw_iodim64_generic<float> { using type = ::fftwf_iodim64; };
template <> struct fftw_iodim64_generic<double> { using type = ::fftw_iodim64; };
template <typename T> using fftw_iodim64 = typename fftw_iodim64_generic<T>::type;

template <typename T> struct fftw_complex_generic;
template <> struct fftw_complex_generic<float> { using type = ::fftwf_complex; };
template <> struct fftw_complex_generic<double> { using type = ::fftw_complex; };
template <typename T> using fftw_complex = typename fftw_complex_generic<T>::type;

template <typename T>
fftw_plan<T> fftw_plan_guru64_dft(
    int rank, const fftw_iodim64<T> *dims, int howmany_rank, const fftw_iodim64<T> *howmany_dims,
    fftw_complex<T> *in, fftw_complex<T> *out, int sign, unsigned flags);
template <>
fftw_plan<float> fftw_plan_guru64_dft<float>(
    int rank, const fftw_iodim64<float> *dims, int howmany_rank,
    const fftw_iodim64<float> *howmany_dims, fftw_complex<float> *in, fftw_complex<float> *out,
    int sign, unsigned flags) {
    return ::fftwf_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
}
template <>
fftw_plan<double> fftw_plan_guru64_dft<double>(
    int rank, const fftw_iodim64<double> *dims, int howmany_rank,
    const fftw_iodim64<float> *howmany_dims, fftw_complex<double> *in, fftw_complex<double> *out,
    int sign, unsigned flags) {
    return ::fftw_plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
}

template <typename T, std::size_t Dim>
FFTWPlanHolder<T> make_fftw_plan(
    finufft::fft::FourierTransformDirection direction, tcb::span<const std::size_t, Dim> size,
    tcb::span<const std::size_t, Dim> stride, T *data, std::size_t n_batch = 1,
    std::size_t stride_batch = 0) {
    fftw_iodim64<T> dims[Dim];
    fftw_iodim64<T> howmany_dims[1];

    for (std::size_t i = 0; i < Dim; ++i) {
        dims[i].n = size[i];
        dims[i].is = stride[i];
        dims[i].os = stride[i];
    }

    howmany_dims[0].n = n_batch;
    howmany_dims[0].is = stride_batch;
    howmany_dims[0].os = stride_batch;

    fftw_plan<T> plan = fftw_plan_guru64_dft<T>(
        Dim,
        dims,
        1,
        howmany_dims,
        reinterpret_cast<fftw_complex<T> *>(data),
        reinterpret_cast<fftw_complex<T> *>(data),
        FFTW_FORWARD,
        FFTW_ESTIMATE);

    if (plan == nullptr) {
        throw std::runtime_error("Failed to create FFTW plan.");
    }

    return FFTWPlanHolder<T>(plan, FFFTWPlanDeleter<T>());
}

/** This class encapsulates a planned FFTW transform and provides
 * an object-oriented interface for calling the plan.
 *
 * Note that the plan is held through a shared pointer, so that
 * the planned transform may be copied and used in multiple places.
 *
 */
template <typename T, std::size_t Dim> struct FFTWPlannedTransform;

template <std::size_t Dim> struct FFTWPlannedTransform<float, Dim> {
    FFTWPlanHolder<float> plan_;
    FFTWPlannedTransform(FFTWPlanHolder<float> &&plan) : plan_(std::move(plan)) {}
    void operator()(float *data) const noexcept {
        auto cdata = reinterpret_cast<fftw_complex<float> *>(data);
        fftwf_execute_dft(plan_.get(), cdata, cdata);
    }
};

template <std::size_t Dim> struct FFTWPlannedTransform<double, Dim> {
    FFTWPlanHolder<double> plan_;
    FFTWPlannedTransform(FFTWPlanHolder<double> &&plan) : plan_(std::move(plan)) {}
    void operator()(double *data) const noexcept {
        auto cdata = reinterpret_cast<fftw_complex<double> *>(data);
        fftw_execute_dft(plan_.get(), cdata, cdata);
    }
};

} // namespace

template <typename T, std::size_t Dim>
PlannedFourierTransformation<T> make_fftw_planned_transform(
    FourierTransformDirection direction, tcb::span<const std::size_t, Dim> size,
    tcb::span<const std::size_t, Dim> stride, T *data, std::size_t n_batch,
    std::size_t stride_batch) {
    auto plan = make_fftw_plan(direction, size, stride, data, n_batch, stride_batch);
    return FFTWPlannedTransform<T, Dim>(std::move(plan));
}

#define INSTANTIATE(T, Dim)                                                                        \
    template PlannedFourierTransformation<T> make_fftw_planned_transform(                          \
        FourierTransformDirection direction,                                                       \
        tcb::span<const std::size_t, Dim>                                                          \
            size,                                                                                  \
        tcb::span<const std::size_t, Dim>                                                          \
            stride,                                                                                \
        T *data,                                                                                   \
        std::size_t n_batch,                                                                       \
        std::size_t stride_batch);

INSTANTIATE(float, 1)
INSTANTIATE(float, 2)
INSTANTIATE(float, 3)

INSTANTIATE(double, 1)
INSTANTIATE(double, 2)
INSTANTIATE(double, 3)

#undef INSTANTIATE

} // namespace fft
} // namespace finufft
