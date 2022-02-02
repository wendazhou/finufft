#include <vector>

#include <benchmark/benchmark.h>

#include <Random123/philox.h>
#include <Random123/uniform.hpp>

#include <spreadinterp.h>

#include <immintrin.h>


void spread_subproblem_1d(BIGINT off1, BIGINT size1, float *du, BIGINT M, float *kx, float  *dd, const spread_opts& opts);

namespace {

#ifdef FLT
#undef FLT
#endif

#define FLT float
__attribute__((always_inline)) inline void eval_kernel_vec_Horner(FLT *ker, const FLT x, const int w)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
    FLT z = 2*x + w - 1.0;         // scale so local grid offset z in [-1,1]
    // insert the auto-generated code which expects z, w args, writes to ker...
    #include "../src/ker_horner_allw_loop.c"
}

template<int w>
__attribute__((always_inline)) inline void accumulate_kernel_vec_horner(float* out, const float x, float w_re, float w_im) {
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

void accumulate_kernel_vec_horner_7_avx2(float* out, float x, float w_re, float w_im) {
    float c0d[] = {3.9948351830487481E+03, 5.4715865608590771E+05, 5.0196413492771760E+06, 9.8206709220713247E+06, 5.0196413492771825E+06, 5.4715865608590783E+05, 3.9948351830642519E+03, 0.0000000000000000E+00};
    float c1d[] = {1.5290160332974696E+04, 8.7628248584320408E+05, 3.4421061790934438E+06, -2.6908159596373561E-10, -3.4421061790934461E+06, -8.7628248584320408E+05, -1.5290160332958067E+04, 0.0000000000000000E+00};
    float c2d[] = {2.4458227486779251E+04, 5.3904618484139396E+05, 2.4315566181017534E+05, -1.6133959371974322E+06, 2.4315566181017453E+05, 5.3904618484139396E+05, 2.4458227486795113E+04, 0.0000000000000000E+00};
    float c3d[] = {2.1166189345881645E+04, 1.3382732160223130E+05, -3.3113450969689694E+05, 6.9013724510092140E-10, 3.3113450969689724E+05, -1.3382732160223136E+05, -2.1166189345866893E+04, 0.0000000000000000E+00};
    float c4d[] = {1.0542795672344864E+04, -7.0739172265098678E+03, -6.5563293056049893E+04, 1.2429734005960064E+05, -6.5563293056049602E+04, -7.0739172265098332E+03, 1.0542795672361213E+04, 0.0000000000000000E+00};
    float c5d[] = {2.7903491906228419E+03, -1.0975382873973093E+04, 1.3656979541144799E+04, 7.7346408577822045E-10, -1.3656979541143772E+04, 1.0975382873973256E+04, -2.7903491906078298E+03, 0.0000000000000000E+00};
    float c6d[] = {1.6069721418053300E+02, -1.5518707872251393E+03, 4.3634273936642621E+03, -5.9891976420595174E+03, 4.3634273936642730E+03, -1.5518707872251064E+03, 1.6069721419533221E+02, 0.0000000000000000E+00};
    float c7d[] = {-1.2289277373867256E+02, 2.8583630927743314E+02, -2.8318194617327981E+02, 6.9043515551118249E-10, 2.8318194617392436E+02, -2.8583630927760140E+02, 1.2289277375319763E+02, 0.0000000000000000E+00};
    float c8d[] = {-3.2270164914249058E+01, 9.1892112257581346E+01, -1.6710678096334209E+02, 2.0317049305432383E+02, -1.6710678096383771E+02, 9.1892112257416159E+01, -3.2270164900224913E+01, 0.0000000000000000E+00};
    float c9d[] = {-1.4761409685186277E-01, -9.1862771280377487E-01, 1.2845147741777752E+00, 5.6547359492808854E-10, -1.2845147728310689E+00, 9.1862771293147971E-01, 1.4761410890866353E-01, 0.0000000000000000E+00};

    __m256 c0 = _mm256_loadu_ps(c0d);
    __m256 c1 = _mm256_loadu_ps(c1d);
    __m256 c2 = _mm256_loadu_ps(c2d);
    __m256 c3 = _mm256_loadu_ps(c3d);
    __m256 c4 = _mm256_loadu_ps(c4d);
    __m256 c5 = _mm256_loadu_ps(c5d);
    __m256 c6 = _mm256_loadu_ps(c6d);
    __m256 c7 = _mm256_loadu_ps(c7d);
    __m256 c8 = _mm256_loadu_ps(c8d);
    __m256 c9  = _mm256_loadu_ps(c9d);

    const int w = 7;

    __m256 z = _mm256_set1_ps(2 * x + w - 1.0);

    __m256 t0 = _mm256_fmadd_ps(z, c9, c8);
    __m256 t1 = _mm256_fmadd_ps(z, t0, c7);
    __m256 t2 = _mm256_fmadd_ps(z, t1, c6);
    __m256 t3 = _mm256_fmadd_ps(z, t2, c5);
    __m256 t4 = _mm256_fmadd_ps(z, t3, c4);
    __m256 t5 = _mm256_fmadd_ps(z, t4, c3);
    __m256 t6 = _mm256_fmadd_ps(z, t5, c2);
    __m256 t7 = _mm256_fmadd_ps(z, t6, c1);
    __m256 k = _mm256_fmadd_ps(z, t7, c0);

    __m256 w_re_v = _mm256_set1_ps(w_re);
    __m256 w_im_v = _mm256_set1_ps(w_im);

    __m256 k_re = _mm256_mul_ps(k, w_re_v);
    __m256 k_im = _mm256_mul_ps(k, w_im_v);

    __m256 lo = _mm256_unpacklo_ps(k_re, k_im);
    __m256 hi = _mm256_unpackhi_ps(k_re, k_im);

    __m256 out_lo = _mm256_loadu_ps(out);
    __m256 out_hi = _mm256_loadu_ps(out + 8);

    out_lo = _mm256_add_ps(out_lo, lo);
    out_hi = _mm256_add_ps(out_hi, hi);

    _mm256_storeu_ps(out, out_lo);
    _mm256_storeu_ps(out + 8, out_hi);
}


std::vector<float> generate_random_data(int n, int seed) {
    typedef r123::Philox2x32 RNG;
    RNG rng;

    RNG::ctr_type ctr = {{}};
    RNG::ukey_type key = {{}};
    key[0] = seed;

    std::vector<float> result(n);
    float scale = 0.8 * n;

    for(int i = 0; i < n; i++) {
        ctr[0] = i;
        auto r = rng(ctr, key);
        result[i] = r123::u01<float>(r[0]) * scale + 0.1 * n;
    }

    return result;
}

void benchmark_spread_subproblem_1d(benchmark::State& state) {
    int num_points = state.range(0);

    auto positions = generate_random_data(num_points, 0);
    auto strengths = generate_random_data(num_points * 2, 1);

    spread_opts opts;
    setup_spreader(opts, 1e-6f, 2.0, 1, 0, 1, 1);

    for(auto _ : state) {
        benchmark::ClobberMemory();

        std::vector<float> result(num_points * 2);
        spread_subproblem_1d(0, result.size() / 2, result.data(), positions.size(), positions.data(), strengths.data(), opts);
        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * num_points);
}

template<typename Fn>
void eval_kernel_impl(benchmark::State& state, int w, Fn const& accumulate) {
    const int unroll = 64;

    int out_size = (w + 3) / 4 * 4;

    std::vector<float> result(2 * out_size);

    const auto data = generate_random_data(10000, 0);
    const auto weights = generate_random_data(2 * 10000, 1);

    size_t ctr = 0;

    for(auto _ : state) {
        std::fill(result.begin(), result.end(), 0.0f);

        benchmark::ClobberMemory();

        for(int j = 0; j < unroll; ++j) {
            accumulate(result.data(), data[ctr], weights[2 * ctr], weights[2 * ctr + 1]);
            ctr += 1;
        }

        benchmark::ClobberMemory();
        benchmark::DoNotOptimize(result.data());

        ctr += unroll;
        if (ctr >= data.size() - unroll) {
            ctr = 0;
        }
    }

    state.SetItemsProcessed(state.iterations() * unroll);
}

void benchmark_eval_kernel(benchmark::State& state) {
    const int unroll = 64;

    int w = state.range(0);
    int out_size = (w + 3) / 4 * 4;

    std::vector<float> result(2 * out_size);
    std::vector<float> buffer(out_size);

    eval_kernel_impl(state, state.range(0), [&](float* out, const float x, const float w_re, const float w_im) {
        eval_kernel_vec_Horner(buffer.data(), x, w);

        for(int j = 0; j < out_size; ++j) {
            out[2 * j] += w_re * buffer[j];
            out[2 * j + 1] += w_im * buffer[j];
        }
    });
}

void benchmark_eval_kernel_7(benchmark::State& state) {
    const int unroll = 64;
    const int w = 7;

    int out_size = (w + 3) / 4 * 4;

    std::vector<float> buffer(2 * out_size);

    eval_kernel_impl(state, w, [&](float* out, const float x, const float w_re, const float w_im) {
        accumulate_kernel_vec_horner<w>(out, x, w_re, w_im);
    });
}

void benchmark_eval_kernel7_avx2(benchmark::State& state) {
    const int unroll = 64;
    const int w = 7;

    int out_size = (w + 3) / 4 * 4;

    std::vector<float> buffer(2 * out_size);

    eval_kernel_impl(state, w, [&](float* out, const float x, const float w_re, const float w_im) {
        accumulate_kernel_vec_horner_7_avx2(out, x, w_re, w_im);
    });
}

}

BENCHMARK(benchmark_spread_subproblem_1d)->RangeMultiplier(4)->Range(128, 1<<14)->Unit(benchmark::kMicrosecond);
BENCHMARK(benchmark_eval_kernel)->Arg(4)->Arg(6)->Arg(7)->Arg(8);
BENCHMARK(benchmark_eval_kernel_7);
BENCHMARK(benchmark_eval_kernel7_avx2);

