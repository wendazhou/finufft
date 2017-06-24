#include "finufft.h"
#include <QTime>
#include <vector>
#include "clparams.h"

struct Test_opts {
    int ndims=3;
    int nufft_type=1;
    int nj=8e6;
    int N=200;
    double eps=1e-6;
    int num_threads=0;
    bool compare_with_brute_force=false;
    bool report_timing=false;
    int timing_flags=0;
};

void test_finufft(Test_opts ooo);
void finufft_type1_brute_force(INT nj,FLT* xj,FLT* yj,FLT* zj,CPX* cj,int iflag,FLT eps,INT ms,INT mt,INT mu,CPX* fk, nufft_opts opts);
double compute_l2_error(INT n,CPX* V,CPX* Vref);

int main(int argc,char *argv[]) {

    CLParams CLP(argc,argv);

    int max_threads=omp_get_max_threads();

    Test_opts oo;
    oo.eps=1e-6;
    oo.nufft_type=1;

    QuteString arg1=CLP.unnamed_parameters.value(0,"accuracy");
    int ndims=CLP.named_parameters.value("ndims","3").toInt();
    printf("ndims=%d\n",ndims);
    return 0;

    ///////////////////////////////////////////////////////////////////////////////
    /// 1-dimensional accuracy tests
    printf("\n******************* 1-dimensional accuracy tests *******************\n");
    oo.ndims=1;
    oo.compare_with_brute_force=true;
    oo.report_timing=false;
    oo.num_threads=max_threads;
    oo.timing_flags=0;
    {
        oo.nj=32;
        oo.N=32;
        test_finufft(oo);
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// 1-dimensional speed tests
    printf("\n******************* 1-dimensional speed tests *******************\n");
    oo.ndims=1;
    oo.compare_with_brute_force=false;
    oo.report_timing=true;
    {
        oo.nj=8e6;
        oo.N=200;
        oo.num_threads=max_threads;
        oo.timing_flags=0;
        test_finufft(oo);
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// 3-dimensional accuracy tests
    printf("\n******************* 3-dimensional accuracy tests *******************\n");
    oo.ndims=3;
    oo.compare_with_brute_force=true;
    oo.report_timing=false;
    oo.num_threads=max_threads;
    oo.timing_flags=0;
    {
        oo.nj=32;
        oo.N=32;
        test_finufft(oo);
    }

    ///////////////////////////////////////////////////////////////////////////////
    /// 3-dimensional speed tests
    printf("\n******************* 3-dimensional speed tests\n");
    oo.ndims=3;
    oo.compare_with_brute_force=false;
    oo.report_timing=true;
    {
        oo.nj=8e6;
        oo.N=200;
        oo.num_threads=max_threads;
        oo.timing_flags=0;
        test_finufft(oo);
    }
    {
        oo.nj=8e6;
        oo.N=200;
        oo.num_threads=max_threads;
        oo.timing_flags=TF_OMIT_FFT;
        test_finufft(oo);
    }



    return 0;
}

double randu() {
    INT maxint=1e6;
    return (rand()%maxint)*1.0/maxint;
}

void test_finufft(Test_opts ooo) {
    if (ooo.num_threads==0)
        ooo.num_threads=omp_get_max_threads();
    omp_set_num_threads(ooo.num_threads);

    nufft_opts opts;
    opts.debug=0;
    opts.fftw=FFTW_ESTIMATE;
    opts.R=2;
    opts.spread_debug=0;
    opts.spread_sort=1;
    opts.timing_flags=ooo.timing_flags;
    int nj=ooo.nj;
    std::vector<FLT> xj(nj),yj(nj),zj(nj);
    std::vector<CPX> cj(nj);
    for (INT i=0; i<nj; i++) {
        xj[i]=-PI+randu()*2*PI;
        if (ooo.ndims>=2)
            yj[i]=-PI+randu()*2*PI;
        else
            yj[i]=0;
        if (ooo.ndims>=3)
            zj[i]=-PI+randu()*2*PI;
        else
            zj[i]=0;
        cj[i]=CPX(randu()*2-1,randu()*2-1);
    }
    int iflag=1;
    int ms=1,mt=1,mu=1;
    ms=ooo.N;
    if (ooo.ndims>=2)
        mt=ooo.N;
    if (ooo.ndims>=3)
        mu=ooo.N;
    std::vector<CPX> fk(ms*mt*mu);

    printf("nj=%d, (ms,mt,mu)=(%d,%d,%d), %d threads\n",nj,ms,mt,mu,ooo.num_threads);

    QTime timer; timer.start();
    if (ooo.ndims==1) {
        finufft1d1(nj,xj.data(),cj.data(),iflag,ooo.eps,ms,fk.data(),opts);
    }
    else if (ooo.ndims==2) {
        finufft2d1(nj,xj.data(),yj.data(),cj.data(),iflag,ooo.eps,ms,mt,fk.data(),opts);
    }
    else if (ooo.ndims==3) {
        finufft3d1(nj,xj.data(),yj.data(),zj.data(),cj.data(),iflag,ooo.eps,ms,mt,mu,fk.data(),opts);
    }
    double elapsed_sec=timer.elapsed()*1.0/1000;

    if (ooo.compare_with_brute_force) {
        std::vector<CPX> fk_brute_force(ms*mt*mu);
        finufft_type1_brute_force(nj,xj.data(),yj.data(),zj.data(),cj.data(),iflag,ooo.eps,ms,mt,mu,fk_brute_force.data(),opts);

        double err=compute_l2_error(ms*mt*mu,fk.data(),fk_brute_force.data());
        printf("###### Should be small: [%g]\n",err);
    }
    if (ooo.report_timing) {
        printf("###### Elapsed time: %.2f sec (%g million n.u. pts/sec)\n",elapsed_sec,nj/elapsed_sec/1e6);
    }
}

/*
         nj-1
fk(k1) = SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2
         j=0
*/
void finufft_type1_brute_force(INT nj,FLT* xj,FLT* yj,FLT* zj,CPX* cj,int iflag,FLT eps,INT ms,INT mt,INT mu,CPX* fk, nufft_opts opts) {
    INT sign=1;
    if (iflag<0) sign=-1;
    for (INT i1=0; i1<ms; i1++) {
        INT k1=i1-ms/2;
        for (INT i2=0; i2<mt; i2++) {
            INT k2=i2-mt/2;
            for (INT i3=0; i3<mu; i3++) {
                INT k3=i3-mu/2;
                CPX val=0;
                for (INT j=0; j<nj; j++) {
                    val=val+cj[j]*exp(CPX(0,sign)*(xj[j]*CPX(k1,0)+yj[j]*CPX(k2,0)+zj[j]*CPX(k3,0)));
                }
                fk[i1+ms*i2+ms*mt*i3]=val;
            }
        }
    }
}

double compute_l2_error(INT n,CPX* V,CPX* Vref) {
    double sumsqr_diff=0;
    double sumsqr_ref=0;
    for (INT i=0; i<n; i++) {
        CPX diff=V[i]-Vref[i];
        sumsqr_diff+=diff.real()*diff.real()+diff.imag()*diff.imag();
        sumsqr_ref+=Vref[i].real()*Vref[i].real()+Vref[i].imag()*Vref[i].imag();
    }
    if (sumsqr_ref==0) return 0;
    return sqrt(sumsqr_diff/sumsqr_ref);
}
