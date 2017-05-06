#include "cnufftspread_advanced.h"
#include "cnufftspread.h"
#include <vector>
#include <set>
#include <algorithm>

namespace Advanced {
std::vector<BIGINT> get_bin_sort_indices(BIGINT M,FLT *kx, FLT *ky, FLT *kz,double bin_size_x,double bin_size_y,double bin_size_z);

void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,BIGINT &size2,BIGINT &size3,BIGINT M0,FLT* kx0,FLT* ky0,FLT* kz0,int nspread);
double compute_min_par(BIGINT M,FLT *X);
double compute_max_par(BIGINT M,FLT *X);

void cnufftspread_type1_subproblem(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
         BIGINT M, FLT *kx, FLT *ky, FLT *kz,
         FLT *data_nonuniform, spread_opts opts);

void to_pi_range(BIGINT M, FLT *X, BIGINT N);
void from_pi_range(BIGINT M, FLT *X, BIGINT N);
}

int cnufftspread_advanced(BIGINT N1, BIGINT N2, BIGINT N3, FLT* data_uniform, BIGINT M, FLT* kx, FLT* ky, FLT* kz, FLT* data_nonuniform, spread_opts opts, int num_threads)
{
    BIGINT NNN=N1*N2*N3*2;
    for (BIGINT i=0; i<NNN; i++) {
        data_uniform[i]=0;
    }

    if (opts.pirange) {
        Advanced::from_pi_range(M,kx,N1);
        Advanced::from_pi_range(M,ky,N2);
        Advanced::from_pi_range(M,kz,N3);
    }

    std::vector<BIGINT> sort_indices(M);
    //sort_indices=Advanced::get_bin_sort_indices(M,kx,ky,kz,0,opts.nspread*2,opts.nspread*2);
    printf("Get bin sort indices...\n");
    sort_indices=Advanced::get_bin_sort_indices(M,kx,ky,kz,3,3,3);

    BIGINT max_points_per_subproblem=1e6;
    int num_subproblems=num_threads*4;
    if (num_subproblems*max_points_per_subproblem<M)
        num_subproblems=M/max_points_per_subproblem;
    BIGINT subproblem_size=(M+num_subproblems-1)/num_subproblems;

    printf ("Using %d subproblems of size %ld (M=%ld)...\n",num_subproblems,subproblem_size,M);

#pragma omp parallel for
    for (int isub=0; isub<num_subproblems; isub++) {
        BIGINT M0=subproblem_size;
        if (isub*subproblem_size+M0>M) {
            M0=M-isub*subproblem_size;
            if (M0<0) M0=0;
        }
        if (M0>0) {
            BIGINT j_offset=isub*subproblem_size;
            FLT* kx0=(FLT*)malloc(sizeof(FLT)*M0);
            FLT* ky0=(FLT*)malloc(sizeof(FLT)*M0);
            FLT* kz0=(FLT*)malloc(sizeof(FLT)*M0);
            FLT* dd0=(FLT*)malloc(sizeof(FLT)*M0*2);
            for (BIGINT j=0; j<M0; j++) {
                BIGINT kk=sort_indices[j_offset+j];
                kx0[j]=kx[kk];
                ky0[j]=ky[kk];
                kz0[j]=kz[kk];
                dd0[j*2]=data_nonuniform[kk*2];
                dd0[j*2+1]=data_nonuniform[kk*2+1];
            }
            BIGINT offset1,offset2,offset3,size1,size2,size3;
            Advanced::get_subgrid(offset1,offset2,offset3,size1,size2,size3,M0,kx0,ky0,kz0,opts.nspread);
            //printf("subgrid: %ld,%ld,%ld,%ld,%ld,%ld\n",offset1,offset2,offset3,size1,size2,size3);
            for (BIGINT j=0; j<M0; j++) {
                kx0[j]-=offset1;
                ky0[j]-=offset2;;
                kz0[j]-=offset3;;
            }
            FLT* data_uniform_0=(FLT*)malloc(sizeof(FLT)*size1*size2*size3*2);
            Advanced::cnufftspread_type1_subproblem(size1,size2,size3,data_uniform_0,M0,kx0,ky0,kz0,dd0,opts);
            free(kx0);
            free(ky0);
            free(kz0);
            free(dd0);

#pragma omp critical
            {
                BIGINT output_inds1[size1];
                BIGINT output_inds2[size2];
                BIGINT output_inds3[size3];
                for (BIGINT a=0; a<size1; a++) {
                    BIGINT ind0=offset1+a;
                    while (ind0<0) ind0+=N1;
                    while (ind0>=N1) ind0-=N1;
                    output_inds1[a]=ind0;
                }
                for (BIGINT a=0; a<size2; a++) {
                    BIGINT ind0=offset2+a;
                    while (ind0<0) ind0+=N2;
                    while (ind0>=N2) ind0-=N2;
                    output_inds2[a]=ind0;
                }
                for (BIGINT a=0; a<size3; a++) {
                    BIGINT ind0=offset3+a;
                    while (ind0<0) ind0+=N3;
                    while (ind0>=N3) ind0-=N3;
                    output_inds3[a]=ind0;
                }

                BIGINT input_index=0;
                for (BIGINT i3=0; i3<size3; i3++) {
                    BIGINT output_index3=output_inds3[i3]*N1*N2;
                    for (BIGINT i2=0; i2<size2; i2++) {
                        BIGINT output_index2=output_index3+output_inds2[i2]*N1;
                        for (BIGINT i1=0; i1<size1; i1++) {
                            BIGINT output_index1=output_index2+output_inds1[i1];
                            data_uniform[output_index1*2]+=data_uniform_0[input_index*2];
                            data_uniform[output_index1*2+1]+=data_uniform_0[input_index*2+1];
                            input_index++;
                        }
                    }
                }
            }

            free(data_uniform_0);
        }
    }

    if (opts.pirange) {
        Advanced::to_pi_range(M,kx,N1);
        Advanced::to_pi_range(M,ky,N2);
        Advanced::to_pi_range(M,kz,N3);
    }

    return 0;
}

namespace Advanced {

void to_pi_range(BIGINT M, FLT *X, BIGINT N)
{
    for (BIGINT i=0; i<M; i++) {
        X[i]=(X[i]/N-0.5)*2*PI;
    }
}

void from_pi_range(BIGINT M, FLT *X, BIGINT N)
{
    for (BIGINT i=0; i<M; i++) {
        X[i]=(X[i]/(2*PI)+0.5)*N;
    }
}

/*
std::vector<BIGINT> get_sort_indices_0(const std::vector<double> &X) {
    std::vector<BIGINT> result(X.size());
    for (BIGINT i = 0; i < (BIGINT)X.size(); i++)
        result[i]=i;
    std::stable_sort(result.begin(), result.end(),
        [&X](BIGINT i1, BIGINT i2) { return X[i1] < X[i2]; });
    return result;
}
*/

std::vector<BIGINT> get_bin_sort_indices(BIGINT M,FLT *kx, FLT *ky, FLT *kz,double bin_size_x,double bin_size_y,double bin_size_z) {

    FLT kx_min=compute_min_par(M,kx);
    FLT kx_max=compute_max_par(M,kx);
    FLT ky_min=compute_min_par(M,ky);
    FLT ky_max=compute_max_par(M,ky);
    FLT kz_min=compute_min_par(M,kz);
    FLT kz_max=compute_max_par(M,kz);

    BIGINT nbins1=(kx_max-kx_min)/bin_size_x+1;
    BIGINT nbins2=(ky_max-ky_min)/bin_size_y+1;
    BIGINT nbins3=(kz_max-kz_min)/bin_size_z+1;

    std::vector<BIGINT> bins(M);
#pragma omp parallel for
    for (BIGINT i=0; i<M; i++) {
        BIGINT i1=(kx[i]-kx_min)/bin_size_x;
        BIGINT i2=(ky[i]-ky_min)/bin_size_y;
        BIGINT i3=(kz[i]-kz_min)/bin_size_z;
        bins[i]=i1+nbins1*i2+nbins1*nbins2*i3;
    }

    std::vector<BIGINT> counts(nbins1*nbins2*nbins3,0);
    //how to parallelize this?
    for (BIGINT i=0; i<M; i++) {
        counts[bins[i]]++;
    }
    std::vector<BIGINT> offsets(nbins1*nbins2*nbins3);
    offsets[0]=0;
    //how to parallelize a cumulative sum?
    for (BIGINT i=1; i<nbins1*nbins2*nbins3; i++) {
        offsets[i]=offsets[i-1]+counts[i-1];
    }

    std::vector<BIGINT> inv(M);
    //how to parallelize this?
    for (BIGINT i=0; i<M; i++) {
        BIGINT offset=offsets[bins[i]];
        offsets[bins[i]]++;
        inv[i]=offset;
    }

    std::vector<BIGINT> ret(M);
    //I think this is safe to parallelize, but afraid to do it just yet
    for (BIGINT i=0; i<M; i++) {
        ret[inv[i]]=i;
    }
    return ret;
}

double compute_min_par(BIGINT M,FLT *X) {
    if (M==0) return 0;

    double global_best=X[0];
#pragma omp parallel
    {
        double best=X[0];
#pragma omp for
        for (BIGINT i=0; i<M; i++) {
            if (X[i]<best) best=X[i];
        }
#pragma omp critical
        {
            if (best<global_best) global_best=best;
        }
    }
    return global_best;
}

double compute_max_par(BIGINT M,FLT *X) {
    if (M==0) return 0;

    double global_best=X[0];
#pragma omp parallel
    {
        double best=X[0];
#pragma omp for
        for (BIGINT i=0; i<M; i++) {
            if (X[i]>best) best=X[i];
        }
#pragma omp critical
        {
            if (best>global_best) global_best=best;
        }
    }
    return global_best;
}
void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,BIGINT &size2,BIGINT &size3,BIGINT M,FLT* kx,FLT* ky,FLT* kz,int nspread) {
    double min_kx=compute_min_par(M,kx);
    double max_kx=compute_max_par(M,kx);
    double min_ky=compute_min_par(M,ky);
    double max_ky=compute_max_par(M,ky);
    double min_kz=compute_min_par(M,kz);
    double max_kz=compute_max_par(M,kz);

    min_kx=min_kx;
    max_kx=max_kx;
    min_ky=min_ky;
    max_ky=max_ky;
    min_kz=min_kz;
    max_kz=max_kz;

    int min_radius=(nspread+1)/2+2; //there is a reason for using the +2, and it has to do with the minimum grid size relative to nspread that the system will tolerate
    BIGINT a1=floor(min_kx-min_radius);
    BIGINT a2=ceil(max_kx+min_radius);
    BIGINT b1=floor(min_ky-min_radius);
    BIGINT b2=ceil(max_ky+min_radius);
    BIGINT c1=floor(min_kz-min_radius);
    BIGINT c2=ceil(max_kz+min_radius);

    offset1=a1;
    size1=a2-a1+1;
    offset2=b1;
    size2=b2-b1+1;
    offset3=c1;
    size3=c2-c1+1;
}

void compute_kernel_values(FLT x1, FLT x2, FLT x3,
    const spread_opts& opts,FLT* ker)
{
    int ns=opts.nspread;
    FLT v1[ns], v2[ns], v3[ns];
    for (int i = 0; i <= ns; i++)
        v1[i] = evaluate_kernel(x1 + (FLT)i, opts);
    for (int i = 0; i <= ns; i++)
        v2[i] = evaluate_kernel(x2 + (FLT)i, opts);
    for (int i = 0; i <= ns; i++)
        v3[i] = evaluate_kernel(x3 + (FLT)i, opts);
    int aa = 0; // pointer for writing to output ker array
    for (int k = 0; k < ns; k++) {
        FLT val3 = v3[k];
        for (int j = 0; j < ns; j++) {
            FLT val2 = val3 * v2[j];
            for (int i = 0; i < ns; i++)
                ker[aa++] = val2 * v1[i];
        }
    }
}

void cnufftspread_type1_subproblem(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
         BIGINT M, FLT *kx, FLT *ky, FLT *kz,
         FLT *data_nonuniform, spread_opts opts) {

    int ns = opts.nspread;
    FLT ns2 = (FLT)ns / 2;
    for (BIGINT i = 0; i < 2 * N1 * N2 * N3; i++)
        data_uniform[i] = 0.0;

    FLT kernel_values[ns*ns*ns];

    for (BIGINT i=0; i<M; i++) {

        FLT re0 = data_nonuniform[2 * i];
        FLT im0 = data_nonuniform[2 * i + 1];

        BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
        BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
        BIGINT i3 = (BIGINT)std::ceil(kz[i] - ns2);
        FLT x1 = (FLT)i1 - kx[i];
        FLT x2 = (FLT)i2 - ky[i];
        FLT x3 = (FLT)i3 - kz[i];

        compute_kernel_values(x1, x2, x3, opts, kernel_values);

        int aa = 0;
        for (int dz = 0; dz < ns; dz++) {
            BIGINT tmp=(i3 + dz) * N1 * N2;
            for (int dy = 0; dy < ns; dy++) {
                BIGINT jjj = tmp + (i2 + dy) * N1 + i1;
                for (int dx = 0; dx < ns; dx++) {
                    FLT kern0 = kernel_values[aa];
                    data_uniform[jjj * 2] += re0 * kern0; // accumulate complex value to grid
                    data_uniform[jjj * 2 + 1] += im0 * kern0;
                    jjj++;
                    aa++;
                }
            }
        }
    }
}

}

