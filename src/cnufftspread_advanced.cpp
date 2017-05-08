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

void optimized_write_to_output_grid(BIGINT offset1,BIGINT offset2,BIGINT offset3,BIGINT size1,BIGINT size2,BIGINT size3,BIGINT N1,BIGINT N2,BIGINT N3,FLT* data_uniform_0,FLT* data_uniform);

struct Subgrid {
    Subgrid(BIGINT x1,BIGINT x2,BIGINT y1,BIGINT y2,BIGINT z1,BIGINT z2);
    Subgrid(const Subgrid &other);
    void operator=(const Subgrid &other);
    bool operator==(const Subgrid &other);
    bool intersects(const Subgrid &other); //N1,N2,N3 needed for wrapping
    BIGINT x1=0,x2=0;
    BIGINT y1=0,y2=0;
    BIGINT z1=0,z2=0;
private:
    void copy_from(const Subgrid &other);
};

class SpreadingLocker {
public:
    SpreadingLocker();
    void acquireLock(BIGINT x1,BIGINT x2,BIGINT y1,BIGINT y2,BIGINT z1,BIGINT z2);
    void releaseLock(BIGINT x1,BIGINT x2,BIGINT y1,BIGINT y2,BIGINT z1,BIGINT z2);
    double totalNumLockBlocks();

private:
    std::vector<Subgrid> m_locked_subgrids;
    double m_total_lock_blocks=0;
};

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

    Advanced::SpreadingLocker SL;

    std::vector<BIGINT> sort_indices(M);
    //sort_indices=Advanced::get_bin_sort_indices(M,kx,ky,kz,0,opts.nspread*2,opts.nspread*2);
    printf("Get bin sort indices...\n");
    sort_indices=Advanced::get_bin_sort_indices(M,kx,ky,kz,3,3,3);
    
    BIGINT max_points_per_subproblem=1000;
    int num_subproblems=num_threads*4;
    if (num_subproblems*max_points_per_subproblem<M)
        num_subproblems=M/max_points_per_subproblem;
    BIGINT subproblem_size=(M+num_subproblems-1)/num_subproblems;

    printf ("Using %d subproblems of size %ld (M=%ld)...\n",num_subproblems,subproblem_size,M);

#pragma omp parallel for
    for (int jsub=0; jsub<num_subproblems; jsub++) {
        int isub=(jsub*1)%num_subproblems; // I was thinking of doing the subproblems in a different order to avoid lock conflicts, but doesn't seem to make a difference
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

//#pragma omp critical
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

                SL.acquireLock(offset1,offset1+size1-1,offset2,offset2+size2-1,offset3,offset3+size3-1);
                {
                    BIGINT input_index=0;
                    for (BIGINT i3=0; i3<size3; i3++) {
                        BIGINT output_index3=output_inds3[i3]*N1*N2;
                        for (BIGINT i2=0; i2<size2; i2++) {
                            BIGINT output_index2=output_index3+output_inds2[i2]*N1;
                            for (BIGINT i1=0; i1<size1; i1++) {
                                BIGINT output_index=output_index2+output_inds1[i1];
                                data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                                data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                                input_index++;
                            }
                        }
                    }
                }
                SL.releaseLock(offset1,offset1+size1-1,offset2,offset2+size2-1,offset3,offset3+size3-1);
            }

            free(kx0);
            free(ky0);
            free(kz0);
            free(dd0);
            free(data_uniform_0);
        }
    }

    if (opts.pirange) {
        Advanced::to_pi_range(M,kx,N1);
        Advanced::to_pi_range(M,ky,N2);
        Advanced::to_pi_range(M,kz,N3);
    }

    printf("Total num lock blocks per thread: %g\n",SL.totalNumLockBlocks()/omp_get_max_threads());

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

    FLT kernel_values[ns*ns*ns];   // is this dynamic alloc?

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

SpreadingLocker::SpreadingLocker()
{

}

void SpreadingLocker::acquireLock(BIGINT x1, BIGINT x2, BIGINT y1, BIGINT y2, BIGINT z1, BIGINT z2)
{
    //printf("Acquiring lock %ld,%ld %ld,%ld %ld,%ld\n",x1,x2,y1,y2,z1,z2);
    Subgrid SG(x1,x2,y1,y2,z1,z2);
    while (1) {
        bool intersects_something=false;
        #pragma omp critical
        {
            for (int i=0; i<m_locked_subgrids.size(); i++) {
                if (m_locked_subgrids.at(i).intersects(SG)) {
                    intersects_something=true;
                }
            }
        }
        if (!intersects_something)
            break;
        else
            m_total_lock_blocks++;
    }
#pragma omp critical
    {
        m_locked_subgrids.push_back(SG);
        //printf("    Locked: %ld,%ld %ld,%ld %ld,%ld (%ld locked)\n",x1,x2,y1,y2,z1,z2,m_locked_subgrids.size());
    }
}

void SpreadingLocker::releaseLock(BIGINT x1, BIGINT x2, BIGINT y1, BIGINT y2, BIGINT z1, BIGINT z2)
{
    #pragma omp critical
    {
        Subgrid SG(x1,x2,y1,y2,z1,z2);
        bool found=false;
        for (int i=0; i<m_locked_subgrids.size(); i++) {
            if (m_locked_subgrids.at(i)==SG) {
                m_locked_subgrids.erase(m_locked_subgrids.begin()+i);
                found=true;
                break;
            }
        }
        if (!found) {
            printf("Warning: Unexpected problem in releaseLock... unable to find subgrid.\n");
        }
    }
    //printf("        Released: %ld,%ld %ld,%ld %ld,%ld\n",x1,x2,y1,y2,z1,z2);
}

double SpreadingLocker::totalNumLockBlocks()
{
    return m_total_lock_blocks;
}

Subgrid::Subgrid(BIGINT x1_in, BIGINT x2_in, BIGINT y1_in, BIGINT y2_in, BIGINT z1_in, BIGINT z2_in)
{
    x1=x1_in; x2=x2_in;
    y1=y1_in; y2=y2_in;
    z1=z1_in; z2=z2_in;
}

Subgrid::Subgrid(const Subgrid &other)
{
    copy_from(other);
}

void Subgrid::operator=(const Subgrid &other)
{
    copy_from(other);
}

bool Subgrid::operator==(const Subgrid &other)
{
    if ((other.x1!=x1)||(other.x2!=x2))
        return false;
    if ((other.y1!=y1)||(other.y2!=y2))
        return false;
    if ((other.z1!=z1)||(other.z2!=z2))
        return false;
    return true;
}

bool Subgrid::intersects(const Subgrid &other)
{
    //there are 6 ways they could not intersect
    if (other.x1>x2) return false;
    if (other.x2<x1) return false;
    if (other.y1>y2) return false;
    if (other.y2<y1) return false;
    if (other.z1>z2) return false;
    if (other.z2<z1) return false;
    return true;
}

void Subgrid::copy_from(const Subgrid &other)
{
    x1=other.x1;
    x2=other.x2;
    y1=other.y1;
    y2=other.y2;
    z1=other.z1;
    z2=other.z2;
}

/*
 The following was a big waste of time and effort -- causes crash and doesn't even speed it up

void optimized_write_to_output_grid(BIGINT d1_offset,BIGINT d2_offset,BIGINT d3_offset,BIGINT d1_size,BIGINT d2_size,BIGINT d3_size,BIGINT d1_N,BIGINT d2_N,BIGINT d3_N,FLT* data_uniform_0,FLT* data_uniform) {
    BIGINT d1_lower_src_A,d1_upper_src_A,d1_lower_dst_A,d1_upper_dst_A;
    BIGINT d1_lower_src_B,d1_upper_src_B,d1_lower_dst_B,d1_upper_dst_B;
    while (d1_offset<0) d1_offset+=d1_N;
    while (d1_offset>=d1_N) d1_offset-=d1_N;
    if (d1_offset+d1_size-1>=d1_N) {
        d1_lower_src_A=0;
        d1_upper_src_A=d1_N-1-d1_offset; //d1_size-1-(d1_N-1-d1_offset)=d1_size-d1_N+d1_offset>=0
        d1_lower_dst_A=d1_offset;
        d1_upper_dst_A=d1_N-1;
        d1_lower_src_B=d1_N-d1_offset;
        d1_upper_src_B=d1_size-1;
        d1_lower_dst_B=0;
        d1_upper_dst_B=d1_size-1-d1_N+d1_offset; //d1_N-1-(d1_size-1-d1_N+d1_offset)=2*d1_N-(d1_size+d1_offset)>=d1_N-d1_size>=0
    }
    else {
        d1_lower_src_A=0;
        d1_upper_src_A=d1_size-1;
        d1_lower_dst_A=d1_offset;
        d1_upper_dst_A=d1_offset+d1_size-1;
        d1_lower_src_B=-1;
        d1_upper_src_B=-1;
        d1_lower_dst_B=-1;
        d1_upper_dst_B=-1;
    }

    BIGINT d2_lower_src_A,d2_upper_src_A,d2_lower_dst_A,d2_upper_dst_A;
    BIGINT d2_lower_src_B,d2_upper_src_B,d2_lower_dst_B,d2_upper_dst_B;
    while (d2_offset<0) d2_offset+=d2_N;
    while (d2_offset>=d2_N) d2_offset-=d2_N;
    if (d2_offset+d2_size-1>=d2_N) {
        d2_lower_src_A=0;
        d2_upper_src_A=d2_N-1-d2_offset;
        d2_lower_dst_A=d2_offset;
        d2_upper_dst_A=d2_N-1;
        d2_lower_src_B=d2_N-d2_offset;
        d2_upper_src_B=d2_size-1;
        d2_lower_dst_B=0;
        d2_upper_dst_B=d2_size-1-d2_N+d2_offset;
    }
    else {
        d2_lower_src_A=0;
        d2_upper_src_A=d2_size-1;
        d2_lower_dst_A=d2_offset;
        d2_upper_dst_A=d2_offset+d2_size-1;
        d2_lower_src_B=-1;
        d2_upper_src_B=-1;
        d2_lower_dst_B=-1;
        d2_upper_dst_B=-1;
    }

    BIGINT d3_lower_src_A,d3_upper_src_A,d3_lower_dst_A,d3_upper_dst_A;
    BIGINT d3_lower_src_B,d3_upper_src_B,d3_lower_dst_B,d3_upper_dst_B;
    while (d3_offset<0) d3_offset+=d3_N;
    while (d3_offset>=d3_N) d3_offset-=d3_N;
    if (d3_offset+d3_size-1>=d3_N) {
        d3_lower_src_A=0;
        d3_upper_src_A=d3_N-1-d3_offset;
        d3_lower_dst_A=d3_offset;
        d3_upper_dst_A=d3_N-1;
        d3_lower_src_B=d3_N-d3_offset;
        d3_upper_src_B=d3_size-1;
        d3_lower_dst_B=0;
        d3_upper_dst_B=d3_size-1-d3_N+d3_offset;
    }
    else {
        d3_lower_src_A=0;
        d3_upper_src_A=d3_size-1;
        d3_lower_dst_A=d3_offset;
        d3_upper_dst_A=d3_offset+d3_size-1;
        d3_lower_src_B=-1;
        d3_upper_src_B=-1;
        d3_lower_dst_B=-1;
        d3_upper_dst_B=-1;
    }

    // AAA
    if ((d1_lower_src_A>=0)&&(d2_lower_src_A>=0)&&(d3_lower_src_A>=0)) {
        BIGINT input_index=0;
        BIGINT output_index=0;
        BIGINT d1_cc=-d1_lower_src_A+d1_lower_dst_A;
        BIGINT d2_cc=-d2_lower_src_A+d2_lower_dst_A;
        BIGINT d3_cc=-d3_lower_src_A+d3_lower_dst_A;
        for (BIGINT i3=d3_lower_src_A; i3<=d3_upper_src_A; i3++) {
            BIGINT in_tmp00=d1_lower_src_A+d1_size*d2_size*i3;
            BIGINT out_tmp00=(d1_lower_src_A+d1_cc)+d1_N*d2_N*(i3+d3_cc);
            for (BIGINT i2=d2_lower_src_A; i2<=d2_upper_src_A; i2++) {
                BIGINT input_index=in_tmp00+d1_size*i2;
                BIGINT output_index=out_tmp00+d1_N*(i2+d2_cc);
                for (BIGINT i1=d1_lower_src_A; i1<=d1_upper_src_A; i1++) {
                    //BIGINT input_index=i1+d1_size*i2+d1_size*d2_size*i3; //for reference
                    //BIGINT output_index=(i1+d1_cc)+d1_N*(i2+d2_cc)+d1_N*d2_N*(i3+d3_cc); //for reference
                    data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                    data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                    input_index++;
                    output_index++;
                }
            }
        }
    }

    // AAB
    if ((d1_lower_src_A>=0)&&(d2_lower_src_A>=0)&&(d3_lower_src_B>=0)) {
        BIGINT input_index=0;
        BIGINT output_index=0;
        BIGINT d1_cc=-d1_lower_src_A+d1_lower_dst_A;
        BIGINT d2_cc=-d2_lower_src_A+d2_lower_dst_A;
        BIGINT d3_cc=-d3_lower_src_B+d3_lower_dst_B;
        for (BIGINT i3=d3_lower_src_B; i3<=d3_upper_src_B; i3++) {
            BIGINT in_tmp00=d1_lower_src_A+d1_size*d2_size*i3;
            BIGINT out_tmp00=(d1_lower_src_A+d1_cc)+d1_N*d2_N*(i3+d3_cc);
            for (BIGINT i2=d2_lower_src_A; i2<=d2_upper_src_A; i2++) {
                BIGINT input_index=in_tmp00+d1_size*i2;
                BIGINT output_index=out_tmp00+d1_N*(i2+d2_cc);
                for (BIGINT i1=d1_lower_src_A; i1<=d1_upper_src_A; i1++) {
                    //BIGINT input_index=i1+d1_size*i2+d1_size*d2_size*i3; //for reference
                    //BIGINT output_index=(i1+d1_cc)+d1_N*(i2+d2_cc)+d1_N*d2_N*(i3+d3_cc); //for reference
                    data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                    data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                    input_index++;
                    output_index++;
                }
            }
        }
    }

    // ABA
    if ((d1_lower_src_A>=0)&&(d2_lower_src_B>=0)&&(d3_lower_src_A>=0)) {
        BIGINT input_index=0;
        BIGINT output_index=0;
        BIGINT d1_cc=-d1_lower_src_A+d1_lower_dst_A;
        BIGINT d2_cc=-d2_lower_src_B+d2_lower_dst_B;
        BIGINT d3_cc=-d3_lower_src_A+d3_lower_dst_A;
        for (BIGINT i3=d3_lower_src_A; i3<=d3_upper_src_A; i3++) {
            BIGINT in_tmp00=d1_lower_src_A+d1_size*d2_size*i3;
            BIGINT out_tmp00=(d1_lower_src_A+d1_cc)+d1_N*d2_N*(i3+d3_cc);
            for (BIGINT i2=d2_lower_src_B; i2<=d2_upper_src_B; i2++) {
                BIGINT input_index=in_tmp00+d1_size*i2;
                BIGINT output_index=out_tmp00+d1_N*(i2+d2_cc);
                for (BIGINT i1=d1_lower_src_A; i1<=d1_upper_src_A; i1++) {
                    //BIGINT input_index=i1+d1_size*i2+d1_size*d2_size*i3; //for reference
                    //BIGINT output_index=(i1+d1_cc)+d1_N*(i2+d2_cc)+d1_N*d2_N*(i3+d3_cc); //for reference
                    data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                    data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                    input_index++;
                    output_index++;
                }
            }
        }
    }

    // ABB
    if ((d1_lower_src_A>=0)&&(d2_lower_src_B>=0)&&(d3_lower_src_B>=0)) {
        BIGINT input_index=0;
        BIGINT output_index=0;
        BIGINT d1_cc=-d1_lower_src_A+d1_lower_dst_A;
        BIGINT d2_cc=-d2_lower_src_B+d2_lower_dst_B;
        BIGINT d3_cc=-d3_lower_src_B+d3_lower_dst_B;
        for (BIGINT i3=d3_lower_src_B; i3<=d3_upper_src_B; i3++) {
            BIGINT in_tmp00=d1_lower_src_A+d1_size*d2_size*i3;
            BIGINT out_tmp00=(d1_lower_src_A+d1_cc)+d1_N*d2_N*(i3+d3_cc);
            for (BIGINT i2=d2_lower_src_B; i2<=d2_upper_src_B; i2++) {
                BIGINT input_index=in_tmp00+d1_size*i2;
                BIGINT output_index=out_tmp00+d1_N*(i2+d2_cc);
                for (BIGINT i1=d1_lower_src_A; i1<=d1_upper_src_A; i1++) {
                    //BIGINT input_index=i1+d1_size*i2+d1_size*d2_size*i3; //for reference
                    //BIGINT output_index=(i1+d1_cc)+d1_N*(i2+d2_cc)+d1_N*d2_N*(i3+d3_cc); //for reference
                    data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                    data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                    input_index++;
                    output_index++;
                }
            }
        }
    }

    // BAA
    if ((d1_lower_src_B>=0)&&(d2_lower_src_A>=0)&&(d3_lower_src_A>=0)) {
        BIGINT input_index=0;
        BIGINT output_index=0;
        BIGINT d1_cc=-d1_lower_src_B+d1_lower_dst_B;
        BIGINT d2_cc=-d2_lower_src_A+d2_lower_dst_A;
        BIGINT d3_cc=-d3_lower_src_A+d3_lower_dst_A;
        for (BIGINT i3=d3_lower_src_A; i3<=d3_upper_src_A; i3++) {
            BIGINT in_tmp00=d1_lower_src_B+d1_size*d2_size*i3;
            BIGINT out_tmp00=(d1_lower_src_B+d1_cc)+d1_N*d2_N*(i3+d3_cc);
            for (BIGINT i2=d2_lower_src_A; i2<=d2_upper_src_A; i2++) {
                BIGINT input_index=in_tmp00+d1_size*i2;
                BIGINT output_index=out_tmp00+d1_N*(i2+d2_cc);
                for (BIGINT i1=d1_lower_src_B; i1<=d1_upper_src_B; i1++) {
                    //BIGINT input_index=i1+d1_size*i2+d1_size*d2_size*i3; //for reference
                    //BIGINT output_index=(i1+d1_cc)+d1_N*(i2+d2_cc)+d1_N*d2_N*(i3+d3_cc); //for reference
                    data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                    data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                    input_index++;
                    output_index++;
                }
            }
        }
    }

    // BAB
    if ((d1_lower_src_B>=0)&&(d2_lower_src_A>=0)&&(d3_lower_src_B>=0)) {
        BIGINT input_index=0;
        BIGINT output_index=0;
        BIGINT d1_cc=-d1_lower_src_B+d1_lower_dst_B;
        BIGINT d2_cc=-d2_lower_src_A+d2_lower_dst_A;
        BIGINT d3_cc=-d3_lower_src_B+d3_lower_dst_B;
        for (BIGINT i3=d3_lower_src_B; i3<=d3_upper_src_B; i3++) {
            BIGINT in_tmp00=d1_lower_src_B+d1_size*d2_size*i3;
            BIGINT out_tmp00=(d1_lower_src_B+d1_cc)+d1_N*d2_N*(i3+d3_cc);
            for (BIGINT i2=d2_lower_src_A; i2<=d2_upper_src_A; i2++) {
                BIGINT input_index=in_tmp00+d1_size*i2;
                BIGINT output_index=out_tmp00+d1_N*(i2+d2_cc);
                for (BIGINT i1=d1_lower_src_B; i1<=d1_upper_src_B; i1++) {
                    //BIGINT input_index=i1+d1_size*i2+d1_size*d2_size*i3; //for reference
                    //BIGINT output_index=(i1+d1_cc)+d1_N*(i2+d2_cc)+d1_N*d2_N*(i3+d3_cc); //for reference
                    data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                    data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                    input_index++;
                    output_index++;
                }
            }
        }
    }

    // BBA
    if ((d1_lower_src_B>=0)&&(d2_lower_src_B>=0)&&(d3_lower_src_A>=0)) {
        BIGINT input_index=0;
        BIGINT output_index=0;
        BIGINT d1_cc=-d1_lower_src_B+d1_lower_dst_B;
        BIGINT d2_cc=-d2_lower_src_B+d2_lower_dst_B;
        BIGINT d3_cc=-d3_lower_src_A+d3_lower_dst_A;
        for (BIGINT i3=d3_lower_src_A; i3<=d3_upper_src_A; i3++) {
            BIGINT in_tmp00=d1_lower_src_B+d1_size*d2_size*i3;
            BIGINT out_tmp00=(d1_lower_src_B+d1_cc)+d1_N*d2_N*(i3+d3_cc);
            for (BIGINT i2=d2_lower_src_B; i2<=d2_upper_src_B; i2++) {
                BIGINT input_index=in_tmp00+d1_size*i2;
                BIGINT output_index=out_tmp00+d1_N*(i2+d2_cc);
                for (BIGINT i1=d1_lower_src_B; i1<=d1_upper_src_B; i1++) {
                    //BIGINT input_index=i1+d1_size*i2+d1_size*d2_size*i3; //for reference
                    //BIGINT output_index=(i1+d1_cc)+d1_N*(i2+d2_cc)+d1_N*d2_N*(i3+d3_cc); //for reference
                    data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                    data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                    input_index++;
                    output_index++;
                }
            }
        }
    }

    // BBB
    if ((d1_lower_src_B>=0)&&(d2_lower_src_B>=0)&&(d3_lower_src_B>=0)) {
        BIGINT input_index=0;
        BIGINT output_index=0;
        BIGINT d1_cc=-d1_lower_src_B+d1_lower_dst_B;
        BIGINT d2_cc=-d2_lower_src_B+d2_lower_dst_B;
        BIGINT d3_cc=-d3_lower_src_B+d3_lower_dst_B;
        for (BIGINT i3=d3_lower_src_B; i3<=d3_upper_src_B; i3++) {
            BIGINT in_tmp00=d1_lower_src_B+d1_size*d2_size*i3;
            BIGINT out_tmp00=(d1_lower_src_B+d1_cc)+d1_N*d2_N*(i3+d3_cc);
            for (BIGINT i2=d2_lower_src_B; i2<=d2_upper_src_B; i2++) {
                BIGINT input_index=in_tmp00+d1_size*i2;
                BIGINT output_index=out_tmp00+d1_N*(i2+d2_cc);
                for (BIGINT i1=d1_lower_src_B; i1<=d1_upper_src_B; i1++) {
                    //BIGINT input_index=i1+d1_size*i2+d1_size*d2_size*i3; //for reference
                    //BIGINT output_index=(i1+d1_cc)+d1_N*(i2+d2_cc)+d1_N*d2_N*(i3+d3_cc); //for reference
                    data_uniform[output_index*2]+=data_uniform_0[input_index*2];
                    data_uniform[output_index*2+1]+=data_uniform_0[input_index*2+1];
                    input_index++;
                    output_index++;
                }
            }
        }
    }
}

*/

}

