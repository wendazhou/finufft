#include "cnufftspread_advanced.h"
#include "cnufftspread.h"
#include <vector>
#include <set>
#include <algorithm>

namespace Advanced {
// Use a namespace to keep things tidy, self-contained and avoid conflicts with other source files

// A subproblem is simply a collection of indices to nonuniform points
struct Subproblem {
    std::vector<BIGINT> nonuniform_indices;
};

// A subgrid is a rectilinear subset of the uniform grid defined by x1,x2,y1,y2,z1,z2
// This class contains methods for copying and comparing subgrids and for
// determining whether two subgrids intersect
struct Subgrid {
    Subgrid(BIGINT x1,BIGINT x2,BIGINT y1,BIGINT y2,BIGINT z1,BIGINT z2);
    Subgrid(const Subgrid &other);
    void operator=(const Subgrid &other);
    bool operator==(const Subgrid &other);
    bool intersects(const Subgrid &other,BIGINT N1,BIGINT N2,BIGINT N3); //N1,N2,N3 needed for wrapping
    BIGINT x1=0,x2=0;
    BIGINT y1=0,y2=0;
    BIGINT z1=0,z2=0;
private:
    void copy_from(const Subgrid &other);
};

// For each subproblem, we should sort the indices for cache efficiency
void bin_sort_subproblem_inds(std::vector<BIGINT> &inds,FLT *kx, FLT *ky, FLT *kz,double bin_size_x,double bin_size_y,double bin_size_z);

// utility function: get the offsets and sizes of the subgrid defined by the nonuniform points and the spreading diameter
void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,BIGINT &size2,BIGINT &size3,BIGINT M0,FLT* kx0,FLT* ky0,FLT* kz0,int nspread);

// do the spreading for the subproblem
void cnufftspread_type1_subproblem(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
         BIGINT M, FLT *kx, FLT *ky, FLT *kz,
         FLT *data_nonuniform, spread_opts opts);

// convert nonuniform locations to [-pi,pi) range from [0,N)
void to_pi_range(BIGINT M, FLT *X, BIGINT N);

// convert nonuniform locations from [-pi,pi) range to [0,N)
void from_pi_range(BIGINT M, FLT *X, BIGINT N);

}

// Convenience: Return true if opts.timing_flags does not contain the flag (usually it is a flag of omission)
bool noflag(const spread_opts &opts,int flag) {
    return (!(opts.timing_flags & flag));
}

int cnufftspread_advanced(BIGINT N1, BIGINT N2, BIGINT N3, FLT* data_uniform, BIGINT M, FLT* kx, FLT* ky, FLT* kz, FLT* data_nonuniform, spread_opts opts)
{
    // Initialize the output data
    BIGINT NNN=N1*N2*N3*2;
    for (BIGINT i=0; i<NNN; i++) {
        data_uniform[i]=0;
    }

    // If needed, change units for the nonuniform locations
    if (noflag(opts,TF_OMIT_PI_RANGE)) {
        if (opts.pirange) {
            Advanced::from_pi_range(M,kx,N1);
            Advanced::from_pi_range(M,ky,N2);
            Advanced::from_pi_range(M,kz,N3);
        }
        // Check whether everything is in the right range
        for (BIGINT i=0; i<M; i++) {
            if ((kx[i]<0)||(kx[i]>=N1)) {
                printf("Location is out of range kx=%g, N1=%ld (pirange=%d)\n",kx[i],N1,opts.pirange);
                return -1;
            }
            if ((ky[i]<0)||(ky[i]>=N2)) {
                printf("Location is out of range ky=%g, N2=%ld (pirange=%d)\n",ky[i],N2,opts.pirange);
                return -1;
            }
            if ((kz[i]<0)||(kz[i]>=N2)) {
                printf("Location is out of range kz=%g, N3=%ld (pirange=%d)\n",kz[i],N3,opts.pirange);
                return -1;
            }
        }
    }

    // some hard-coded parameters
    // w2 and w3 are the bin sizes in the 2nd and 3rd dimensions for dividing the
    // nonuniform points into subproblems
    // Those subproblems will then be split further in order to satisfy the
    // max_subproblem_size requirement
    int nthr=omp_get_max_threads();
    int cc=4;
    int w2=ceil(N1*1.0/sqrt(nthr*cc)),w3=ceil(N2*1.0/sqrt(nthr*cc));
    BIGINT max_subproblem_size=1e5; //controls extra RAM allocation per thread, but it really won't be very much

    // Define the subproblems
    BIGINT A1=ceil(N2*1.0/w2); // number of subproblem bins in 2nd dimension
    BIGINT A2=ceil(N3*1.0/w3); // number of subproblem bins in 3rd dimension
    std::vector<Advanced::Subproblem> subproblems;
    subproblems.resize(A1*A2); // the initial number of subproblems is A1*A2
    for (BIGINT i=0; i<M; i++) {
        BIGINT i2=floor(ky[i]/w2);
        BIGINT i3=floor(kz[i]/w3);
        BIGINT subproblem_index=i2+A1*i3;
        // add the index to the appropriate subproblem (bin)
        subproblems.at(subproblem_index).nonuniform_indices.push_back(i);
    }
    // next we split some of the subproblems so that non have size greater than max_subproblem_size
    BIGINT original_num_subproblems=subproblems.size();
    BIGINT num_nonempty_subproblems=0; // for information only (verbose output)
    for (BIGINT i=0; i<original_num_subproblems; i++) {
        std::vector<BIGINT> inds=subproblems.at(i).nonuniform_indices;
        BIGINT num_nonuniform_points=inds.size();
        if (num_nonuniform_points>max_subproblem_size) { // does the size exceed maximum allowed?
            BIGINT next=0;
            for (BIGINT j=0; j+max_subproblem_size<=num_nonuniform_points; j+=max_subproblem_size) {
                Advanced::Subproblem X; // the new subproblem (shall we say sub-subproblem?)
                X.nonuniform_indices=std::vector<BIGINT>(inds.begin()+j,inds.begin()+j+max_subproblem_size);
                subproblems.push_back(X);
                num_nonempty_subproblems++; // for info only
                next=j+max_subproblem_size;
            }
            // the remainder of the indices go to the ith subproblem
            // it is possible that this will now be an empty subproblem (that's okay)
            subproblems.at(i).nonuniform_indices=std::vector<BIGINT>(inds.begin()+next,inds.end());
        }
        if (subproblems.at(i).nonuniform_indices.size()>0) {
            num_nonempty_subproblems++; // for info only
        }
    }
    printf("Using %ld subproblems.\n",num_nonempty_subproblems);

    // The main spreading starts now
    BIGINT num_subproblems=subproblems.size();
    #pragma omp parallel for
    for (int jsub=0; jsub<num_subproblems; jsub++) { // Loop through the subproblems
        // I was thinking of doing the subproblems in a different order to avoid
        // lock conflicts whenever possible, but doesn't seem to make a difference.
        // The time used in waiting for locks to be released seems to be very minimal
        int isub=(jsub*1)%num_subproblems; // the 1 would be changed to a number relatively prime with num_subproblems and approximately equal to sqrt(num_subproblems)
        std::vector<BIGINT> inds=subproblems.at(isub).nonuniform_indices;
        BIGINT M0=inds.size(); // number of nonuniform points in the subproblem
        if (M0>0) {
            if (noflag(opts,TF_OMIT_SORT_SUBPROBLEMS)) {
                // Bin-sorting the nonuniform points in the subproblem reduces cache-misses
                // Mainly this is for the 1st dimension since the bin width is already limited in the 2nd and 3rd dimensions
                Advanced::bin_sort_subproblem_inds(inds,kx,ky,kz,3,3,3);
            }

            // Fill the location and data vectors for the nonuniform points
            FLT* kx0=(FLT*)malloc(sizeof(FLT)*M0);
            FLT* ky0=(FLT*)malloc(sizeof(FLT)*M0);
            FLT* kz0=(FLT*)malloc(sizeof(FLT)*M0);
            FLT* dd0=(FLT*)malloc(sizeof(FLT)*M0*2); // remember it's complex
            for (BIGINT j=0; j<M0; j++) {
                BIGINT kk=inds[j];
                kx0[j]=kx[kk]; //later we'll subtract the offset
                ky0[j]=ky[kk];
                kz0[j]=kz[kk];
                dd0[j*2]=data_nonuniform[kk*2]; // real part
                dd0[j*2+1]=data_nonuniform[kk*2+1]; // imag part
            }
            // get the subgrid which will include padding by approximately nspread/2
            BIGINT offset1,offset2,offset3,size1,size2,size3;
            Advanced::get_subgrid(offset1,offset2,offset3,size1,size2,size3,M0,kx0,ky0,kz0,opts.nspread);
            // printf("subgrid: %ld,%ld,%ld,%ld,%ld,%ld\n",offset1,offset2,offset3,size1,size2,size3);
            for (BIGINT j=0; j<M0; j++) {
                kx0[j]-=offset1; // subtract the offsets so this is really a subproblem
                ky0[j]-=offset2;
                kz0[j]-=offset3;
            }
            // allocate the output data for this subproblem
            FLT* data_uniform_0=(FLT*)malloc(sizeof(FLT)*size1*size2*size3*2); // remember complex
            if (noflag(opts,TF_OMIT_SPREADING)) {
                // Actually do the spreading
                Advanced::cnufftspread_type1_subproblem(size1,size2,size3,data_uniform_0,M0,kx0,ky0,kz0,dd0,opts);
            }


            {
                // Now we write to the output grid

                // Prepare the output indices taking into consideration wrapping
                BIGINT output_inds1[size1];
                BIGINT output_inds2[size2];
                BIGINT output_inds3[size3];
                for (BIGINT a=0; a<size1; a++) {
                    BIGINT ind0=offset1+a;
                    while (ind0<0) ind0+=N1; //wrapping
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

                #pragma omp critical
                {
                    if (noflag(opts,TF_OMIT_WRITE_TO_GRID)) {
                        // The innermost loop of writing to the output grid
                        // But ideally this will contribute a lot less time compared with the
                        // innermost loop of the subproblem spreading
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
                }
            }

            // Free the data for the subproblem (minimal RAM used)
            free(kx0);
            free(ky0);
            free(kz0);
            free(dd0);
            free(data_uniform_0);
        }
    }

    // Convert the data back to [-pi,pi) range if needed
    if (noflag(opts,TF_OMIT_PI_RANGE)) {
        if (opts.pirange) {
            Advanced::to_pi_range(M,kx,N1);
            Advanced::to_pi_range(M,ky,N2);
            Advanced::to_pi_range(M,kz,N3);
        }
    }

    return 0;
}

namespace Advanced {


// Utility
void to_pi_range(BIGINT M, FLT *X, BIGINT N)
{
    for (BIGINT i=0; i<M; i++) {
        X[i]=(X[i]/N-0.5)*2*PI;
    }
}

// Utility
void from_pi_range(BIGINT M, FLT *X, BIGINT N)
{
    for (BIGINT i=0; i<M; i++) {
        X[i]=(X[i]/(2*PI)+0.5)*N;
    }
}

// Utility
double compute_min(const std::vector<BIGINT>& inds,FLT *X) {
    BIGINT M=inds.size();
    if (M==0) return 0;

    double best=X[inds[0]];
    for (BIGINT i=0; i<M; i++) {
        double val=X[inds[i]];
        if (val<best) best=val;
    }
    return best;
}

// Utility
double compute_max(const std::vector<BIGINT>& inds,FLT *X) {
    BIGINT M=inds.size();
    if (M==0) return 0;

    double best=X[inds[0]];
    for (BIGINT i=0; i<M; i++) {
        double val=X[inds[i]];
        if (val>best) best=val;
    }
    return best;
}

// Utility
double compute_min(BIGINT M,FLT *X) {
    if (M==0) return 0;

    double best=X[0];
    for (BIGINT i=0; i<M; i++) {
        double val=X[i];
        if (val<best) best=val;
    }
    return best;
}

// Utility
double compute_max(BIGINT M,FLT *X) {
    if (M==0) return 0;

    double best=X[0];
    for (BIGINT i=0; i<M; i++) {
        double val=X[i];
        if (val>best) best=val;
    }
    return best;
}

// Bin-sorting the nonuniform points in the subproblem reduces cache-misses
void bin_sort_subproblem_inds(std::vector<BIGINT> &inds,FLT *kx, FLT *ky, FLT *kz,double bin_size_x,double bin_size_y,double bin_size_z) {
    BIGINT M=inds.size();

    FLT kx_min=compute_min(inds,kx);
    FLT kx_max=compute_max(inds,kx);
    FLT ky_min=compute_min(inds,ky);
    FLT ky_max=compute_max(inds,ky);
    FLT kz_min=compute_min(inds,kz);
    FLT kz_max=compute_max(inds,kz);

    BIGINT nbins1=(kx_max-kx_min)/bin_size_x+1;
    BIGINT nbins2=(ky_max-ky_min)/bin_size_y+1;
    BIGINT nbins3=(kz_max-kz_min)/bin_size_z+1;

    std::vector<BIGINT> bins(M);
    for (BIGINT i=0; i<M; i++) {
        BIGINT i1=(kx[inds[i]]-kx_min)/bin_size_x;
        BIGINT i2=(ky[inds[i]]-ky_min)/bin_size_y;
        BIGINT i3=(kz[inds[i]]-kz_min)/bin_size_z;
        bins[i]=i1+nbins1*i2+nbins1*nbins2*i3;
    }

    std::vector<BIGINT> counts(nbins1*nbins2*nbins3,0);
    for (BIGINT i=0; i<M; i++) {
        counts[bins[i]]++;
    }
    std::vector<BIGINT> offsets(nbins1*nbins2*nbins3);
    offsets[0]=0;
    for (BIGINT i=1; i<nbins1*nbins2*nbins3; i++) {
        offsets[i]=offsets[i-1]+counts[i-1];
    }

    std::vector<BIGINT> inv(M);
    for (BIGINT i=0; i<M; i++) {
        BIGINT offset=offsets[bins[i]];
        offsets[bins[i]]++;
        inv[i]=offset;
    }

    std::vector<BIGINT> ret(M);
    for (BIGINT i=0; i<M; i++) {
        ret[inv[i]]=inds[i];
    }
    inds=ret;
}

// get the subgrid which will include padding by approximately nspread/2
void get_subgrid(BIGINT &offset1,BIGINT &offset2,BIGINT &offset3,BIGINT &size1,BIGINT &size2,BIGINT &size3,BIGINT M,FLT* kx,FLT* ky,FLT* kz,int nspread) {
    // compute the min/max of the k-space locations of the nonuniform points
    double min_kx=compute_min(M,kx);
    double max_kx=compute_max(M,kx);
    double min_ky=compute_min(M,ky);
    double max_ky=compute_max(M,ky);
    double min_kz=compute_min(M,kz);
    double max_kz=compute_max(M,kz);

    // because we are very silly
    min_kx=min_kx;
    max_kx=max_kx;
    min_ky=min_ky;
    max_ky=max_ky;
    min_kz=min_kz;
    max_kz=max_kz;

    // we need some padding for the spreading
    int min_radius=(nspread+1)/2;
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

// Just for timing purposes!!!
FLT evaluate_kernel_noexp(FLT x, const spread_opts &opts)
{
  if (abs(x)>=opts.ES_halfwidth)
    return 0.0;
  else
    return (opts.ES_beta * sqrt(1.0 - opts.ES_c*x*x));
}

// Fill the ker vector with values (length = ns*ns*ns, where ns=opts.nspread)
void compute_kernel_values(FLT x1, FLT x2, FLT x3,
    const spread_opts& opts,FLT* ker)
{
    int ns=opts.nspread;
    FLT v1[ns], v2[ns], v3[ns];
    if (noflag(opts,TF_OMIT_EVALUATE_EXPONENTIAL)) {
        for (int i = 0; i <= ns; i++)
            v1[i] = evaluate_kernel(x1 + (FLT)i, opts);
        for (int i = 0; i <= ns; i++)
            v2[i] = evaluate_kernel(x2 + (FLT)i, opts);
        for (int i = 0; i <= ns; i++)
            v3[i] = evaluate_kernel(x3 + (FLT)i, opts);
    }
    else {
        for (int i = 0; i <= ns; i++)
            v1[i] = evaluate_kernel_noexp(x1 + (FLT)i, opts);
        for (int i = 0; i <= ns; i++)
            v2[i] = evaluate_kernel_noexp(x2 + (FLT)i, opts);
        for (int i = 0; i <= ns; i++)
            v3[i] = evaluate_kernel_noexp(x3 + (FLT)i, opts);
    }
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

// The actually spreading
void cnufftspread_type1_subproblem(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
         BIGINT M, FLT *kx, FLT *ky, FLT *kz,
         FLT *data_nonuniform, spread_opts opts) {

    // Initialize the output data
    int ns = opts.nspread;
    FLT ns2 = (FLT)ns / 2;
    for (BIGINT i = 0; i < 2 * N1 * N2 * N3; i++)
        data_uniform[i] = 0.0;

    FLT kernel_values[ns*ns*ns];   // is this the right way to allocate here?

    for (BIGINT i=0; i<M; i++) {

        FLT re0 = data_nonuniform[2 * i];
        FLT im0 = data_nonuniform[2 * i + 1];

        BIGINT i1 = (BIGINT)std::ceil(kx[i] - ns2);
        BIGINT i2 = (BIGINT)std::ceil(ky[i] - ns2);
        BIGINT i3 = (BIGINT)std::ceil(kz[i] - ns2);
        FLT x1 = (FLT)i1 - kx[i];
        FLT x2 = (FLT)i2 - ky[i];
        FLT x3 = (FLT)i3 - kz[i];

        if (noflag(opts,TF_OMIT_EVALUATE_KERNEL)) {
            compute_kernel_values(x1, x2, x3, opts, kernel_values);
        }

        // This is the critical innermost loop!
        // If you can speed this up you will have made a nice contribution to finufft!
        // But be aware that the compiler may be doing a lot of optimization already.
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

// Constructor
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

void Subgrid::copy_from(const Subgrid &other)
{
    x1=other.x1;
    x2=other.x2;
    y1=other.y1;
    y2=other.y2;
    z1=other.z1;
    z2=other.z2;
}

}

