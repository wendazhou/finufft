#include <spreadinterp.h>
#include "ker_horner_avx2.h"
#include "spread_kernels.h"

namespace finufft
{
    template <typename FT>
    void spread_subproblem_1d_avx2_7(BIGINT off1, BIGINT size1, FT *du, BIGINT M,
                                     FT *kx, FT *dd, const spread_opts &opts)
    /* 1D spreader from nonuniform to uniform subproblem grid, without wrapping.
       Inputs:
       off1 - integer offset of left end of du subgrid from that of overall fine
              periodized output grid {0,1,..N-1}.
       size1 - integer length of output subgrid du
       M - number of NU pts in subproblem
       kx (length M) - are rescaled NU source locations, should lie in
                       [off1+ns/2,off1+size1-1-ns/2] so as kernels stay in bounds
       dd (length M complex, interleaved) - source strengths
       Outputs:
       du (length size1 complex, interleaved) - preallocated uniform subgrid array

       The reason periodic wrapping is avoided in subproblems is speed: avoids
       conditionals, indirection (pointers), and integer mod. Originally 2017.
       Kernel eval mods by Ludvig al Klinteberg.
       Fixed so rounding to integer grid consistent w/ get_subgrid, prevents
       chance of segfault when epsmach*N1>O(1), assuming max() and ceil() commute.
       This needed off1 as extra arg. AHB 11/30/20.
    */
    {
        int ns = opts.nspread; // a.k.a. w
        FT ns2 = (FT)ns / 2;   // half spread width

        for (BIGINT i = 0; i < 2 * size1; ++i) // zero output
            du[i] = 0.0;

        for (BIGINT i = 0; i < M; i++)
        { // loop over NU pts
            FT re0 = dd[2 * i];
            FT im0 = dd[2 * i + 1];
            // ceil offset, hence rounding, must match that in get_subgrid...
            FT i1 = std::ceil(kx[i] - ns2); // fine grid start index
            FT x1 = i1 - kx[i];                   // x1 in [-w/2,-w/2+1], up to rounding
            // However if N1*epsmach>O(1) then can cause O(1) errors in x1, hence ppoly
            // kernel evaluation will fall outside their designed domains, >>1 errors.
            // This can only happen if the overall error would be O(1) anyway. Clip x1??
            if (x1 < -ns2)
                x1 = -ns2;
            if (x1 > -ns2 + 1)
                x1 = -ns2 + 1; // ***

            BIGINT j = i1 - off1; // offset rel to subgrid, starts the output indices

            accumulate_kernel_vec_horner_7_avx2(
                du + 2 * j, x1, re0, im0);
        }
    }

}

template void finufft::spread_subproblem_1d_avx2_7<float>(BIGINT, BIGINT, float *, BIGINT, float *, float *, const spread_opts &);
