#ifndef CNUFFTSPREAD_ADVANCED_H
#define CNUFFTSPREAD_ADVANCED_H

#include "cnufftspread.h"
#include <vector>

/*
NUFFT type 1 spreader (only does the spreading -- nonuniform -> uniform)

Inputs:
    N1xN2xN3 -- the dimensions of the uniform output grid
    data_uniform -- the non-uniform input data (complex vector of size M)
    M -- the number of non-uniform points
    kx,ky,kz -- the locations of the non-uniform points
    opts -- the options for the spreader

Outputs:
    data_nonuniform -- the output data on the uniform grid
*/
int cnufftspread_advanced(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
         BIGINT M, FLT *kx, FLT *ky, FLT *kz,
         FLT *data_nonuniform, spread_opts opts);

/////////////////////////////////////////////////////////////////////////////////
/*
== Timing flags ==
See spread_opts.timing_flags.

    This is an unobtrusive way to determine the time contributions of the different
    components of the algorithm by selectively leaving them out.
    For example, running the following two tests should show the modest gain
    achieved by bin-sorting the subproblems (the last argument is the flag)
    > test/spreadtestnd 3 1e7 1e6 1e-6 2 0
    > test/spreadtestnd 3 1e7 1e6 1e-6 2 32

*/
#define TF_OMIT_WRITE_TO_GRID          1  // don't write to the output grid at all
#define TF_OMIT_LOCK_GRID              2  // don't lock the output grid (should produce inaccurate output)
#define TF_OMIT_EVALUATE_KERNEL        4  // don't evaluate the kernel at all
#define TF_OMIT_EVALUATE_EXPONENTIAL   8  // don't evaluate the exp operation in the kernel
#define TF_OMIT_PI_RANGE               16 // don't convert the data to/from [-pi,pi) range
#define TF_OMIT_SORT_SUBPROBLEMS       32 // don't bin-sort the subproblems
#define TF_OMIT_SPREADING              64 // don't spread at all!
/////////////////////////////////////////////////////////////////////////////////

#endif // CNUFFTSPREAD_ADVANCED_H

