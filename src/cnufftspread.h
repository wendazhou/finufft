#ifndef CNUFFTSPREAD_H
#define CNUFFTSPREAD_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

#define MAX_NSPREAD 16     // upper bound on w, ie nspread; also for common

// Note -std=c++11 is needed to avoid warning for static initialization here:
struct spread_opts {
  int nspread=6;           // w, the kernel width in grid pts
  // opts controlling spreading method (indep of kernel)...
  int spread_direction=1;  // 1 means spread NU->U, 2 means interpolate U->NU
  int pirange=0;           // 0: coords in [0,N), 1 coords in [-pi,pi)
  int chkbnds=1;           // 0: don't check NU pts are in range; 1: do
  int sort=1;              // 0: don't sort NU pts, 1: do sort (better on i7)
  BIGINT max_subproblem_size=1e5; // sets extra RAM per thread
  int flags=0;             // binary flags for timing only (may give wrong ans!)
  int debug=0;             // 0: silent, 1: small text output, 2: verbose
  // ES kernel specific...
  FLT ES_beta;
  FLT ES_halfwidth;
  FLT ES_c;
};

// macro: if p is true, rescales from [-pi,pi] to [0,N] ...
#define RESCALE(x,N,p) (p ? ((x*M_1_2PI + 0.5)*N) : x)


// things external interface needs...
int cnufftspread(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
		 BIGINT M, FLT *kx, FLT *ky, FLT *kz,
		 FLT *data_nonuniform, spread_opts opts);
FLT evaluate_kernel(FLT x,const spread_opts &opts);
FLT evaluate_kernel_noexp(FLT x,const spread_opts &opts);
int setup_kernel(spread_opts &opts,FLT eps,FLT R);

#endif // CNUFFTSPREAD_H
