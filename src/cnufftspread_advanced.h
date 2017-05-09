#ifndef CNUFFTSPREAD_ADVANCED_H
#define CNUFFTSPREAD_ADVANCED_H

#include "cnufftspread.h"
#include <vector>

int cnufftspread_advanced(BIGINT N1, BIGINT N2, BIGINT N3, FLT *data_uniform,
         BIGINT M, FLT *kx, FLT *ky, FLT *kz,
         FLT *data_nonuniform, spread_opts opts);

#endif // CNUFFTSPREAD_ADVANCED_H

