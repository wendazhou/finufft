#pragma once

#include <spreadinterp.h>

namespace finufft
{
    template <typename FT>
    void spread_subproblem_1d_avx2_7(BIGINT off1, BIGINT size1, FT *du, BIGINT M,
                                     FT *kx, FT *dd, const spread_opts &opts);
}
