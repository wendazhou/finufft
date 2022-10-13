#pragma once

#include "../../plan.h"

namespace finufft {
    namespace avx512 {
        template<typename T, std::size_t Dim>
        Type1Plan<T, Dim> make_type1_plan(Type1TransformConfiguration<Dim> const &configuration);
    }
}
