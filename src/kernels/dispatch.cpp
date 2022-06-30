#include "dispatch.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include <cpuid.h>

namespace {
std::string to_lower(std::string const &s) {
    std::string result(s);

    for (auto &c : result) {
        c = std::tolower(c);
    }

    return result;
}
} // namespace

namespace finufft {

DispatchCapability get_current_capability() noexcept {
    // baseline dispatch type
    DispatchCapability result = DispatchCapability::Scalar;

    unsigned eax, ebx, ecx, edx, flag = 0;

    // query basic cpuid info
    int cpuidret = __get_cpuid(1, &eax, &ebx, &ecx, &edx);

    if (!cpuidret) {
        // failed to query cpuid
        return result;
    }

    if (ecx & bit_SSE4_1) {
        result = DispatchCapability::SSE4;
    }

    // query advanced cpuid info
    cpuidret = __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);

    if (!cpuidret) {
        // failed to query advanced cpuid
        return result;
    }

    if ((ebx & bit_AVX512F) && (ebx & bit_AVX512VL) && (ebx & bit_AVX512DQ)) {
        return DispatchCapability::AVX512;
    }

    if (ebx & bit_AVX2) {
        return DispatchCapability::AVX2;
    }

    return result;
}

DispatchCapability get_current_dispatch_target() noexcept {
    auto dispatch_c_str = std::getenv("FINUFFT_DISPATCH");
    DispatchCapability user_requested_type;

    if (dispatch_c_str) {
        // If the environment variable is set, use it.

        std::string dispatch_str(dispatch_c_str);
        dispatch_str = to_lower(dispatch_str);

        if (dispatch_str == "scalar") {
            user_requested_type = DispatchCapability::Scalar;
        } else if (dispatch_str == "sse4") {
            user_requested_type = DispatchCapability::SSE4;
        } else if (dispatch_str == "avx2") {
            user_requested_type = DispatchCapability::AVX2;
        } else if (dispatch_str == "avx512") {
            user_requested_type = DispatchCapability::AVX512;
        } else {
            // User requested unknown dispatch type, warn and fall back to default.
            std::cerr << "WARNING: FINUFFT_DISPATCH environment variable set to unknown value "
                      << dispatch_str << ", falling back to default dispatch type." << std::endl;
        }
    }

    auto current_capability = get_current_capability();

    if (dispatch_c_str && (user_requested_type > current_capability)) {
        // User requested a dispatch type that we believe is not supported.
        // Warn user here.

        std::cerr << "WARNING: FINUFFT_DISPATCH environment variable is set to " << dispatch_c_str
                  << ", but the current CPU does not support " << dispatch_c_str << ".\n";
    }

    return dispatch_c_str ? user_requested_type : current_capability;
}

} // namespace finufft
