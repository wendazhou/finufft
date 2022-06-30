#pragma once

// Utilities for cpuid-based dispatching of functions.

namespace finufft {

/** Enumeration representing the capability of kernels to be executed.
 *
 * This is intended to be a simplified mechanism for selecting kernels
 * based on cpuid capabilities. The enum is ordered from least to most
 * powerful instruction set, and it is assumed that higher instruction
 * sets are inclusive of lower ones.
 */
enum class DispatchCapability : int {
    Scalar = 0, //< No SIMD instruction capability
    SSE4 = 1,   //< SSE4.1 instructions
    AVX2 = 2,   //< AVX2 + FMA instructions
    AVX512 = 3  //< AVX512F + AVX512VL + AVX512DQ

}; // namespace finufft

/// Query the current cpu for its capability.
DispatchCapability get_current_capability() noexcept;

/// Query the current dispatch target, can be different from the current capability
/// on user request (e.g. by setting the environment variable FINUFFT_DISPATCH).
DispatchCapability get_current_dispatch_target() noexcept;

} // namespace finufft