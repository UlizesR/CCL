/**
 * Indirect Command Buffer Demo - MSL 6.16
 * Pre-record compute dispatches for efficient replay
 */

#include <metal_stdlib>
using namespace metal;

// Simple kernel for ICB testing
kernel void icb_process_step(
    device float* data [[buffer(0)]],
    constant uint& step_id [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // Apply transformation based on step
    data[gid] = data[gid] * float(step_id + 1) + 0.1;
}

// Kernel that generates indirect dispatch parameters
kernel void generate_indirect_params(
    device MTLDispatchThreadgroupsIndirectArguments* params [[buffer(0)]],
    constant uint& num_elements [[buffer(1)]]
) {
    // Calculate threadgroups needed for num_elements
    uint threads_per_group = 256;
    params->threadgroupsPerGrid[0] = (num_elements + threads_per_group - 1) / threads_per_group;
    params->threadgroupsPerGrid[1] = 1;
    params->threadgroupsPerGrid[2] = 1;
}

