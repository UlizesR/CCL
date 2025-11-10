/**
 * MTLComp Standard Kernel Library
 * Pre-built kernels for common operations (numerics, testing, utilities)
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// BASIC OPERATIONS
// ============================================================================

// Fill buffer with constant value
kernel void fill_float(
    device float* output [[buffer(0)]],
    constant float& value [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = value;
}

// Copy buffer
kernel void copy_float(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = input[gid];
}

// Vector addition (SAXPY style)
kernel void saxpy(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = alpha * x[gid] + y[gid];
}

// Element-wise multiply
kernel void multiply_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = a[gid] * b[gid];
}

// ============================================================================
// REDUCTIONS
// ============================================================================

// Parallel reduction (sum) - Phase 1: Threadgroup reduce
kernel void reduce_sum_threadgroup(
    device const float* input [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Load into threadgroup memory
    float value = (gid < count) ? input[gid] : 0.0f;
    shared[lid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && (lid + stride) < group_size) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write partial sum
    if (lid == 0) {
        partial_sums[tid] = shared[0];
    }
}

// Min/Max reduction
kernel void reduce_min(
    device const float* input [[buffer(0)]],
    device float* partial_mins [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    float value = (gid < count) ? input[gid] : FLT_MAX;
    shared[lid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && (lid + stride) < group_size) {
            shared[lid] = min(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        partial_mins[tid] = shared[0];
    }
}

// ============================================================================
// SCIENTIFIC COMPUTING
// ============================================================================

// 2D Heat Equation (5-point stencil)
kernel void heat_2d_step(
    device const float* current [[buffer(0)]],
    device float* next [[buffer(1)]],
    constant float& dt [[buffer(2)]],
    constant float& dx [[buffer(3)]],
    constant uint2& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = dims.x;
    uint height = dims.y;
    
    if (gid.x == 0 || gid.x >= width - 1 || gid.y == 0 || gid.y >= height - 1) {
        // Boundary - keep same
        next[gid.y * width + gid.x] = current[gid.y * width + gid.x];
        return;
    }
    
    uint idx = gid.y * width + gid.x;
    float center = current[idx];
    float left   = current[idx - 1];
    float right  = current[idx + 1];
    float up     = current[idx - width];
    float down   = current[idx + width];
    
    // Laplacian
    float laplacian = (left + right + up + down - 4.0f * center) / (dx * dx);
    
    // Forward Euler
    next[idx] = center + dt * laplacian;
}

// 3D Diffusion (7-point stencil)
kernel void diffusion_3d_step(
    device const float* current [[buffer(0)]],
    device float* next [[buffer(1)]],
    constant float& dt [[buffer(2)]],
    constant float& dx [[buffer(3)]],
    constant uint3& dims [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint width = dims.x;
    uint height = dims.y;
    uint depth = dims.z;
    
    if (gid.x == 0 || gid.x >= width - 1 ||
        gid.y == 0 || gid.y >= height - 1 ||
        gid.z == 0 || gid.z >= depth - 1) {
        // Boundary
        uint idx = (gid.z * height + gid.y) * width + gid.x;
        next[idx] = current[idx];
        return;
    }
    
    uint idx = (gid.z * height + gid.y) * width + gid.x;
    uint slice = width * height;
    
    float center = current[idx];
    float left   = current[idx - 1];
    float right  = current[idx + 1];
    float up     = current[idx - width];
    float down   = current[idx + width];
    float front  = current[idx - slice];
    float back   = current[idx + slice];
    
    float laplacian = (left + right + up + down + front + back - 6.0f * center) / (dx * dx);
    
    next[idx] = center + dt * laplacian;
}

// ============================================================================
// LINEAR ALGEBRA
// ============================================================================

// Dot product (partial, requires reduction)
kernel void dot_product_partial(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* partial_dots [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]]
) {
    float value = (gid < count) ? (a[gid] * b[gid]) : 0.0f;
    shared[lid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && (lid + stride) < group_size) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        partial_dots[tid] = shared[0];
    }
}

// Matrix-vector multiply (Ax = y)
kernel void matvec_multiply(
    device const float* A [[buffer(0)]],     // Matrix (row-major)
    device const float* x [[buffer(1)]],     // Input vector
    device float* y [[buffer(2)]],           // Output vector
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    
    float sum = 0.0f;
    for (uint col = 0; col < cols; col++) {
        sum += A[gid * cols + col] * x[col];
    }
    y[gid] = sum;
}

// ============================================================================
// SIGNAL PROCESSING
// ============================================================================

// 1D convolution
kernel void convolve_1d(
    device const float* signal [[buffer(0)]],
    device const float* kernel_weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& signal_len [[buffer(3)]],
    constant uint& kernel_len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= signal_len) return;
    
    int half_kernel = kernel_len / 2;
    float sum = 0.0f;
    
    for (int k = 0; k < (int)kernel_len; k++) {
        int idx = (int)gid + k - half_kernel;
        if (idx >= 0 && idx < (int)signal_len) {
            sum += signal[idx] * kernel_weights[k];
        }
    }
    
    output[gid] = sum;
}

// Prefix sum (scan) - Hillis-Steele algorithm
kernel void prefix_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* temp [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& offset [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    if (offset == 0) {
        output[gid] = input[gid];
    } else if (gid >= offset) {
        output[gid] = temp[gid] + temp[gid - offset];
    } else {
        output[gid] = temp[gid];
    }
}

// ============================================================================
// UTILITIES
// ============================================================================

// Generate random-ish test data (deterministic)
kernel void generate_test_data(
    device float* output [[buffer(0)]],
    constant uint& seed [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    // Simple LCG for reproducible test data
    uint state = seed + gid;
    state = state * 1664525u + 1013904223u;
    output[gid] = float(state & 0xFFFFFFu) / float(0xFFFFFFu);
}

// Validate results (count mismatches)
kernel void validate_results(
    device const float* computed [[buffer(0)]],
    device const float* expected [[buffer(1)]],
    device atomic_uint* mismatch_count [[buffer(2)]],
    constant float& tolerance [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float diff = abs(computed[gid] - expected[gid]);
    if (diff > tolerance) {
        atomic_fetch_add_explicit(mismatch_count, 1, memory_order_relaxed);
    }
}

