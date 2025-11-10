/**
 * MTL Tensor Utilities
 * Helper functions for tensor buffer management and layout
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "mtl_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// TENSOR HELPERS
// ============================================================================

void mtl_compute_tensor_make_row_major(MTLComputeTensorDesc* desc) {
    if (!desc) {
        return;
    }
    
    // Compute row-major strides: Z-Y-X (rightmost has stride 1)
    desc->stride_x = 1;
    desc->stride_y = desc->width;
    desc->stride_z = desc->width * desc->height;
}

void* mtl_compute_tensor_data_ptr(MTLComputeBuffer* tensor_buffer) {
    if (!tensor_buffer) {
        return NULL;
    }
    
    void* base = mtl_compute_buffer_contents(tensor_buffer);
    if (!base) {
        return NULL;
    }
    
    // Skip descriptor header (16-byte aligned)
    size_t offset = (sizeof(MTLComputeTensorDesc) + 15) & ~15;
    return (uint8_t*)base + offset;
}

// ============================================================================
// AUTO-TUNING
// ============================================================================

MTLComputeError mtl_compute_auto_tune(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    size_t total_threads,
    size_t* out_width,
    size_t* out_height,
    size_t* out_depth
) {
    if (!device || !pipeline || !out_width) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        // Get pipeline limits
        NSUInteger max_threads = pipeline->pipelineState.maxTotalThreadsPerThreadgroup;
        NSUInteger exec_width = pipeline->pipelineState.threadExecutionWidth;
        
        // Try different threadgroup sizes (multiples of SIMD width)
        size_t candidates[] = {32, 64, 128, 256, 512, 1024};
        size_t best_size = (size_t)exec_width;
        double best_time = INFINITY;
        
        // Create small test buffer
        size_t test_size = 1024 * sizeof(float);
        MTLComputeBuffer* test_buf = mtl_compute_buffer_create(device, test_size, MTL_STORAGE_SHARED);
        if (!test_buf) {
            *out_width = (size_t)exec_width;
            if (out_height) *out_height = 1;
            if (out_depth) *out_depth = 1;
            return MTL_SUCCESS;
        }
        
        MTL_LOG("Auto-tuning threadgroup size for pipeline...");
        
        for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
            size_t tg_size = candidates[i];
            
            // Skip if too large
            if (tg_size > max_threads) {
                continue;
            }
            
            // Must be multiple of execution width
            if (tg_size % exec_width != 0) {
                continue;
            }
            
            // Benchmark this size
            MTLComputePerformanceStats stats;
            MTLComputeBuffer* buffers[] = {test_buf};
            
            MTLComputeError err = mtl_compute_dispatch_profiled(
                device, pipeline, buffers, 1, 256, tg_size, &stats
            );
            
            if (err == MTL_SUCCESS && stats.gpu_time_ms < best_time) {
                best_time = stats.gpu_time_ms;
                best_size = tg_size;
            }
        }
        
        mtl_compute_buffer_destroy(test_buf);
        
        MTL_LOG("Auto-tuned threadgroup size: %zu (%.3f ms)", best_size, best_time);
        *out_width = best_size;
        if (out_height) *out_height = 1;
        if (out_depth) *out_depth = 1;
        
        return MTL_SUCCESS;
    }
}

// ============================================================================
// TENSOR OPERATIONS (Using Standard Kernels)
// ============================================================================

MTLComputeError mtl_tensor_fill(
    MTLComputeDevice* device,
    MTLComputeBuffer* tensor,
    float value
) {
    if (!device || !tensor) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    // Load fill kernel from standard library
    MTLComputeError err;
    MTLComputePipeline* fill_pipeline = mtl_compute_pipeline_create_from_file(
        device, "shaders/standard_kernels.metal", "fill_float", &err, NULL, 0
    );
    
    if (!fill_pipeline) {
        MTL_LOG("Failed to load fill_float kernel");
        return MTL_ERROR_SHADER_COMPILATION;
    }
    
    MTLComputeBuffer* value_buf = mtl_compute_buffer_create_with_data(
        device, &value, sizeof(float), MTL_STORAGE_SHARED
    );
    
    if (!value_buf) {
        mtl_compute_pipeline_destroy(fill_pipeline);
        return MTL_ERROR_BUFFER_CREATION;
    }
    
    // Calculate tensor size from dimensions (3D)
    void* tensor_ptr = mtl_compute_buffer_contents(tensor);
    MTLComputeTensorDesc* desc = (MTLComputeTensorDesc*)tensor_ptr;
    size_t elements = desc->width * desc->height * desc->depth;
    
    // Dispatch
    MTLComputeBuffer* buffers[] = {tensor, value_buf};
    err = mtl_compute_dispatch_1d(
        device, fill_pipeline, buffers, 2, elements, 256
    );
    
    mtl_compute_buffer_destroy(value_buf);
    mtl_compute_pipeline_destroy(fill_pipeline);
    
    return err;
}

MTLComputeError mtl_tensor_saxpy(
    MTLComputeDevice* device,
    float alpha,
    MTLComputeBuffer* x,
    MTLComputeBuffer* y,
    MTLComputeBuffer* result
) {
    if (!device || !x || !y || !result) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    MTLComputePipeline* saxpy_pipeline = mtl_compute_pipeline_create_from_file(
        device, "shaders/standard_kernels.metal", "saxpy", NULL, NULL, 0
    );
    
    if (!saxpy_pipeline) {
        return MTL_ERROR_SHADER_COMPILATION;
    }
    
    MTLComputeBuffer* alpha_buf = mtl_compute_buffer_create_with_data(
        device, &alpha, sizeof(float), MTL_STORAGE_SHARED
    );
    
    if (!alpha_buf) {
        mtl_compute_pipeline_destroy(saxpy_pipeline);
        return MTL_ERROR_BUFFER_CREATION;
    }
    
    // Get tensor size from first buffer (3D)
    void* x_ptr = mtl_compute_buffer_contents(x);
    MTLComputeTensorDesc* desc = (MTLComputeTensorDesc*)x_ptr;
    size_t elements = desc->width * desc->height * desc->depth;
    
    MTLComputeBuffer* buffers[] = {x, y, result, alpha_buf};
    MTLComputeError err = mtl_compute_dispatch_1d(
        device, saxpy_pipeline, buffers, 4, elements, 256
    );
    
    mtl_compute_buffer_destroy(alpha_buf);
    mtl_compute_pipeline_destroy(saxpy_pipeline);
    
    return err;
}

