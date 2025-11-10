/**
 * MTL Advanced Dispatch
 * Async, profiled, and indirect dispatch variants (Tier 4)
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "mtl_internal.h"
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// ASYNC DISPATCH WITH EVENTS
// ============================================================================

MTLComputeError mtl_compute_dispatch_async(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t total_threads,
    size_t threads_per_group,
    MTLComputeSharedEvent* event,
    uint64_t signal_value
) {
    if (!device || !pipeline) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    if (event && !device->caps.supports_shared_events) {
        return MTL_ERROR_UNSUPPORTED;
    }
    
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = mtl_create_command_buffer(device);
        if (!commandBuffer) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        id<MTLComputeCommandEncoder> computeEncoder = mtl_create_compute_encoder(commandBuffer);
        if (!computeEncoder) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        // Build descriptor and use unified encoder
        MTLComputeDispatchDesc desc = {
            .pipeline = pipeline,
            .buffers = buffers,
            .buffer_count = buffer_count,
            .textures = NULL,
            .texture_count = 0,
            .samplers = NULL,
            .sampler_count = 0,
            .grid_width = total_threads,
            .grid_height = 1,
            .grid_depth = 1,
            .threadgroup_width = threads_per_group,
            .threadgroup_height = 1,
            .threadgroup_depth = 1
        };
        
        MTLComputeError err = mtl_encode_core(computeEncoder, &desc);
        if (err != MTL_SUCCESS) {
            return err;
        }
        
        [computeEncoder endEncoding];
        
        // Signal event when complete
        if (event && event->event) {
            [commandBuffer encodeSignalEvent:event->event value:signal_value];
        }
        
        [commandBuffer commit];
        
        return MTL_SUCCESS;
    }
}

// ============================================================================
// PROFILED DISPATCH
// ============================================================================

MTLComputeError mtl_compute_dispatch_profiled(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t total_threads,
    size_t threads_per_group,
    MTLComputePerformanceStats* stats
) {
    if (!device || !pipeline || !stats) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        CFAbsoluteTime cpuStart = CFAbsoluteTimeGetCurrent();
        
        id<MTLCommandBuffer> commandBuffer = mtl_create_command_buffer(device);
        if (!commandBuffer) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        id<MTLComputeCommandEncoder> computeEncoder = mtl_create_compute_encoder(commandBuffer);
        if (!computeEncoder) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        // Build descriptor and use unified encoder
        MTLComputeDispatchDesc desc = {
            .pipeline = pipeline,
            .buffers = buffers,
            .buffer_count = buffer_count,
            .textures = NULL,
            .texture_count = 0,
            .samplers = NULL,
            .sampler_count = 0,
            .grid_width = total_threads,
            .grid_height = 1,
            .grid_depth = 1,
            .threadgroup_width = threads_per_group,
            .threadgroup_height = 1,
            .threadgroup_depth = 1
        };
        
        size_t memory_used = 0;
        for (size_t i = 0; i < buffer_count; i++) {
            if (buffers[i] && buffers[i]->buffer) {
                memory_used += [buffers[i]->buffer length];
            }
        }
        
        MTLComputeError err = mtl_encode_core(computeEncoder, &desc);
        if (err != MTL_SUCCESS) {
            return err;
        }
        
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        CFAbsoluteTime cpuEnd = CFAbsoluteTimeGetCurrent();
        
        if ([commandBuffer error]) {
            MTL_LOG("Compute execution failed: %s", [[commandBuffer error] localizedDescription].UTF8String);
            return MTL_ERROR_EXECUTION;
        }
        
        // Fill enhanced statistics
        stats->cpu_time_ms = (cpuEnd - cpuStart) * 1000.0;
        stats->gpu_time_ms = ([commandBuffer GPUEndTime] - [commandBuffer GPUStartTime]) * 1000.0;
        stats->threads_executed = total_threads;
        
        size_t groups = (total_threads + threads_per_group - 1) / threads_per_group;
        stats->threadgroups_executed = groups;
        
        stats->memory_used_bytes = memory_used;
        stats->threadgroup_memory_used = [pipeline->pipelineState threadExecutionWidth];
        stats->execution_width = [pipeline->pipelineState threadExecutionWidth];
        stats->throughput_gflops = 0.0;
        
        return MTL_SUCCESS;
    }
}

// ============================================================================
// INDIRECT DISPATCH
// ============================================================================

MTLComputeError mtl_compute_dispatch_indirect(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    MTLComputeBuffer* indirect_buffer,
    size_t indirect_offset,
    size_t threads_per_group
) {
    if (!device || !pipeline || !indirect_buffer) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    if (!device->caps.supports_indirect_dispatch) {
        return MTL_ERROR_UNSUPPORTED;
    }
    
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = mtl_create_command_buffer(device);
        if (!commandBuffer) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        id<MTLComputeCommandEncoder> computeEncoder = mtl_create_compute_encoder(commandBuffer);
        if (!computeEncoder) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        [computeEncoder setComputePipelineState:pipeline->pipelineState];
        
        // Bind resources using shared helper
        mtl_set_buffers(computeEncoder, buffers, buffer_count);
        
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        
        // Indirect dispatch (GPU-driven)
        [computeEncoder dispatchThreadgroupsWithIndirectBuffer:indirect_buffer->buffer
                                              indirectBufferOffset:indirect_offset
                                             threadsPerThreadgroup:tg];
        
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if ([commandBuffer error]) {
            MTL_LOG("Indirect dispatch failed: %s", [[commandBuffer error] localizedDescription].UTF8String);
            return MTL_ERROR_EXECUTION;
        }
        
        return MTL_SUCCESS;
    }
}

