/**
 * MTL Dispatch & Encoding
 * Unified dispatch implementation - all dispatch variants use this module
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "mtl_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// COMMAND BUFFER HELPERS (shared across dispatch modules)
// ============================================================================

id<MTLCommandBuffer> mtl_create_command_buffer(MTLComputeDevice* device) {
    if (!device || !device->commandQueue) {
        return nil;
    }
    return [device->commandQueue commandBuffer];
}

id<MTLComputeCommandEncoder> mtl_create_compute_encoder(id<MTLCommandBuffer> commandBuffer) {
    if (!commandBuffer) {
        return nil;
    }
    return [commandBuffer computeCommandEncoder];
}

// ============================================================================
// RESOURCE BINDING HELPERS
// ============================================================================

void mtl_set_buffers(id<MTLComputeCommandEncoder> encoder, MTLComputeBuffer** buffers, size_t count) {
    if (!encoder || !buffers || count == 0) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        if (buffers[i] && buffers[i]->buffer) {
            [encoder setBuffer:buffers[i]->buffer offset:0 atIndex:i];
        }
    }
}

void mtl_set_textures(id<MTLComputeCommandEncoder> encoder, MTLComputeTexture** textures, size_t count) {
    if (!encoder || !textures || count == 0) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        if (textures[i] && textures[i]->texture) {
            [encoder setTexture:textures[i]->texture atIndex:i];
        }
    }
}

void mtl_set_samplers(id<MTLComputeCommandEncoder> encoder, MTLComputeSampler** samplers, size_t count) {
    if (!encoder || !samplers || count == 0) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        if (samplers[i] && samplers[i]->samplerState) {
            [encoder setSamplerState:samplers[i]->samplerState atIndex:i];
        }
    }
}

// ============================================================================
// UNIFIED CORE ENCODER - Single source of truth for all dispatches
// ============================================================================

MTLComputeError mtl_encode_core(id<MTLComputeCommandEncoder> enc, const MTLComputeDispatchDesc* desc) {
    if (!enc || !desc || !desc->pipeline || !desc->pipeline->pipelineState) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    // Validate grid dimensions
    if (desc->grid_width == 0 || desc->grid_height == 0 || desc->grid_depth == 0) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    // Null check for resource arrays (consistent validation)
    if (desc->buffer_count > 0 && !desc->buffers) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    if (desc->texture_count > 0 && !desc->textures) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    if (desc->sampler_count > 0 && !desc->samplers) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    // Set pipeline state
    [enc setComputePipelineState:desc->pipeline->pipelineState];
    
    // Bind all resources (unified path - buffer/texture/sampler triad)
    mtl_set_buffers(enc, desc->buffers, desc->buffer_count);
    mtl_set_textures(enc, desc->textures, desc->texture_count);
    mtl_set_samplers(enc, desc->samplers, desc->sampler_count);
    
    // Auto-select threadgroup size if not specified (0 = auto)
    size_t tg_w = desc->threadgroup_width;
    size_t tg_h = desc->threadgroup_height;
    size_t tg_d = desc->threadgroup_depth;
    
    if (tg_w == 0 || tg_h == 0 || tg_d == 0) {
        // Auto mode - pick optimal based on grid dimensionality
        if (desc->grid_depth > 1) {
            // 3D dispatch - use conservative defaults
            tg_w = tg_w ? tg_w : 4;
            tg_h = tg_h ? tg_h : 4;
            tg_d = tg_d ? tg_d : 4;
        } else if (desc->grid_height > 1) {
            // 2D dispatch - optimal for image processing
            if (tg_w == 0 || tg_h == 0) {
                mtl_compute_auto_threadgroup_2d(desc->pipeline, desc->grid_width, desc->grid_height,
                                               &tg_w, &tg_h);
            }
            tg_d = 1;
        } else {
            // 1D dispatch
            if (tg_w == 0) {
                size_t dummy;
                mtl_compute_auto_threadgroup_1d(desc->pipeline, desc->grid_width, &tg_w, &dummy);
            }
            tg_h = 1;
            tg_d = 1;
        }
    }
    
    // Dispatch
    MTLSize grid = MTLSizeMake(desc->grid_width, desc->grid_height, desc->grid_depth);
    MTLSize tg   = MTLSizeMake(tg_w, tg_h, tg_d);
    
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    
    return MTL_SUCCESS;
}

// ============================================================================
// AUTO THREADGROUP SIZING
// ============================================================================

void mtl_compute_auto_threadgroup_1d(
    MTLComputePipeline* pipeline,
    size_t total_threads,
    size_t* threads_per_group,
    size_t* num_groups
) {
    if (!pipeline || !threads_per_group || !num_groups) {
        return;
    }
    
    size_t max_threads = mtl_compute_pipeline_max_threads_per_threadgroup(pipeline);
    if (max_threads == 0) {
        max_threads = 256;
    }
    
    // For 1D, prefer power-of-2 sizes for better occupancy
    size_t optimal_size = 256;
    if (max_threads < 256) {
        optimal_size = max_threads;
    }
    
    // Round down to nearest power of 2
    while (optimal_size > 1 && optimal_size > max_threads) {
        optimal_size >>= 1;
    }
    
    *threads_per_group = optimal_size;
    *num_groups = (total_threads + optimal_size - 1) / optimal_size;
}

void mtl_compute_auto_threadgroup_2d(
    MTLComputePipeline* pipeline,
    size_t grid_width,
    size_t grid_height,
    size_t* threadgroup_width,
    size_t* threadgroup_height
) {
    if (!pipeline || !threadgroup_width || !threadgroup_height) {
        return;
    }
    
    size_t max_threads = mtl_compute_pipeline_max_threads_per_threadgroup(pipeline);
    if (max_threads == 0) {
        max_threads = 256;
    }
    
    // For 2D image processing, 16x16 is typically optimal
    size_t tw = 16;
    size_t th = 16;
    
    // Adjust if max threads is lower
    while (tw * th > max_threads && (tw > 1 || th > 1)) {
        if (tw > th) {
            tw >>= 1;
        } else {
            th >>= 1;
        }
    }
    
    // Adjust for small images
    if (grid_width < tw) tw = grid_width;
    if (grid_height < th) th = grid_height;
    
    *threadgroup_width = tw;
    *threadgroup_height = th;
}

// ============================================================================
// ENCODER-BASED API (Tier 3)
// ============================================================================

MTLComputeError mtl_compute_begin(
    MTLComputeDevice* device,
    MTLComputeCommandList** out_command_list
) {
    if (!device || !out_command_list) {
        return MTL_ERROR_INVALID_PARAMETER;
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
        
        MTLComputeCommandList* cmdList = (MTLComputeCommandList*)malloc(sizeof(MTLComputeCommandList));
        if (!cmdList) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        cmdList->commandBuffer = commandBuffer;
        cmdList->computeEncoder = computeEncoder;
        cmdList->device = device;
        
        *out_command_list = cmdList;
        return MTL_SUCCESS;
    }
}

MTLComputeError mtl_compute_encode_dispatch(
    MTLComputeCommandList* command_list,
    const MTLComputeDispatchDesc* desc
) {
    if (!command_list) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        // Pre-flight validation (optional, enabled in debug builds)
#ifdef DEBUG
        char validation_log[256];
        MTLComputeError validation_err = mtl_compute_validate_dispatch(desc->pipeline, desc, validation_log, sizeof(validation_log));
        if (validation_err != MTL_SUCCESS) {
            MTL_LOG("Validation failed: %s", validation_log);
            return validation_err;
        }
#endif
        
        // Use unified encoder
        return mtl_encode_core(command_list->computeEncoder, desc);
    }
}

MTLComputeError mtl_compute_end_submit(MTLComputeCommandList* command_list) {
    if (!command_list) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        [command_list->computeEncoder endEncoding];
        [command_list->commandBuffer commit];
        [command_list->commandBuffer waitUntilCompleted];
        
        if ([command_list->commandBuffer error]) {
            NSError* error = [command_list->commandBuffer error];
            MTL_LOG("Compute execution failed: %s", [[error localizedDescription] UTF8String]);
            free(command_list);
            return MTL_ERROR_EXECUTION;
        }
        
        free(command_list);
        return MTL_SUCCESS;
    }
}

MTLComputeError mtl_compute_end_submit_nowait(MTLComputeCommandList* command_list) {
    if (!command_list) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        [command_list->computeEncoder endEncoding];
        [command_list->commandBuffer commit];
        
        free(command_list);
        return MTL_SUCCESS;
    }
}

void mtl_compute_command_list_set_label(MTLComputeCommandList* command_list, const char* label) {
    if (!command_list || !label) {
        return;
    }
    
    @autoreleasepool {
        NSString* labelString = [NSString stringWithUTF8String:label];
        [command_list->commandBuffer setLabel:labelString];
    }
}

// ============================================================================
// UNIFIED DESCRIPTOR DISPATCH (Tier 2)
// ============================================================================

MTLComputeError mtl_compute_dispatch_desc(
    MTLComputeDevice* device,
    const MTLComputeDispatchDesc* desc
) {
    if (!device || !desc) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    MTLComputeCommandList* cmdList = NULL;
    MTLComputeError err = mtl_compute_begin(device, &cmdList);
    if (err != MTL_SUCCESS) {
        return err;
    }
    
    err = mtl_compute_encode_dispatch(cmdList, desc);
    if (err != MTL_SUCCESS) {
        free(cmdList);
        return err;
    }
    
    return mtl_compute_end_submit(cmdList);
}

// ============================================================================
// IMMEDIATE DISPATCH (Tier 1) - All build descriptor and forward
// ============================================================================

static MTLComputeError mtl_compute_dispatch_internal(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t grid_width,
    size_t grid_height,
    size_t grid_depth,
    size_t threadgroup_width,
    size_t threadgroup_height,
    size_t threadgroup_depth,
    bool wait
) {
    MTLComputeDispatchDesc desc = {
        .pipeline = pipeline,
        .buffers = buffers,
        .buffer_count = buffer_count,
        .textures = NULL,
        .texture_count = 0,
        .samplers = NULL,
        .sampler_count = 0,
        .grid_width = grid_width,
        .grid_height = grid_height,
        .grid_depth = grid_depth,
        .threadgroup_width = threadgroup_width,
        .threadgroup_height = threadgroup_height,
        .threadgroup_depth = threadgroup_depth
    };
    
    MTLComputeCommandList* cmdList = NULL;
    MTLComputeError err = mtl_compute_begin(device, &cmdList);
    if (err != MTL_SUCCESS) {
        return err;
    }
    
    err = mtl_compute_encode_dispatch(cmdList, &desc);
    if (err != MTL_SUCCESS) {
        free(cmdList);
        return err;
    }
    
    if (wait) {
        return mtl_compute_end_submit(cmdList);
    } else {
        return mtl_compute_end_submit_nowait(cmdList);
    }
}

MTLComputeError mtl_compute_dispatch_sync(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t grid_width,
    size_t grid_height,
    size_t grid_depth,
    size_t threadgroup_width,
    size_t threadgroup_height,
    size_t threadgroup_depth
) {
    return mtl_compute_dispatch_internal(device, pipeline, buffers, buffer_count,
                                         grid_width, grid_height, grid_depth,
                                         threadgroup_width, threadgroup_height, threadgroup_depth,
                                         true);
}

MTLComputeError mtl_compute_dispatch_nowait(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t grid_width,
    size_t grid_height,
    size_t grid_depth,
    size_t threadgroup_width,
    size_t threadgroup_height,
    size_t threadgroup_depth
) {
    return mtl_compute_dispatch_internal(device, pipeline, buffers, buffer_count,
                                         grid_width, grid_height, grid_depth,
                                         threadgroup_width, threadgroup_height, threadgroup_depth,
                                         false);
}

// Legacy alias
MTLComputeError mtl_compute_dispatch(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t grid_width,
    size_t grid_height,
    size_t grid_depth,
    size_t threadgroup_width,
    size_t threadgroup_height,
    size_t threadgroup_depth
) {
    return mtl_compute_dispatch_sync(device, pipeline, buffers, buffer_count,
                                     grid_width, grid_height, grid_depth,
                                     threadgroup_width, threadgroup_height, threadgroup_depth);
}

MTLComputeError mtl_compute_dispatch_1d(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t total_threads,
    size_t threads_per_group
) {
    return mtl_compute_dispatch_sync(
        device, pipeline, buffers, buffer_count,
        total_threads, 1, 1,
        threads_per_group, 1, 1
    );
}

// ============================================================================
// ADVANCED DISPATCH VARIANTS (Tier 4)
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
        id<MTLCommandBuffer> commandBuffer = [device->commandQueue commandBuffer];
        if (!commandBuffer) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        if (!computeEncoder) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        // Use unified encoder via descriptor
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
        
        id<MTLCommandBuffer> commandBuffer = [device->commandQueue commandBuffer];
        if (!commandBuffer) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        if (!computeEncoder) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        // Use unified encoder
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
        id<MTLCommandBuffer> commandBuffer = [device->commandQueue commandBuffer];
        if (!commandBuffer) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        if (!computeEncoder) {
            return MTL_ERROR_COMMAND_ENCODING;
        }
        
        [computeEncoder setComputePipelineState:pipeline->pipelineState];
        
        // Bind resources
        mtl_set_buffers(computeEncoder, buffers, buffer_count);
        
        MTLSize tg = MTLSizeMake(threads_per_group, 1, 1);
        
        // Indirect dispatch
        [computeEncoder dispatchThreadgroupsWithIndirectBuffer:indirect_buffer->buffer
                                              indirectBufferOffset:indirect_offset
                                             threadsPerThreadgroup:tg];
        
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if ([commandBuffer error]) {
            return MTL_ERROR_EXECUTION;
        }
        
        return MTL_SUCCESS;
    }
}

// ============================================================================
// VALIDATION
// ============================================================================

MTLComputeError mtl_compute_validate_dispatch(
    MTLComputePipeline* pipeline,
    const MTLComputeDispatchDesc* desc,
    char* error_log,
    size_t error_log_size
) {
    if (!pipeline || !desc) {
        mtl_copy_error_log("NULL pipeline or descriptor", error_log, error_log_size);
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    // Validate threadgroup size
    if (desc->threadgroup_width > 0 && desc->threadgroup_height > 0 && desc->threadgroup_depth > 0) {
        if (!mtl_compute_pipeline_validate_threadgroup_size(pipeline, 
            desc->threadgroup_width, desc->threadgroup_height, desc->threadgroup_depth)) {
            mtl_copy_error_log("Threadgroup size exceeds pipeline maximum", error_log, error_log_size);
            return MTL_ERROR_INVALID_PARAMETER;
        }
    }
    
    // TODO: Validate resource counts against reflection data
    
    return MTL_SUCCESS;
}

bool mtl_compute_pipeline_validate_threadgroup_size(
    MTLComputePipeline* pipeline,
    size_t tg_width,
    size_t tg_height,
    size_t tg_depth
) {
    if (!pipeline || !pipeline->pipelineState) {
        return false;
    }
    
    // Check against max
    size_t max_total = mtl_compute_pipeline_max_threads_per_threadgroup(pipeline);
    if (tg_width * tg_height * tg_depth > max_total) {
        return false;
    }
    
    // TODO: Check against [[required_threads_per_threadgroup]] via reflection
    
    return true;
}

