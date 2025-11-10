/**
 * MTL Device Management
 * Device creation, capabilities detection, and device-level utilities
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "mtl_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Detect device capabilities at runtime
void mtl_detect_capabilities(MTLComputeDevice* ctx) {
    @autoreleasepool {
        id<MTLDevice> device = ctx->device;
        MTLComputeDeviceCapabilities* caps = &ctx->caps;
        
        // Clear all
        memset(caps, 0, sizeof(MTLComputeDeviceCapabilities));
        
        // Metal version support (runtime check via selectors)
        caps->supports_metal_3 = [device respondsToSelector:@selector(supportsFamily:)] &&
                                 [device supportsFamily:MTLGPUFamilyMetal3];
        caps->supports_metal_4 = false; // Metal 4 detection when available
        
        // Core features
#if TARGET_OS_OSX
        caps->supports_managed_storage = true;
#else
        caps->supports_managed_storage = false;
#endif
        caps->supports_non_uniform_threadgroups = true; // Metal 2+
        
        // Advanced features (runtime detection)
        caps->supports_shared_events = [device respondsToSelector:@selector(newSharedEvent)];
        caps->supports_binary_archives = [device respondsToSelector:@selector(newBinaryArchiveWithDescriptor:error:)];
        caps->supports_heaps = [device respondsToSelector:@selector(newHeapWithDescriptor:)];
        caps->supports_indirect_dispatch = true; // Available since Metal 1.2
        caps->supports_function_pointers = caps->supports_metal_3;
        caps->supports_argument_buffers = [device argumentBuffersSupport] != MTLArgumentBuffersTier1;
        caps->supports_indirect_command_buffers = [device respondsToSelector:@selector(newIndirectCommandBufferWithDescriptor:maxCommandCount:options:)];
        
        // SIMD-group matrix support (Metal 3+, Apple7+)
        if (@available(macOS 13.0, iOS 16.0, *)) {
            caps->supports_simdgroup_matrix = [device supportsFamily:MTLGPUFamilyApple7];
        } else {
            caps->supports_simdgroup_matrix = false;
        }
        
        // Ray tracing (Metal 3+)
        if (@available(macOS 13.0, iOS 16.0, *)) {
            caps->supports_raytracing = [device supportsRaytracing];
        } else {
            caps->supports_raytracing = false;
        }
        
        // Device limits
        caps->max_threadgroup_memory = [device maxThreadgroupMemoryLength];
        
        MTLSize maxThreads = [device maxThreadsPerThreadgroup];
        caps->max_threads_per_threadgroup = maxThreads.width * maxThreads.height * maxThreads.depth;
        caps->recommended_max_working_set_size = [device recommendedMaxWorkingSetSize];
        caps->device_name = [[device name] UTF8String];
    }
}

// Device management
MTLComputeDevice* mtl_compute_device_create(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            MTL_LOG("Metal is not supported on this device");
            return NULL;
        }
        
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            MTL_LOG("Failed to create command queue");
            return NULL;
        }
        
        MTLComputeDevice* ctx = (MTLComputeDevice*)malloc(sizeof(MTLComputeDevice));
        if (!ctx) {
            return NULL;
        }
        
        ctx->device = device;
        ctx->commandQueue = commandQueue;
        ctx->defaultSharedEvent = nil;
        ctx->defaultArchive = nil;
        
        // Detect capabilities
        mtl_detect_capabilities(ctx);
        
        // Create default shared event if supported
        if (ctx->caps.supports_shared_events) {
            ctx->defaultSharedEvent = [device newSharedEvent];
        }
        
        return ctx;
    }
}

MTLComputeError mtl_compute_device_create_with_index(
    uint32_t device_index,
    MTLComputeDevice** out_device
) {
    if (!out_device) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        
        if (device_index >= [devices count]) {
            return MTL_ERROR_NO_DEVICE;
        }
        
        id<MTLDevice> device = devices[device_index];
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            return MTL_ERROR_NO_DEVICE;
        }
        
        MTLComputeDevice* ctx = (MTLComputeDevice*)malloc(sizeof(MTLComputeDevice));
        if (!ctx) {
            return MTL_ERROR_NO_DEVICE;
        }
        
        ctx->device = device;
        ctx->commandQueue = commandQueue;
        ctx->defaultSharedEvent = nil;
        ctx->defaultArchive = nil;
        
        mtl_detect_capabilities(ctx);
        
        if (ctx->caps.supports_shared_events) {
            ctx->defaultSharedEvent = [device newSharedEvent];
        }
        
        *out_device = ctx;
        return MTL_SUCCESS;
    }
}

void mtl_compute_device_destroy(MTLComputeDevice* device) {
    if (device) {
        free(device);
    }
}

const char* mtl_compute_device_get_name(MTLComputeDevice* device) {
    if (!device || !device->device) {
        return "Unknown";
    }
    @autoreleasepool {
        return [[device->device name] UTF8String];
    }
}

void* mtl_compute_device_get_mtl_device(MTLComputeDevice* device) {
    if (!device) {
        return NULL;
    }
    return (__bridge void*)device->device;
}

bool mtl_compute_device_is_managed_supported(MTLComputeDevice* device) {
    if (!device) {
        return false;
    }
#if TARGET_OS_OSX
    return true;
#else
    return false;
#endif
}

void mtl_compute_device_get_capabilities(
    MTLComputeDevice* device,
    MTLComputeDeviceCapabilities* caps
) {
    if (!device || !caps) {
        return;
    }
    *caps = device->caps;
}

void mtl_compute_device_print_features(MTLComputeDevice* device) {
    if (!device) {
        return;
    }
    
    MTLComputeDeviceCapabilities* caps = &device->caps;
    
    printf("\n========== MTLComp Device Capabilities ==========\n");
    printf("Device: %s\n", caps->device_name);
    printf("\n--- Metal Version Support ---\n");
    printf("  Metal 3: %s\n", caps->supports_metal_3 ? "✓" : "✗");
    printf("  Metal 4: %s\n", caps->supports_metal_4 ? "✓" : "✗");
    
    printf("\n--- Core Features ---\n");
    printf("  Managed Storage: %s\n", caps->supports_managed_storage ? "✓" : "✗");
    printf("  Non-Uniform Threadgroups: %s\n", caps->supports_non_uniform_threadgroups ? "✓" : "✗");
    
    printf("\n--- Advanced Features (MSL Spec) ---\n");
    printf("  Shared Events (async): %s\n", caps->supports_shared_events ? "✓" : "✗");
    printf("  Binary Archives: %s\n", caps->supports_binary_archives ? "✓" : "✗");
    printf("  Memory Heaps: %s\n", caps->supports_heaps ? "✓" : "✗");
    printf("  Indirect Dispatch: %s\n", caps->supports_indirect_dispatch ? "✓" : "✗");
    printf("  Function Pointers (2.15): %s\n", caps->supports_function_pointers ? "✓" : "✗");
    printf("  Argument Buffers (2.13): %s\n", caps->supports_argument_buffers ? "✓" : "✗");
    printf("  Indirect Command Buffers (6.16): %s\n", caps->supports_indirect_command_buffers ? "✓" : "✗");
    printf("  SIMD-group Matrix (6.7): %s\n", caps->supports_simdgroup_matrix ? "✓" : "✗");
    printf("  Ray Tracing (2.17): %s\n", caps->supports_raytracing ? "✓" : "✗");
    
    printf("\n--- Device Limits ---\n");
    printf("  Max Threadgroup Memory: %zu bytes\n", caps->max_threadgroup_memory);
    printf("  Max Threads/Threadgroup: %zu\n", caps->max_threads_per_threadgroup);
    printf("  Recommended Working Set: %zu bytes (%.1f MB)\n", 
           caps->recommended_max_working_set_size,
           caps->recommended_max_working_set_size / (1024.0 * 1024.0));
    printf("================================================\n\n");
}

// Shared events
MTLComputeSharedEvent* mtl_compute_event_create(MTLComputeDevice* device) {
    if (!device || !device->device) {
        return NULL;
    }
    
    if (!device->caps.supports_shared_events) {
        return NULL;
    }
    
    @autoreleasepool {
        id<MTLSharedEvent> event = [device->device newSharedEvent];
        if (!event) {
            return NULL;
        }
        
        MTLComputeSharedEvent* sharedEvent = (MTLComputeSharedEvent*)malloc(sizeof(MTLComputeSharedEvent));
        if (!sharedEvent) {
            return NULL;
        }
        
        sharedEvent->event = event;
        return sharedEvent;
    }
}

bool mtl_compute_event_wait(MTLComputeSharedEvent* event, uint64_t value, uint64_t timeout_ns) {
    if (!event || !event->event) {
        return false;
    }
    
    @autoreleasepool {
        if (timeout_ns == 0) {
            while ([event->event signaledValue] < value) {
                usleep(100);
            }
            return true;
        } else {
            uint64_t start = clock_gettime_nsec_np(CLOCK_MONOTONIC);
            while ([event->event signaledValue] < value) {
                uint64_t elapsed = clock_gettime_nsec_np(CLOCK_MONOTONIC) - start;
                if (elapsed >= timeout_ns) {
                    return false;
                }
                usleep(100);
            }
            return true;
        }
    }
}

bool mtl_compute_event_check(MTLComputeSharedEvent* event, uint64_t value) {
    if (!event || !event->event) {
        return false;
    }
    
    @autoreleasepool {
        return [event->event signaledValue] >= value;
    }
}

void mtl_compute_event_destroy(MTLComputeSharedEvent* event) {
    if (event) {
        free(event);
    }
}

// Error strings
const char* mtl_compute_error_string(MTLComputeError error) {
    static const char* unknown = "Unknown error (code not recognized)";
    
    switch (error) {
        case MTL_SUCCESS:
            return "Success";
        case MTL_ERROR_NO_DEVICE:
            return "No Metal device available";
        case MTL_ERROR_SHADER_COMPILATION:
            return "Shader compilation failed";
        case MTL_ERROR_PIPELINE_CREATION:
            return "Pipeline creation failed";
        case MTL_ERROR_BUFFER_CREATION:
            return "Buffer creation failed";
        case MTL_ERROR_COMMAND_ENCODING:
            return "Command encoding failed";
        case MTL_ERROR_EXECUTION:
            return "Compute execution failed";
        case MTL_ERROR_INVALID_PARAMETER:
            return "Invalid parameter";
        case MTL_ERROR_IO:
            return "File I/O error";
        case MTL_ERROR_UNSUPPORTED:
            return "Feature not supported on this device/OS";
        default:
            return unknown;
    }
}

