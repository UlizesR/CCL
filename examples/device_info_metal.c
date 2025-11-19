// examples/device_info_metal.c
// Example demonstrating device information queries

#include "ccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int main(void) {
    printf("=== CCL Device Information Example ===\n\n");
    
    // Create context
    ccl_context *ctx = NULL;
    if (ccl_create_context(CCL_BACKEND_METAL, &ctx) != CCL_OK) {
        fprintf(stderr, "Failed to create Metal CCL context\n");
        return 1;
    }
    
    printf("--- Device Information ---\n");
    
    // Query device name
    char deviceName[256];
    size_t nameSize = sizeof(deviceName);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_NAME, deviceName, &nameSize) == CCL_OK) {
        printf("Device Name: %s\n", deviceName);
    }
    
    // Query max threads per threadgroup
    uint64_t maxThreads = 0;
    size_t size = sizeof(maxThreads);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_MAX_THREADS_PER_THREADGROUP, 
                           &maxThreads, &size) == CCL_OK) {
        printf("Max Threads Per Threadgroup: %" PRIu64 "\n", maxThreads);
    }
    
    // Query thread execution width (SIMD width)
    uint64_t simdWidth = 0;
    size = sizeof(simdWidth);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_THREAD_EXECUTION_WIDTH, 
                           &simdWidth, &size) == CCL_OK) {
        printf("Thread Execution Width (SIMD): %" PRIu64 "\n", simdWidth);
    }
    
    // Query max buffer length
    uint64_t maxBuffer = 0;
    size = sizeof(maxBuffer);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_MAX_BUFFER_LENGTH, 
                           &maxBuffer, &size) == CCL_OK) {
        printf("Max Buffer Length: %" PRIu64 " bytes (%.2f MB)\n", 
               maxBuffer, maxBuffer / (1024.0 * 1024.0));
    }
    
    // Query GPU-only buffer support
    bool supportsGPUOnly = false;
    size = sizeof(supportsGPUOnly);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_SUPPORTS_GPU_ONLY_BUFFERS, 
                           &supportsGPUOnly, &size) == CCL_OK) {
        printf("Supports GPU-Only Buffers: %s\n", supportsGPUOnly ? "Yes" : "No");
    }
    
    // Query max compute units
    uint64_t computeUnits = 0;
    size = sizeof(computeUnits);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_MAX_COMPUTE_UNITS, 
                           &computeUnits, &size) == CCL_OK && computeUnits > 0) {
        printf("Max Compute Units: %" PRIu64 "\n", computeUnits);
    }
    
    printf("\n--- Notes ---\n");
    printf("Note: Max threads per threadgroup and SIMD width are per-pipeline\n");
    printf("properties in Metal. The values shown are reasonable defaults.\n");
    printf("Actual values depend on the specific kernel being used.\n");
    
    ccl_destroy_context(ctx);
    return 0;
}

