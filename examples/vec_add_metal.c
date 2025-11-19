// examples/vec_add_metal.c
// Basic example demonstrating CCL Metal backend
// Features: log callback, device info, async dispatch with error checking

#include "ccl.h"
#include <stdio.h>
#include <stdlib.h>

static void log_callback(const char *msg, void *user_data) {
    (void)user_data;
    fprintf(stderr, "[CCL] %s\n", msg);
}

int main(void) {
    ccl_context *ctx = NULL;
    if (ccl_create_context(CCL_BACKEND_METAL, &ctx) != CCL_OK) {
        fprintf(stderr, "Failed to create Metal CCL context\n");
        return 1;
    }
    
    // Set up log callback for error reporting
    ccl_set_log_callback(ctx, log_callback, NULL);
    
    // Query device info
    char deviceName[256];
    size_t nameSize = sizeof(deviceName);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_NAME, deviceName, &nameSize) == CCL_OK) {
        printf("Device: %s\n", deviceName);
    }

    const size_t N = 1024;
    float *a   = malloc(N * sizeof(float));
    float *b   = malloc(N * sizeof(float));
    float *out = malloc(N * sizeof(float));

    for (size_t i = 0; i < N; ++i) {
        a[i] = (float)i;
        b[i] = (float)(2 * i);
    }

    ccl_buffer *bufA, *bufB, *bufOut;
    ccl_create_buffer(ctx, N * sizeof(float), CCL_BUFFER_READ,      a,   &bufA);
    ccl_create_buffer(ctx, N * sizeof(float), CCL_BUFFER_READ,      b,   &bufB);
    ccl_create_buffer(ctx, N * sizeof(float), CCL_BUFFER_WRITE,     NULL,&bufOut);

    const char *src =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void vec_add(device const float* a [[ buffer(0) ]],\n"
        "                    device const float* b [[ buffer(1) ]],\n"
        "                    device float* out [[ buffer(2) ]],\n"
        "                    uint gid [[ thread_position_in_grid ]]) {\n"
        "    out[gid] = a[gid] + b[gid];\n"
        "}\n";

    ccl_kernel *kernel = NULL;
    char log[4096] = {0};

    ccl_error err = ccl_create_kernel_from_source(
        ctx, src, "vec_add", &kernel, log, sizeof(log)
    );
    if (err != CCL_OK) {
        fprintf(stderr, "Kernel compile failed: %s\n", log);
        return 1;
    }

    ccl_buffer *buffers[3] = { bufA, bufB, bufOut };

    // Use async dispatch with fence for better error reporting
    ccl_fence *fence = NULL;
    err = ccl_dispatch_1d_async(ctx, kernel, N, 0, buffers, 3, &fence);
    if (err != CCL_OK) {
        fprintf(stderr, "Dispatch failed\n");
        return 1;
    }
    
    // Wait for completion
    ccl_fence_wait(fence);
    
    // Check for errors
    const char *error_msg = ccl_fence_get_error_message(fence);
    if (error_msg) {
        fprintf(stderr, "Dispatch error: %s\n", error_msg);
        ccl_fence_destroy(fence);
        return 1;
    }
    
    ccl_buffer_download(bufOut, 0, out, N * sizeof(float));
    ccl_fence_destroy(fence);

    printf("out[10] = %f (expected %f)\n", out[10], a[10] + b[10]);

    ccl_destroy_kernel(kernel);
    ccl_destroy_buffer(bufA);
    ccl_destroy_buffer(bufB);
    ccl_destroy_buffer(bufOut);
    ccl_destroy_context(ctx);

    free(a);
    free(b);
    free(out);
    return 0;
}

