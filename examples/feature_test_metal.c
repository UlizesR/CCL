// examples/feature_test_metal.c
// Comprehensive test of all CCL Metal backend features

#include "ccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>

// Test kernel source
static const char *test_kernel_source = 
    "kernel void test_kernel(\n"
    "    device const float* a [[ buffer(0) ]],\n"
    "    device const float* b [[ buffer(1) ]],\n"
    "    device float* c [[ buffer(2) ]],\n"
    "    constant uint* n [[ buffer(3) ]],\n"
    "    uint gid [[ thread_position_in_grid ]]\n"
    ") {\n"
    "    uint size = n[0];\n"
    "    if (gid >= size) return;\n"
    "    c[gid] = a[gid] + b[gid];\n"
    "}\n";

static void test_gpu_only_buffers(ccl_context *ctx) {
    printf("\n=== Test 1: GPU_ONLY Buffers ===\n");
    
    const size_t n = 1024;
    const size_t size = n * sizeof(float);
    
    // Create test data
    float *a_data = (float *)malloc(size);
    float *b_data = (float *)malloc(size);
    float *new_a = (float *)malloc(size);
    float *result = (float *)malloc(size);
    
    if (!a_data || !b_data || !new_a || !result) {
        printf("❌ Memory allocation failed\n");
        free(a_data);
        free(b_data);
        free(new_a);
        free(result);
        return;
    }
    
    for (size_t i = 0; i < n; ++i) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i * 2);
        new_a[i] = (float)(i + 100);
    }
    
    // Create GPU_ONLY buffers
    ccl_buffer *a_gpu = NULL, *b_gpu = NULL, *c_gpu = NULL;
    ccl_error err;
    
    err = ccl_create_buffer_ex(ctx, size, CCL_BUFFER_READ, CCL_BUFFER_USAGE_GPU_ONLY, a_data, &a_gpu);
    if (err != CCL_OK || !a_gpu) {
        printf("❌ Failed to create GPU_ONLY buffer A\n");
        goto cleanup;
    }
    
    err = ccl_create_buffer_ex(ctx, size, CCL_BUFFER_READ, CCL_BUFFER_USAGE_GPU_ONLY, b_data, &b_gpu);
    if (err != CCL_OK || !b_gpu) {
        printf("❌ Failed to create GPU_ONLY buffer B\n");
        goto cleanup;
    }
    
    err = ccl_create_buffer_ex(ctx, size, CCL_BUFFER_WRITE, CCL_BUFFER_USAGE_GPU_ONLY, NULL, &c_gpu);
    if (err != CCL_OK || !c_gpu) {
        printf("❌ Failed to create GPU_ONLY buffer C\n");
        goto cleanup;
    }
    printf("✓ Created 3 GPU_ONLY buffers\n");
    
    // Test upload after creation
    err = ccl_buffer_upload_ex(ctx, a_gpu, 0, new_a, size);
    if (err != CCL_OK) {
        printf("❌ Failed to upload to GPU_ONLY buffer\n");
        goto cleanup;
    }
    printf("✓ Uploaded data to GPU_ONLY buffer\n");
    
    // Create kernel
    ccl_kernel *kernel = NULL;
    char log[512];
    err = ccl_create_kernel_from_source(ctx, test_kernel_source, "test_kernel", &kernel, log, sizeof(log));
    if (err != CCL_OK || !kernel) {
        printf("❌ Failed to create kernel: %s\n", log);
        goto cleanup;
    }
    
    // Set uniform
    uint32_t n_val = (uint32_t)n;
    err = ccl_set_bytes(kernel, 3, &n_val, sizeof(n_val));
    if (err != CCL_OK) {
        printf("❌ Failed to set uniform\n");
        goto cleanup;
    }
    printf("✓ Set uniform parameter\n");
    
    // Dispatch
    ccl_buffer *buffers[] = { a_gpu, b_gpu, c_gpu };
    err = ccl_dispatch_1d(ctx, kernel, n, 0, buffers, 3);
    if (err != CCL_OK) {
        printf("❌ Dispatch failed\n");
        goto cleanup;
    }
    printf("✓ Dispatched kernel\n");
    
    // Download result
    err = ccl_buffer_download_ex(ctx, c_gpu, 0, result, size);
    if (err != CCL_OK) {
        printf("❌ Failed to download from GPU_ONLY buffer\n");
        goto cleanup;
    }
    printf("✓ Downloaded result from GPU_ONLY buffer\n");
    
    // Verify (using new_a + b_data as expected)
    bool match = true;
    for (size_t i = 0; i < n; ++i) {
        float expected_val = new_a[i] + b_data[i];
        if (fabsf(result[i] - expected_val) > 0.001f) {
            printf("❌ Mismatch at index %zu: got %f, expected %f\n", i, result[i], expected_val);
            match = false;
            break;
        }
    }
    
    printf(match ? "✓ Results match!\n" : "❌ Results don't match!\n");
    
cleanup:
    if (a_gpu) ccl_destroy_buffer(a_gpu);
    if (b_gpu) ccl_destroy_buffer(b_gpu);
    if (c_gpu) ccl_destroy_buffer(c_gpu);
    if (kernel) ccl_destroy_kernel(kernel);
    free(a_data);
    free(b_data);
    free(new_a);
    free(result);
}

static void test_batching(ccl_context *ctx) {
    printf("\n=== Test 2: Command Buffer Batching ===\n");
    
    const size_t n = 512;
    const size_t size = n * sizeof(float);
    
    // Create shared buffers
    float *a_data = (float *)malloc(size);
    float *b_data = (float *)malloc(size);
    float *r1 = (float *)malloc(size);
    float *r2 = (float *)malloc(size);
    
    if (!a_data || !b_data || !r1 || !r2) {
        printf("❌ Memory allocation failed\n");
        free(a_data);
        free(b_data);
        free(r1);
        free(r2);
        return;
    }
    
    for (size_t i = 0; i < n; ++i) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i * 2);
    }
    
    ccl_buffer *a = NULL, *b = NULL, *c1 = NULL, *c2 = NULL;
    ccl_error err;
    
    err = ccl_create_buffer(ctx, size, CCL_BUFFER_READ, a_data, &a);
    if (err != CCL_OK || !a) goto cleanup;
    
    err = ccl_create_buffer(ctx, size, CCL_BUFFER_READ, b_data, &b);
    if (err != CCL_OK || !b) goto cleanup;
    
    err = ccl_create_buffer(ctx, size, CCL_BUFFER_WRITE, NULL, &c1);
    if (err != CCL_OK || !c1) goto cleanup;
    
    err = ccl_create_buffer(ctx, size, CCL_BUFFER_WRITE, NULL, &c2);
    if (err != CCL_OK || !c2) goto cleanup;
    
    // Create kernel
    ccl_kernel *kernel = NULL;
    char log[512];
    err = ccl_create_kernel_from_source(ctx, test_kernel_source, "test_kernel", &kernel, log, sizeof(log));
    if (err != CCL_OK || !kernel) {
        printf("❌ Failed to create kernel: %s\n", log);
        goto cleanup;
    }
    
    uint32_t n_val = (uint32_t)n;
    err = ccl_set_bytes(kernel, 3, &n_val, sizeof(n_val));
    if (err != CCL_OK) goto cleanup;
    
    // Test batching
    err = ccl_begin_batch(ctx);
    if (err != CCL_OK) {
        printf("❌ Failed to begin batch\n");
        goto cleanup;
    }
    printf("✓ Began batch\n");
    
    // First dispatch
    ccl_buffer *bufs1[] = { a, b, c1 };
    err = ccl_dispatch_1d(ctx, kernel, n, 0, bufs1, 3);
    if (err != CCL_OK) {
        printf("❌ First dispatch failed\n");
        goto cleanup;
    }
    printf("✓ Added first dispatch to batch\n");
    
    // Second dispatch
    ccl_buffer *bufs2[] = { a, b, c2 };
    err = ccl_dispatch_1d(ctx, kernel, n, 0, bufs2, 3);
    if (err != CCL_OK) {
        printf("❌ Second dispatch failed\n");
        goto cleanup;
    }
    printf("✓ Added second dispatch to batch\n");
    
    // End batch and get fence
    ccl_fence *fence = NULL;
    err = ccl_end_batch(ctx, &fence);
    if (err != CCL_OK) {
        printf("❌ Failed to end batch\n");
        goto cleanup;
    }
    printf("✓ Ended batch, got fence\n");
    
    // Wait for completion
    ccl_fence_wait(fence);
    printf("✓ Batch completed\n");
    
    // Verify results
    ccl_buffer_download(c1, 0, r1, size);
    ccl_buffer_download(c2, 0, r2, size);
    
    bool match = true;
    for (size_t i = 0; i < n; ++i) {
        float expected = a_data[i] + b_data[i];
        if (fabsf(r1[i] - expected) > 0.001f || fabsf(r2[i] - expected) > 0.001f) {
            printf("❌ Result mismatch at index %zu\n", i);
            match = false;
            break;
        }
    }
    
    printf(match ? "✓ Both batched dispatches produced correct results!\n" 
                 : "❌ Results don't match!\n");
    
    ccl_fence_destroy(fence);
    
cleanup:
    if (a) ccl_destroy_buffer(a);
    if (b) ccl_destroy_buffer(b);
    if (c1) ccl_destroy_buffer(c1);
    if (c2) ccl_destroy_buffer(c2);
    if (kernel) ccl_destroy_kernel(kernel);
    free(a_data);
    free(b_data);
    free(r1);
    free(r2);
}

static void test_uniforms(ccl_context *ctx) {
    printf("\n=== Test 3: Uniforms/Constants ===\n");
    
    const size_t n = 256;
    const size_t size = n * sizeof(float);
    
    float *a_data = (float *)malloc(size);
    float *b_data = (float *)malloc(size);
    float *result = (float *)malloc(size);
    
    if (!a_data || !b_data || !result) {
        printf("❌ Memory allocation failed\n");
        free(a_data);
        free(b_data);
        free(result);
        return;
    }
    
    for (size_t i = 0; i < n; ++i) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i * 2);
    }
    
    ccl_buffer *a = NULL, *b = NULL, *c = NULL;
    ccl_error err;
    
    err = ccl_create_buffer(ctx, size, CCL_BUFFER_READ, a_data, &a);
    if (err != CCL_OK || !a) goto cleanup;
    
    err = ccl_create_buffer(ctx, size, CCL_BUFFER_READ, b_data, &b);
    if (err != CCL_OK || !b) goto cleanup;
    
    err = ccl_create_buffer(ctx, size, CCL_BUFFER_WRITE, NULL, &c);
    if (err != CCL_OK || !c) goto cleanup;
    
    ccl_kernel *kernel = NULL;
    char log[512];
    err = ccl_create_kernel_from_source(ctx, test_kernel_source, "test_kernel", &kernel, log, sizeof(log));
    if (err != CCL_OK || !kernel) {
        printf("❌ Failed to create kernel: %s\n", log);
        goto cleanup;
    }
    
    // Set uniform - should persist across dispatches
    uint32_t n_val = (uint32_t)n;
    err = ccl_set_bytes(kernel, 3, &n_val, sizeof(n_val));
    if (err != CCL_OK) goto cleanup;
    printf("✓ Set uniform parameter\n");
    
    // First dispatch
    ccl_buffer *bufs[] = { a, b, c };
    err = ccl_dispatch_1d(ctx, kernel, n, 0, bufs, 3);
    if (err != CCL_OK) goto cleanup;
    printf("✓ First dispatch (with uniform)\n");
    
    // Clear and set different value
    ccl_clear_bytes(kernel);
    uint32_t n_half = (uint32_t)(n / 2);
    err = ccl_set_bytes(kernel, 3, &n_half, sizeof(n_half));
    if (err != CCL_OK) goto cleanup;
    printf("✓ Cleared and set new uniform\n");
    
    // Second dispatch with different uniform
    err = ccl_dispatch_1d(ctx, kernel, n, 0, bufs, 3);
    if (err != CCL_OK) goto cleanup;
    printf("✓ Second dispatch (with new uniform)\n");
    
    // Verify result
    ccl_buffer_download(c, 0, result, size);
    
    bool match = true;
    for (size_t i = 0; i < n / 2; ++i) {
        float expected = a_data[i] + b_data[i];
        if (fabsf(result[i] - expected) > 0.001f) {
            match = false;
            break;
        }
    }
    
    printf(match ? "✓ Uniforms work correctly across dispatches!\n" : "❌ Uniform behavior incorrect\n");
    
cleanup:
    if (a) ccl_destroy_buffer(a);
    if (b) ccl_destroy_buffer(b);
    if (c) ccl_destroy_buffer(c);
    if (kernel) ccl_destroy_kernel(kernel);
    free(a_data);
    free(b_data);
    free(result);
}

static void test_async_and_fence_error(ccl_context *ctx) {
    printf("\n=== Test 4: Async Dispatch & Fence Error Handling ===\n");
    
    const size_t n = 128;
    const size_t size = n * sizeof(float);
    
    float *a_data = (float *)malloc(size);
    float *b_data = (float *)malloc(size);
    for (size_t i = 0; i < n; ++i) {
        a_data[i] = (float)i;
        b_data[i] = (float)(i * 2);
    }
    
    ccl_buffer *a = NULL, *b = NULL, *c = NULL;
    ccl_create_buffer(ctx, size, CCL_BUFFER_READ, a_data, &a);
    ccl_create_buffer(ctx, size, CCL_BUFFER_READ, b_data, &b);
    ccl_create_buffer(ctx, size, CCL_BUFFER_WRITE, NULL, &c);
    
    ccl_kernel *kernel = NULL;
    char log[512];
    ccl_create_kernel_from_source(ctx, test_kernel_source, "test_kernel",
                                  &kernel, log, sizeof(log));
    
    uint32_t n_val = (uint32_t)n;
    ccl_set_bytes(kernel, 3, &n_val, sizeof(n_val));
    
    // Async dispatch
    ccl_buffer *bufs[] = { a, b, c };
    ccl_fence *fence = NULL;
    ccl_error err = ccl_dispatch_1d_async(ctx, kernel, n, 0, bufs, 3, &fence);
    if (err != CCL_OK || !fence) {
        printf("❌ Failed to get fence from async dispatch: %d\n", err);
        goto cleanup;
    }
    printf("✓ Got fence from async dispatch\n");
    
    // Check completion (non-blocking)
    bool complete = ccl_fence_is_complete(fence);
    printf("✓ Fence initially complete: %s\n", complete ? "yes" : "no");
    
    // Wait for completion
    ccl_fence_wait(fence);
    printf("✓ Waited for fence completion\n");
    
    // Check for errors
    const char *error_msg = ccl_fence_get_error_message(fence);
    if (error_msg) {
        printf("❌ Fence had error: %s\n", error_msg);
    } else {
        printf("✓ No errors reported by fence\n");
    }
    
    // Verify result
    float *result = (float *)malloc(size);
    ccl_buffer_download(c, 0, result, size);
    
    bool match = true;
    for (size_t i = 0; i < n; ++i) {
        float expected = a_data[i] + b_data[i];
        if (fabsf(result[i] - expected) > 0.001f) {
            match = false;
            break;
        }
    }
    
    if (match) {
        printf("✓ Async dispatch produced correct result!\n");
    }
    
    ccl_fence_destroy(fence);
    
cleanup:
    if (a) ccl_destroy_buffer(a);
    if (b) ccl_destroy_buffer(b);
    if (c) ccl_destroy_buffer(c);
    if (kernel) ccl_destroy_kernel(kernel);
    free(a_data);
    free(b_data);
    free(result);
}

static void log_callback(const char *msg, void *user_data) {
    (void)user_data;
    fprintf(stderr, "[CCL] %s\n", msg);
}

int main(void) {
    printf("=== CCL Metal Backend Feature Test Suite ===\n");
    
    // Create context
    ccl_context *ctx = NULL;
    if (ccl_create_context(CCL_BACKEND_METAL, &ctx) != CCL_OK) {
        fprintf(stderr, "Failed to create Metal context\n");
        return 1;
    }
    
    // Set log callback
    ccl_set_log_callback(ctx, log_callback, NULL);
    printf("✓ Set up log callback\n");
    
    // Run tests
    test_gpu_only_buffers(ctx);
    test_batching(ctx);
    test_uniforms(ctx);
    test_async_and_fence_error(ctx);
    
    printf("\n=== All Tests Complete ===\n");
    
    ccl_destroy_context(ctx);
    return 0;
}

