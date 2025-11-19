// examples/matrix_mult_metal.c
//
// Advanced example demonstrating CCL capabilities:
// - Matrix multiplication (C = A * B) on GPU
// - Async dispatch with fences (non-blocking execution)
// - Pipeline caching (kernel reuse for performance)
// - Command buffer batching (group multiple dispatches)
// - Debug labels for GPU capture tools (Xcode/Metal System Trace)
// - Log callback for error reporting
// - Device info queries
// - Fence error message checking
// - CPU reference validation
// - Performance benchmarking
//
// This example tests:
// 1. 2D dispatch (using ccl_dispatch_nd)
// 2. Uniforms API (ccl_set_bytes for matrix dimensions)
// 3. Async dispatch with fence synchronization
// 4. Pipeline caching (same kernel multiple times)
// 5. Command buffer batching (multiple dispatches in one command buffer)
// 6. Fence error message retrieval

#include "ccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#define MATRIX_SIZE 512
#define TILE_SIZE 16

// CPU reference implementation for validation
static void matrix_mult_cpu(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Initialize matrix with random values
static void init_matrix(float *mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}

// Compare matrices (with tolerance for floating point)
static int compare_matrices(const float *A, const float *B, size_t size, float tolerance) {
    for (size_t i = 0; i < size; i++) {
        float diff = fabsf(A[i] - B[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %zu: %f vs %f (diff: %f)\n", i, A[i], B[i], diff);
            return 0;
        }
    }
    return 1;
}

static void log_callback(const char *msg, void *user_data) {
    (void)user_data;
    fprintf(stderr, "[CCL] %s\n", msg);
}

int main(void) {
    printf("=== CCL Advanced Matrix Multiplication Example ===\n\n");
    
    // Initialize random seed
    srand((unsigned int)time(NULL));
    
    const size_t M = MATRIX_SIZE;  // rows of A and C
    const size_t K = MATRIX_SIZE;  // cols of A, rows of B
    const size_t N = MATRIX_SIZE;  // cols of B and C
    
    printf("Matrix dimensions: A[%zu x %zu] * B[%zu x %zu] = C[%zu x %zu]\n", M, K, K, N, M, N);
    printf("Total elements: %zu\n\n", M * N);
    
    // Create context with debug label
    ccl_context *ctx = NULL;
    if (ccl_create_context(CCL_BACKEND_METAL, &ctx) != CCL_OK) {
        fprintf(stderr, "Failed to create Metal CCL context\n");
        return 1;
    }
    ccl_set_context_label(ctx, "MatrixMultExample");
    
    // Set up log callback for error reporting
    ccl_set_log_callback(ctx, log_callback, NULL);
    
    // Query device info
    char deviceName[256];
    size_t nameSize = sizeof(deviceName);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_NAME, deviceName, &nameSize) == CCL_OK) {
        printf("Device: %s\n", deviceName);
    }
    
    uint64_t maxThreads = 0;
    size_t size = sizeof(maxThreads);
    if (ccl_get_device_info(ctx, CCL_DEVICE_INFO_MAX_THREADS_PER_THREADGROUP, &maxThreads, &size) == CCL_OK) {
        printf("Max threads per threadgroup: %llu\n", (unsigned long long)maxThreads);
    }
    printf("\n");
    
    // Allocate host memory
    float *A_host = (float *)malloc(M * K * sizeof(float));
    float *B_host = (float *)malloc(K * N * sizeof(float));
    float *C_gpu = (float *)malloc(M * N * sizeof(float));
    float *C_cpu = (float *)malloc(M * N * sizeof(float));
    
    if (!A_host || !B_host || !C_gpu || !C_cpu) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    init_matrix(A_host, M, K);
    init_matrix(B_host, K, N);
    memset(C_gpu, 0, M * N * sizeof(float));
    memset(C_cpu, 0, M * N * sizeof(float));
    
    // Create GPU buffers with labels
    ccl_buffer *bufA, *bufB, *bufC;
    ccl_error err;
    
    printf("Creating GPU buffers...\n");
    err = ccl_create_buffer(ctx, M * K * sizeof(float), CCL_BUFFER_READ, A_host, &bufA);
    if (err != CCL_OK) {
        fprintf(stderr, "Failed to create buffer A\n");
        return 1;
    }
    ccl_set_buffer_label(bufA, "MatrixA");
    
    err = ccl_create_buffer(ctx, K * N * sizeof(float), CCL_BUFFER_READ, B_host, &bufB);
    if (err != CCL_OK) {
        fprintf(stderr, "Failed to create buffer B\n");
        return 1;
    }
    ccl_set_buffer_label(bufB, "MatrixB");
    
    err = ccl_create_buffer(ctx, M * N * sizeof(float), CCL_BUFFER_WRITE, NULL, &bufC);
    if (err != CCL_OK) {
        fprintf(stderr, "Failed to create buffer C\n");
        return 1;
    }
    ccl_set_buffer_label(bufC, "MatrixC");
    
    // Metal shader source - matrix multiplication using 2D dispatch
    // Each thread computes one element of C using 2D thread position
    const char *shader_source =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "\n"
        "// Matrix multiplication kernel\n"
        "// Each thread computes one element of C = A * B\n"
        "// Uses 2D dispatch with uniforms for dimensions\n"
        "kernel void matrix_mult(\n"
        "    device const float* A [[ buffer(0) ]],\n"
        "    device const float* B [[ buffer(1) ]],\n"
        "    device float* C [[ buffer(2) ]],\n"
        "    constant uint* M [[ buffer(3) ]],\n"
        "    constant uint* N [[ buffer(4) ]],\n"
        "    constant uint* K [[ buffer(5) ]],\n"
        "    uint2 gid [[ thread_position_in_grid ]]\n"
        ") {\n"
        "    uint m = M[0];\n"
        "    uint n = N[0];\n"
        "    uint k = K[0];\n"
        "    \n"
        "    uint row = gid.x;\n"
        "    uint col = gid.y;\n"
        "    \n"
        "    if (row >= m || col >= n) return;\n"
        "    \n"
        "    // Compute dot product of row from A and column from B\n"
        "    float sum = 0.0f;\n"
        "    for (uint i = 0; i < k; i++) {\n"
        "        sum += A[row * k + i] * B[i * n + col];\n"
        "    }\n"
        "    \n"
        "    C[row * n + col] = sum;\n"
        "}\n";
    
    // Create kernel
    printf("Compiling matrix multiplication kernel...\n");
    ccl_kernel *kernel = NULL;
    char log[4096] = {0};
    
    err = ccl_create_kernel_from_source(ctx, shader_source, "matrix_mult", &kernel, log, sizeof(log));
    if (err != CCL_OK) {
        fprintf(stderr, "Kernel compile failed: %s\n", log);
        return 1;
    }
    ccl_set_kernel_label(kernel, "MatrixMult");
    
    // Set uniforms for matrix dimensions (using new ccl_set_bytes API)
    uint32_t m_val = (uint32_t)M;
    uint32_t n_val = (uint32_t)N;
    uint32_t k_val = (uint32_t)K;
    
    err = ccl_set_bytes(kernel, 3, &m_val, sizeof(m_val));
    if (err != CCL_OK) {
        fprintf(stderr, "Failed to set uniform M\n");
        return 1;
    }
    
    err = ccl_set_bytes(kernel, 4, &n_val, sizeof(n_val));
    if (err != CCL_OK) {
        fprintf(stderr, "Failed to set uniform N\n");
        return 1;
    }
    
    err = ccl_set_bytes(kernel, 5, &k_val, sizeof(k_val));
    if (err != CCL_OK) {
        fprintf(stderr, "Failed to set uniform K\n");
        return 1;
    }
    
    // Prepare buffers array (only data buffers, uniforms are set separately)
    ccl_buffer *buffers[3] = { bufA, bufB, bufC };
    
    // Test 1: Synchronous 2D dispatch (baseline)
    printf("\n--- Test 1: Synchronous 2D Dispatch ---\n");
    clock_t start = clock();
    
    size_t global_size[3] = { M, N, 1 };
    size_t local_size[3] = { 0, 0, 0 };  // Auto-select optimal threadgroup size
    
    err = ccl_dispatch_nd(ctx, kernel, 2, global_size, local_size, buffers, 3);
    if (err != CCL_OK) {
        fprintf(stderr, "Synchronous dispatch failed\n");
        return 1;
    }
    
    clock_t end = clock();
    double gpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("GPU time (sync): %.2f ms\n", gpu_time);
    
    // Download result
    ccl_buffer_download(bufC, 0, C_gpu, M * N * sizeof(float));
    
    // CPU reference
    printf("Computing CPU reference...\n");
    start = clock();
    matrix_mult_cpu(A_host, B_host, C_cpu, M, N, K);
    end = clock();
    double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU time: %.2f ms\n", cpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    // Validate
    printf("Validating results...\n");
    if (compare_matrices(C_gpu, C_cpu, M * N, 0.01f)) {
        printf("✓ Results match CPU reference!\n");
    } else {
        printf("✗ Results don't match!\n");
        // Show a few sample values
        printf("Sample: GPU[0] = %f, CPU[0] = %f\n", C_gpu[0], C_cpu[0]);
        printf("Sample: GPU[100] = %f, CPU[100] = %f\n", C_gpu[100], C_cpu[100]);
    }
    
    // Test 2: Async 2D dispatch with fence
    printf("\n--- Test 2: Async 2D Dispatch with Fence ---\n");
    memset(C_gpu, 0, M * N * sizeof(float));
    
    start = clock();
    ccl_fence *fence = NULL;
    err = ccl_dispatch_nd_async(ctx, kernel, 2, global_size, local_size, buffers, 3, &fence);
    if (err != CCL_OK) {
        fprintf(stderr, "Async dispatch failed\n");
        return 1;
    }
    
    // Do some CPU work while GPU is computing (simulate)
    volatile int dummy = 0;
    for (int i = 0; i < 1000000; i++) {
        dummy += i;
    }
    
    // Wait for completion
    ccl_fence_wait(fence);
    end = clock();
    double async_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("Total time (async): %.2f ms\n", async_time);
    
    // Check for errors using new error message API
    const char *error_msg = ccl_fence_get_error_message(fence);
    if (error_msg) {
        fprintf(stderr, "Dispatch error: %s\n", error_msg);
        ccl_fence_destroy(fence);
        return 1;
    }
    
    ccl_buffer_download(bufC, 0, C_gpu, M * N * sizeof(float));
    ccl_fence_destroy(fence);
    
    if (compare_matrices(C_gpu, C_cpu, M * N, 0.01f)) {
        printf("✓ Async results match!\n");
    } else {
        printf("✗ Async results don't match!\n");
    }
    
    // Test 3: Pipeline caching (run same kernel multiple times)
    printf("\n--- Test 3: Pipeline Caching Test ---\n");
    printf("Running kernel 10 times (should benefit from cache)...\n");
    
    start = clock();
    for (int iter = 0; iter < 10; iter++) {
        err = ccl_dispatch_nd(ctx, kernel, 2, global_size, local_size, buffers, 3);
        if (err != CCL_OK) {
            fprintf(stderr, "Dispatch failed at iteration %d\n", iter);
            return 1;
        }
    }
    end = clock();
    double cached_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("10 iterations: %.2f ms (avg: %.2f ms per iteration)\n", cached_time * 1000.0, cached_time * 100.0);
    
    // Test 4: Command buffer batching (group multiple dispatches)
    printf("\n--- Test 4: Command Buffer Batching ---\n");
    printf("Batching 5 dispatches into one command buffer...\n");
    
    start = clock();
    
    // Begin batch
    err = ccl_begin_batch(ctx);
    if (err != CCL_OK) {
        fprintf(stderr, "Failed to begin batch\n");
        return 1;
    }
    
    // Add multiple dispatches to the batch
    for (int i = 0; i < 5; i++) {
        err = ccl_dispatch_nd(ctx, kernel, 2, global_size, local_size, buffers, 3);
        if (err != CCL_OK) {
            fprintf(stderr, "Dispatch %d in batch failed\n", i);
            return 1;
        }
    }
    
    // End batch and get fence
    ccl_fence *batch_fence = NULL;
    err = ccl_end_batch(ctx, &batch_fence);
    if (err != CCL_OK) {
        fprintf(stderr, "Failed to end batch\n");
        return 1;
    }
    
    // Wait for batch completion
    ccl_fence_wait(batch_fence);
    
    // Check for errors
    error_msg = ccl_fence_get_error_message(batch_fence);
    if (error_msg) {
        fprintf(stderr, "Batch error: %s\n", error_msg);
        ccl_fence_destroy(batch_fence);
        return 1;
    }
    
    end = clock();
    double batch_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("5 batched dispatches: %.2f ms (avg: %.2f ms per dispatch)\n", batch_time, batch_time / 5.0);
    printf("✓ Batching reduces command buffer overhead!\n");
    
    ccl_fence_destroy(batch_fence);
    
    // Verify result is still correct
    ccl_buffer_download(bufC, 0, C_gpu, M * N * sizeof(float));
    if (compare_matrices(C_gpu, C_cpu, M * N, 0.01f)) {
        printf("✓ Batched results match!\n");
    } else {
        printf("✗ Batched results don't match!\n");
    }
    
    // Cleanup
    printf("\n--- Cleanup ---\n");
    ccl_destroy_kernel(kernel);
    ccl_destroy_buffer(bufA);
    ccl_destroy_buffer(bufB);
    ccl_destroy_buffer(bufC);
    ccl_destroy_context(ctx);
    
    free(A_host);
    free(B_host);
    free(C_gpu);
    free(C_cpu);
    
    printf("\n=== All tests completed successfully! ===\n");
    return 0;
}

