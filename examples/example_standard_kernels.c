/**
 * Standard Kernels Test
 * Demonstrates the pre-built kernel library for scientific computing
 */

#include "mtl_compute_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void test_saxpy(MTLComputeDevice* device) {
    printf("\n=== TEST: SAXPY (alpha*x + y) ===\n");
    
    MTLComputeError error;
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create_from_file(
        device, "shaders/standard_kernels.metal", "saxpy", &error, NULL, 0
    );
    
    if (!pipeline) {
        printf("  ✗ Failed to load kernel: %s\n", mtl_compute_error_string(error));
        return;
    }
    
    size_t n = 1024;
    float alpha = 2.5f;
    
    float* x = (float*)malloc(n * sizeof(float));
    float* y = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        x[i] = (float)i;
        y[i] = (float)i * 0.5f;
    }
    
    MTLComputeBuffer* x_buf = mtl_compute_buffer_create_with_data(device, x, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* y_buf = mtl_compute_buffer_create_with_data(device, y, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* out_buf = mtl_compute_buffer_create(device, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* alpha_buf = mtl_compute_buffer_create_with_data(device, &alpha, sizeof(float), MTL_STORAGE_SHARED);
    
    MTLComputeBuffer* buffers[] = {x_buf, y_buf, out_buf, alpha_buf};
    
    error = mtl_compute_dispatch_sync(device, pipeline, buffers, 4, n, 1, 1, 256, 1, 1);
    
    if (error == MTL_SUCCESS) {
        float* result = mtl_compute_buffer_contents(out_buf);
        float expected = alpha * x[0] + y[0];  // 2.5 * 0 + 0 = 0
        float expected_10 = alpha * x[10] + y[10];  // 2.5 * 10 + 5 = 30
        
        bool passed = fabs(result[0] - expected) < 0.001f && fabs(result[10] - expected_10) < 0.001f;
        printf("  %s SAXPY computation (result[10]=%.2f, expected=%.2f)\n", 
               passed ? "✓" : "✗", result[10], expected_10);
    } else {
        printf("  ✗ Failed: %s\n", mtl_compute_error_string(error));
    }
    
    free(x);
    free(y);
    mtl_compute_buffer_destroy(x_buf);
    mtl_compute_buffer_destroy(y_buf);
    mtl_compute_buffer_destroy(out_buf);
    mtl_compute_buffer_destroy(alpha_buf);
    mtl_compute_pipeline_destroy(pipeline);
}

void test_compute_pass(MTLComputeDevice* device) {
    printf("\n=== TEST: Compute Pass (Multi-Step Pipeline) ===\n");
    
    MTLComputeError error;
    MTLComputePipeline* fill = mtl_compute_pipeline_create_from_file(
        device, "shaders/standard_kernels.metal", "fill_float", &error, NULL, 0
    );
    MTLComputePipeline* multiply = mtl_compute_pipeline_create_from_file(
        device, "shaders/standard_kernels.metal", "multiply_arrays", &error, NULL, 0
    );
    
    if (!fill || !multiply) {
        printf("  ✗ Failed to load kernels\n");
        return;
    }
    
    size_t n = 512;
    float fill_value = 3.0f;
    
    MTLComputeBuffer* a = mtl_compute_buffer_create(device, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* b = mtl_compute_buffer_create(device, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* c = mtl_compute_buffer_create(device, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* fill_val = mtl_compute_buffer_create_with_data(device, &fill_value, sizeof(float), MTL_STORAGE_SHARED);
    
    // Create a compute pass
    MTLComputePass* pass = mtl_compute_pass_create(device, 3);
    if (!pass) {
        printf("  ✗ Failed to create pass\n");
        return;
    }
    
    // Step 1: Fill buffer A with 3.0
    MTLComputeBuffer* fill_buffers_a[] = {a, fill_val};
    MTLComputeDispatchDesc fill_a = {
        .pipeline = fill,
        .buffers = fill_buffers_a,
        .buffer_count = 2,
        .textures = NULL,
        .texture_count = 0,
        .samplers = NULL,
        .sampler_count = 0,
        .grid_width = n,
        .grid_height = 1,
        .grid_depth = 1,
        .threadgroup_width = 256,
        .threadgroup_height = 1,
        .threadgroup_depth = 1
    };
    mtl_compute_pass_add_dispatch(pass, &fill_a);
    
    // Step 2: Fill buffer B with 3.0
    MTLComputeBuffer* fill_buffers_b[] = {b, fill_val};
    MTLComputeDispatchDesc fill_b = fill_a;
    fill_b.buffers = fill_buffers_b;
    mtl_compute_pass_add_dispatch(pass, &fill_b);
    
    // Step 3: C = A * B
    MTLComputeBuffer* mul_buffers[] = {a, b, c};
    MTLComputeDispatchDesc mul = {
        .pipeline = multiply,
        .buffers = mul_buffers,
        .buffer_count = 3,
        .textures = NULL,
        .texture_count = 0,
        .samplers = NULL,
        .sampler_count = 0,
        .grid_width = n,
        .grid_height = 1,
        .grid_depth = 1,
        .threadgroup_width = 256,
        .threadgroup_height = 1,
        .threadgroup_depth = 1
    };
    mtl_compute_pass_add_dispatch(pass, &mul);
    
    // Execute entire pass
    error = mtl_compute_pass_execute(pass);
    
    if (error == MTL_SUCCESS) {
        float* result = mtl_compute_buffer_contents(c);
        float expected = 3.0f * 3.0f;  // 9.0
        bool passed = fabs(result[0] - expected) < 0.001f;
        printf("  %s Compute pass (3 steps) result=%.1f, expected=%.1f\n",
               passed ? "✓" : "✗", result[0], expected);
    } else {
        printf("  ✗ Pass execution failed: %s\n", mtl_compute_error_string(error));
    }
    
    mtl_compute_pass_destroy(pass);
    mtl_compute_buffer_destroy(a);
    mtl_compute_buffer_destroy(b);
    mtl_compute_buffer_destroy(c);
    mtl_compute_buffer_destroy(fill_val);
    mtl_compute_pipeline_destroy(fill);
    mtl_compute_pipeline_destroy(multiply);
}

void test_reduction(MTLComputeDevice* device) {
    printf("\n=== TEST: Parallel Reduction (Sum) ===\n");
    
    MTLComputeError error;
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create_from_file(
        device, "shaders/standard_kernels.metal", "reduce_sum_threadgroup", &error, NULL, 0
    );
    
    if (!pipeline) {
        printf("  ✗ Failed to load kernel\n");
        return;
    }
    
    size_t n = 1024;
    size_t threads_per_group = 256;
    size_t num_groups = (n + threads_per_group - 1) / threads_per_group;
    
    float* data = (float*)malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        data[i] = 1.0f;  // Sum should be 1024
    }
    
    uint32_t count = (uint32_t)n;
    MTLComputeBuffer* input = mtl_compute_buffer_create_with_data(device, data, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* partial = mtl_compute_buffer_create(device, num_groups * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* count_buf = mtl_compute_buffer_create_with_data(device, &count, sizeof(uint32_t), MTL_STORAGE_SHARED);
    
    MTLComputeBuffer* buffers[] = {input, partial, count_buf};
    
    error = mtl_compute_dispatch_sync(device, pipeline, buffers, 3, n, 1, 1, threads_per_group, 1, 1);
    
    if (error == MTL_SUCCESS) {
        float* partial_sums = mtl_compute_buffer_contents(partial);
        float total = 0.0f;
        for (size_t i = 0; i < num_groups; i++) {
            total += partial_sums[i];
        }
        
        bool passed = fabs(total - 1024.0f) < 0.1f;
        printf("  %s Parallel reduction sum=%.1f, expected=1024.0\n",
               passed ? "✓" : "✗", total);
    } else {
        printf("  ✗ Failed: %s\n", mtl_compute_error_string(error));
    }
    
    free(data);
    mtl_compute_buffer_destroy(input);
    mtl_compute_buffer_destroy(partial);
    mtl_compute_buffer_destroy(count_buf);
    mtl_compute_pipeline_destroy(pipeline);
}

void test_2d_stencil(MTLComputeDevice* device) {
    printf("\n=== TEST: 2D Heat Equation (5-point stencil) ===\n");
    
    MTLComputeError error;
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create_from_file(
        device, "shaders/standard_kernels.metal", "heat_2d_step", &error, NULL, 0
    );
    
    if (!pipeline) {
        printf("  ✗ Failed to load kernel\n");
        return;
    }
    
    uint32_t width = 64;
    uint32_t height = 64;
    size_t n = width * height;
    float dt = 0.01f;
    float dx = 1.0f;
    uint32_t dims[2] = {width, height};
    
    float* current = (float*)calloc(n, sizeof(float));
    // Set initial hot spot in center
    current[(height/2) * width + (width/2)] = 100.0f;
    
    MTLComputeBuffer* current_buf = mtl_compute_buffer_create_with_data(device, current, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* next_buf = mtl_compute_buffer_create(device, n * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* dt_buf = mtl_compute_buffer_create_with_data(device, &dt, sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* dx_buf = mtl_compute_buffer_create_with_data(device, &dx, sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* dims_buf = mtl_compute_buffer_create_with_data(device, dims, sizeof(dims), MTL_STORAGE_SHARED);
    
    MTLComputeBuffer* buffers[] = {current_buf, next_buf, dt_buf, dx_buf, dims_buf};
    
    // Run one timestep
    error = mtl_compute_dispatch_sync(device, pipeline, buffers, 5, width, height, 1, 16, 16, 1);
    
    if (error == MTL_SUCCESS) {
        float* result = mtl_compute_buffer_contents(next_buf);
        float center = result[(height/2) * width + (width/2)];
        float neighbor = result[(height/2) * width + (width/2) + 1];
        
        bool passed = (center < 100.0f && center > 90.0f) && neighbor > 0.0f;
        printf("  %s 2D diffusion step (center=%.2f, diffused=%.4f)\n",
               passed ? "✓" : "✗", center, neighbor);
    } else {
        printf("  ✗ Failed: %s\n", mtl_compute_error_string(error));
    }
    
    free(current);
    mtl_compute_buffer_destroy(current_buf);
    mtl_compute_buffer_destroy(next_buf);
    mtl_compute_buffer_destroy(dt_buf);
    mtl_compute_buffer_destroy(dx_buf);
    mtl_compute_buffer_destroy(dims_buf);
    mtl_compute_pipeline_destroy(pipeline);
}

int main(void) {
    printf("=========================================\n");
    printf("  MTLComp Standard Kernels Demo\n");
    printf("=========================================\n");
    
    MTLComputeDevice* device = mtl_compute_device_create();
    if (!device) {
        fprintf(stderr, "Failed to create Metal device\n");
        return 1;
    }
    
    printf("\nDevice: %s\n", mtl_compute_device_get_name(device));
    
    // Run tests
    test_saxpy(device);
    test_compute_pass(device);
    test_reduction(device);
    test_2d_stencil(device);
    
    printf("\n=========================================\n");
    printf("  All standard kernel tests completed\n");
    printf("=========================================\n\n");
    
    mtl_compute_device_destroy(device);
    return 0;
}

