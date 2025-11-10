/**
 * MTLComp Feature Test Harness
 * Tests all API tiers and Metal 3/4 features with runtime capability checks
 */

#include "mtl_compute_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void test_basic_dispatch(MTLComputeDevice* device) {
    printf("\n=== TEST: Basic Synchronous Dispatch (Tier 1) ===\n");
    
    const char* shader_src = 
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void add_arrays(device const float* a [[buffer(0)]],\n"
        "                       device const float* b [[buffer(1)]],\n"
        "                       device float* c [[buffer(2)]],\n"
        "                       uint gid [[thread_position_in_grid]]) {\n"
        "    c[gid] = a[gid] + b[gid];\n"
        "}\n";
    
    MTLComputeError error;
    char error_log[512];
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create(
        device, shader_src, "add_arrays", &error, error_log, sizeof(error_log)
    );
    
    if (!pipeline) {
        printf("  ✗ Failed: %s\n", error_log);
        return;
    }
    
    mtl_compute_pipeline_set_label(pipeline, "AddArrays");
    
    // Create test data
    size_t count = 1024;
    float* a_data = malloc(count * sizeof(float));
    float* b_data = malloc(count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        a_data[i] = (float)i;
        b_data[i] = (float)i * 2.0f;
    }
    
    MTLComputeBuffer* a = mtl_compute_buffer_create_with_data(device, a_data, count * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* b = mtl_compute_buffer_create_with_data(device, b_data, count * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* c = mtl_compute_buffer_create(device, count * sizeof(float), MTL_STORAGE_SHARED);
    
    mtl_compute_buffer_set_label(a, "InputA");
    mtl_compute_buffer_set_label(b, "InputB");
    mtl_compute_buffer_set_label(c, "Output");
    
    MTLComputeBuffer* buffers[] = {a, b, c};
    
    // Dispatch with auto threadgroup sizing (pass 0 for auto)
    error = mtl_compute_dispatch_sync(device, pipeline, buffers, 3, count, 1, 1, 0, 1, 1);
    
    if (error == MTL_SUCCESS) {
        float* result = mtl_compute_buffer_contents(c);
        bool passed = true;
        for (size_t i = 0; i < 10; i++) {
            float expected = (float)i * 3.0f;
            if (fabs(result[i] - expected) > 0.001f) {
                passed = false;
                break;
            }
        }
        printf("  %s Tier 1 dispatch with auto-threadgroup\n", passed ? "✓" : "✗");
    } else {
        printf("  ✗ Dispatch failed: %s\n", mtl_compute_error_string(error));
    }
    
    free(a_data);
    free(b_data);
    mtl_compute_buffer_destroy(a);
    mtl_compute_buffer_destroy(b);
    mtl_compute_buffer_destroy(c);
    mtl_compute_pipeline_destroy(pipeline);
}

void test_descriptor_dispatch(MTLComputeDevice* device) {
    printf("\n=== TEST: Unified Descriptor Dispatch (Tier 2) ===\n");
    
    const char* shader_src = 
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void multiply(device float* data [[buffer(0)]],\n"
        "                     constant float& factor [[buffer(1)]],\n"
        "                     uint gid [[thread_position_in_grid]]) {\n"
        "    data[gid] *= factor;\n"
        "}\n";
    
    MTLComputeError error;
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create(
        device, shader_src, "multiply", &error, NULL, 0
    );
    
    if (!pipeline) {
        printf("  ✗ Failed to create pipeline\n");
        return;
    }
    
    size_t count = 512;
    float* data = malloc(count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        data[i] = 1.0f;
    }
    float factor = 2.5f;
    
    MTLComputeBuffer* data_buf = mtl_compute_buffer_create_with_data(device, data, count * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* factor_buf = mtl_compute_buffer_create_with_data(device, &factor, sizeof(float), MTL_STORAGE_SHARED);
    
    MTLComputeBuffer* buffers[] = {data_buf, factor_buf};
    
    // Use descriptor with auto threadgroup (0 = auto)
    MTLComputeDispatchDesc desc = {
        .pipeline = pipeline,
        .buffers = buffers,
        .buffer_count = 2,
        .textures = NULL,
        .texture_count = 0,
        .samplers = NULL,
        .sampler_count = 0,
        .grid_width = count,
        .grid_height = 1,
        .grid_depth = 1,
        .threadgroup_width = 0,  // Auto!
        .threadgroup_height = 1,
        .threadgroup_depth = 1
    };
    
    error = mtl_compute_dispatch_desc(device, &desc);
    
    if (error == MTL_SUCCESS) {
        float* result = mtl_compute_buffer_contents(data_buf);
        bool passed = fabs(result[0] - 2.5f) < 0.001f;
        printf("  %s Descriptor dispatch with auto-sizing\n", passed ? "✓" : "✗");
    } else {
        printf("  ✗ Failed: %s\n", mtl_compute_error_string(error));
    }
    
    free(data);
    mtl_compute_buffer_destroy(data_buf);
    mtl_compute_buffer_destroy(factor_buf);
    mtl_compute_pipeline_destroy(pipeline);
}

void test_encoder_batching(MTLComputeDevice* device) {
    printf("\n=== TEST: Encoder Batching (Tier 3) ===\n");
    
    const char* shader_src = 
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void increment(device float* data [[buffer(0)]],\n"
        "                      uint gid [[thread_position_in_grid]]) {\n"
        "    data[gid] += 1.0;\n"
        "}\n";
    
    MTLComputeError error;
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create(
        device, shader_src, "increment", &error, NULL, 0
    );
    
    if (!pipeline) {
        printf("  ✗ Failed to create pipeline\n");
        return;
    }
    
    size_t count = 256;
    float* data = calloc(count, sizeof(float));
    MTLComputeBuffer* buffer = mtl_compute_buffer_create_with_data(device, data, count * sizeof(float), MTL_STORAGE_SHARED);
    
    // Begin encoder
    MTLComputeCommandList* cmd_list;
    error = mtl_compute_begin(device, &cmd_list);
    if (error != MTL_SUCCESS) {
        printf("  ✗ Failed to begin: %s\n", mtl_compute_error_string(error));
        free(data);
        mtl_compute_buffer_destroy(buffer);
        mtl_compute_pipeline_destroy(pipeline);
        return;
    }
    
    mtl_compute_command_list_set_label(cmd_list, "BatchTest");
    
    // Encode 3 dispatches in one command buffer
    MTLComputeDispatchDesc desc = {
        .pipeline = pipeline,
        .buffers = &buffer,
        .buffer_count = 1,
        .textures = NULL,
        .texture_count = 0,
        .samplers = NULL,
        .sampler_count = 0,
        .grid_width = count,
        .grid_height = 1,
        .grid_depth = 1,
        .threadgroup_width = 64,
        .threadgroup_height = 1,
        .threadgroup_depth = 1
    };
    
    for (int i = 0; i < 3; i++) {
        error = mtl_compute_encode_dispatch(cmd_list, &desc);
        if (error != MTL_SUCCESS) {
            printf("  ✗ Encode %d failed\n", i);
            free(data);
            mtl_compute_buffer_destroy(buffer);
            mtl_compute_pipeline_destroy(pipeline);
            return;
        }
    }
    
    // Submit and wait
    error = mtl_compute_end_submit(cmd_list);
    
    if (error == MTL_SUCCESS) {
        float* result = mtl_compute_buffer_contents(buffer);
        bool passed = fabs(result[0] - 3.0f) < 0.001f; // Should be incremented 3 times
        printf("  %s Batched 3 dispatches (value: %.1f, expected 3.0)\n", 
               passed ? "✓" : "✗", result[0]);
    } else {
        printf("  ✗ Failed: %s\n", mtl_compute_error_string(error));
    }
    
    free(data);
    mtl_compute_buffer_destroy(buffer);
    mtl_compute_pipeline_destroy(pipeline);
}

void test_profiling(MTLComputeDevice* device) {
    printf("\n=== TEST: Performance Profiling ===\n");
    
    const char* shader_src = 
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void work(device float* data [[buffer(0)]],\n"
        "                 uint gid [[thread_position_in_grid]]) {\n"
        "    float x = data[gid];\n"
        "    for (int i = 0; i < 100; i++) {\n"
        "        x = sin(x * 0.1) + cos(x * 0.2);\n"
        "    }\n"
        "    data[gid] = x;\n"
        "}\n";
    
    MTLComputeError error;
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create(
        device, shader_src, "work", &error, NULL, 0
    );
    
    if (!pipeline) {
        printf("  ✗ Failed to create pipeline\n");
        return;
    }
    
    size_t count = 65536;
    float* data = malloc(count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        data[i] = (float)i;
    }
    
    MTLComputeBuffer* buffer = mtl_compute_buffer_create_with_data(device, data, count * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* buffers[] = {buffer};
    
    MTLComputePerformanceStats stats;
    error = mtl_compute_dispatch_profiled(device, pipeline, buffers, 1, count, 256, &stats);
    
    if (error == MTL_SUCCESS) {
        printf("  ✓ Profiling completed\n");
        printf("    CPU time: %.3f ms\n", stats.cpu_time_ms);
        printf("    GPU time: %.3f ms\n", stats.gpu_time_ms);
        printf("    Threads: %llu\n", stats.threads_executed);
        printf("    Threadgroups: %llu\n", stats.threadgroups_executed);
        printf("    Memory: %zu bytes\n", stats.memory_used_bytes);
        printf("    SIMD width: %zu\n", stats.execution_width);
    } else {
        printf("  ✗ Failed: %s\n", mtl_compute_error_string(error));
    }
    
    free(data);
    mtl_compute_buffer_destroy(buffer);
    mtl_compute_pipeline_destroy(pipeline);
}

void test_async_dispatch(MTLComputeDevice* device, MTLComputeDeviceCapabilities* caps) {
    printf("\n=== TEST: Async Dispatch with Shared Events ===\n");
    
    if (!caps->supports_shared_events) {
        printf("  ⊘ Shared events not supported on this device\n");
        return;
    }
    
    const char* shader_src = 
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void square(device float* data [[buffer(0)]],\n"
        "                   uint gid [[thread_position_in_grid]]) {\n"
        "    data[gid] = data[gid] * data[gid];\n"
        "}\n";
    
    MTLComputeError error;
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create(
        device, shader_src, "square", &error, NULL, 0
    );
    
    if (!pipeline) {
        printf("  ✗ Failed to create pipeline\n");
        return;
    }
    
    MTLComputeSharedEvent* event = mtl_compute_event_create(device);
    if (!event) {
        printf("  ✗ Failed to create event\n");
        mtl_compute_pipeline_destroy(pipeline);
        return;
    }
    
    size_t count = 1024;
    float* data = malloc(count * sizeof(float));
    for (size_t i = 0; i < count; i++) {
        data[i] = (float)i;
    }
    
    MTLComputeBuffer* buffer = mtl_compute_buffer_create_with_data(device, data, count * sizeof(float), MTL_STORAGE_SHARED);
    MTLComputeBuffer* buffers[] = {buffer};
    
    // Async dispatch
    uint64_t signal_value = 1;
    error = mtl_compute_dispatch_async(device, pipeline, buffers, 1, count, 256, event, signal_value);
    
    if (error == MTL_SUCCESS) {
        printf("  ✓ Async dispatch submitted\n");
        
        // Wait with timeout
        bool completed = mtl_compute_event_wait(event, signal_value, 5000000000); // 5 sec
        
        if (completed) {
            float* result = mtl_compute_buffer_contents(buffer);
            bool passed = fabs(result[0] - 0.0f) < 0.001f && fabs(result[10] - 100.0f) < 0.001f;
            printf("  %s Async execution completed\n", passed ? "✓" : "✗");
        } else {
            printf("  ✗ Timeout waiting for completion\n");
        }
    } else {
        printf("  ✗ Failed: %s\n", mtl_compute_error_string(error));
    }
    
    free(data);
    mtl_compute_buffer_destroy(buffer);
    mtl_compute_event_destroy(event);
    mtl_compute_pipeline_destroy(pipeline);
}

void test_pipeline_library(MTLComputeDevice* device) {
    printf("\n=== TEST: Pipeline Library Management ===\n");
    
    MTLComputePipelineLibrary* library = mtl_compute_pipeline_library_create(device, NULL);
    if (!library) {
        printf("  ✗ Failed to create library\n");
        return;
    }
    
    // Create pipelines
    const char* shader1 = 
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void add(device float* d [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    d[gid] += 1.0;\n"
        "}\n";
    
    const char* shader2 = 
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void mul(device float* d [[buffer(0)]], uint gid [[thread_position_in_grid]]) {\n"
        "    d[gid] *= 2.0;\n"
        "}\n";
    
    MTLComputeError error;
    MTLComputePipeline* p1 = mtl_compute_pipeline_create(device, shader1, "add", &error, NULL, 0);
    MTLComputePipeline* p2 = mtl_compute_pipeline_create(device, shader2, "mul", &error, NULL, 0);
    
    if (!p1 || !p2) {
        printf("  ✗ Failed to create pipelines\n");
        mtl_compute_pipeline_library_destroy(library);
        return;
    }
    
    // Add to library
    mtl_compute_pipeline_library_add(library, "add", p1);
    mtl_compute_pipeline_library_add(library, "mul", p2);
    
    // Retrieve
    MTLComputePipeline* retrieved = mtl_compute_pipeline_library_get(library, "add");
    bool passed = (retrieved == p1);
    
    printf("  %s Pipeline library storage/retrieval\n", passed ? "✓" : "✗");
    
    mtl_compute_pipeline_destroy(p1);
    mtl_compute_pipeline_destroy(p2);
    mtl_compute_pipeline_library_destroy(library);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
    printf("=========================================\n");
    printf("  MTLComp Comprehensive Feature Test\n");
    printf("=========================================\n");
    printf("\nLibrary Version: %s\n", mtl_compute_version());
    
    // Initialize device
    MTLComputeDevice* device = mtl_compute_device_create();
    if (!device) {
        fprintf(stderr, "Failed to create Metal device\n");
        return 1;
    }
    
    // Print capabilities
    mtl_compute_device_print_features(device);
    
    // Get capabilities for conditional tests
    MTLComputeDeviceCapabilities caps;
    mtl_compute_device_get_capabilities(device, &caps);
    
    // Run tests
    test_basic_dispatch(device);
    test_descriptor_dispatch(device);
    test_encoder_batching(device);
    test_profiling(device);
    test_async_dispatch(device, &caps);
    test_pipeline_library(device);
    
    printf("\n=========================================\n");
    printf("  Test suite completed\n");
    printf("=========================================\n\n");
    
    mtl_compute_device_destroy(device);
    return 0;
}

