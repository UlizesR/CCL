# CCL - Cross-Platform Compute Library

CCL is a clean, portable C API for GPU compute that abstracts over multiple backends (Metal, OpenGL Compute, OpenCL). Currently, the Metal backend is fully implemented with comprehensive Metal 3/4 feature support.

## Features

### Core Features
- **Multi-dimensional dispatch**: 1D, 2D, and 3D compute grids
- **Async execution**: Non-blocking dispatches with fence synchronization
- **Pipeline caching**: Automatic kernel reuse for performance
- **Uniform parameters**: Set small constants without creating buffers
- **Buffer usage hints**: GPU-only buffers for optimal performance
- **Debug labels**: Better GPU capture debugging in Xcode/Metal System Trace
- **Thread-safe fences**: Query completion status without blocking

### Metal 3/4 Features
- **Capability detection**: Runtime detection of Metal 3/4 features and GPU families
- **Function tables**: GPU-side function pointer dispatch for dynamic kernel selection
- **Binary archives**: Pipeline caching to disk for faster startup times
- **Ray tracing**: Acceleration structures and ray tracing pipelines (Metal 3+)
- **Indirect command buffers**: GPU-driven command generation for reduced CPU overhead
- **SIMD-group matrix operations**: Optimized matrix operations on Apple7+ GPUs
- **Argument buffers**: Enhanced resource binding with expanded limits (Metal 3/4)

## Quick Start

```c
#include "ccl.h"

// Create context
ccl_context *ctx;
ccl_create_context(CCL_BACKEND_METAL, &ctx);

// Create buffers
ccl_buffer *input, *output;
ccl_create_buffer(ctx, 1024 * sizeof(float), CCL_BUFFER_READ, data, &input);
ccl_create_buffer(ctx, 1024 * sizeof(float), CCL_BUFFER_WRITE, NULL, &output);

// Compile kernel
const char *source = "kernel void add(device const float* a [[buffer(0)]], "
                     "device float* b [[buffer(1)]], uint i [[thread_position_in_grid]]) {"
                     "  b[i] = a[i] + 1.0f; }";
ccl_kernel *kernel;
char log[4096];
ccl_create_kernel_from_source(ctx, source, "add", &kernel, log, sizeof(log));

// Dispatch
ccl_buffer *buffers[2] = {input, output};
ccl_dispatch_1d(ctx, kernel, 1024, 0, buffers, 2);

// Download results
float results[1024];
ccl_buffer_download(output, 0, results, sizeof(results));

// Cleanup
ccl_destroy_kernel(kernel);
ccl_destroy_buffer(input);
ccl_destroy_buffer(output);
ccl_destroy_context(ctx);
```

## API Overview

### Context Management

```c
ccl_error ccl_create_context(ccl_backend backend, ccl_context **out_ctx);
void ccl_destroy_context(ccl_context *ctx);
```

**Thread Safety**: Contexts are not thread-safe. Use each context from a single thread. Multiple contexts can be used concurrently from different threads.

### Buffers

```c
// Standard buffer creation (shared CPU/GPU memory)
ccl_error ccl_create_buffer(
    ccl_context *ctx,
    size_t size,
    ccl_buffer_flags flags,
    const void *initial_data,
    ccl_buffer **out_buf
);

// Extended creation with usage hints
ccl_error ccl_create_buffer_ex(
    ccl_context *ctx,
    size_t size,
    ccl_buffer_flags flags,
    ccl_buffer_usage usage,  // GPU_ONLY, CPU_TO_GPU, etc.
    const void *initial_data,
    ccl_buffer **out_buf
);

ccl_error ccl_buffer_upload(ccl_buffer *buf, size_t offset, const void *data, size_t size);
ccl_error ccl_buffer_download(ccl_buffer *buf, size_t offset, void *data, size_t size);
```

**Buffer Usage Hints**:
- `CCL_BUFFER_USAGE_DEFAULT`: Shared memory (CPU/GPU accessible)
- `CCL_BUFFER_USAGE_GPU_ONLY`: Private memory (GPU-only, faster)
- `CCL_BUFFER_USAGE_CPU_TO_GPU`: Optimized for CPU→GPU transfers
- `CCL_BUFFER_USAGE_GPU_TO_CPU`: Optimized for GPU→CPU transfers

**Note**: GPU_ONLY buffers currently don't support `ccl_buffer_upload`/`ccl_buffer_download` after creation. Use initial data at creation time.

### Kernels

```c
// Compile from source
ccl_error ccl_create_kernel_from_source(
    ccl_context *ctx,
    const char *source,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,        // optional, receives compile errors
    size_t log_buffer_size
);

// Create from precompiled library (faster startup, better obfuscation)
ccl_error ccl_create_kernel_from_library(
    ccl_context *ctx,
    const uint8_t *lib_data,
    size_t lib_size,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,
    size_t log_buffer_size
);

// Set uniform/constant parameters (persist across dispatches)
ccl_error ccl_set_bytes(ccl_kernel *kernel, uint32_t index, const void *data, size_t size);
void ccl_clear_bytes(ccl_kernel *kernel);  // Clear all uniforms
```

**Uniforms**: Uniforms set via `ccl_set_bytes` persist across all dispatches of that kernel until explicitly cleared. This is useful for setting small constants (scalars, structs) without creating buffers.

**Important**: Uniforms and buffers share the same index space. If you set a uniform at index N and also pass a buffer at index N, the buffer will override the uniform. Avoid reusing indices between uniforms and buffers.

### Dispatch

```c
// N-dimensional dispatch (1D, 2D, or 3D)
ccl_error ccl_dispatch_nd(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t dim,                    // 1, 2, or 3
    const size_t global_size[3],   // [x, y, z] dimensions
    const size_t local_size[3],    // threadgroup size (0 = auto)
    ccl_buffer **buffers,
    size_t num_buffers
);

// Async version (non-blocking)
ccl_error ccl_dispatch_nd_async(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t dim,
    const size_t global_size[3],
    const size_t local_size[3],
    ccl_buffer **buffers,
    size_t num_buffers,
    ccl_fence **out_fence   // optional, NULL for fire-and-forget
);

// Convenience wrappers for 1D
ccl_error ccl_dispatch_1d(...);
ccl_error ccl_dispatch_1d_async(...);
```

**Threadgroup Sizing**: If `local_size` is 0, CCL automatically selects optimal threadgroup sizes based on hardware capabilities (e.g., `threadExecutionWidth` on Metal).

### Fences

```c
bool ccl_fence_is_complete(ccl_fence *fence);  // Non-blocking check
void ccl_fence_wait(ccl_fence *fence);         // Blocking wait
void ccl_fence_destroy(ccl_fence *fence);
```

### Debug Labels

```c
void ccl_set_context_label(ccl_context *ctx, const char *label);
void ccl_set_buffer_label(ccl_buffer *buf, const char *label);
void ccl_set_kernel_label(ccl_kernel *kernel, const char *label);
```

Labels improve readability in GPU capture tools (Xcode, Metal System Trace).

### Log Callback

```c
typedef void (*ccl_log_fn)(const char *msg, void *user_data);

void ccl_set_log_callback(ccl_context *ctx, ccl_log_fn fn, void *user_data);
```

Set a callback to receive runtime error messages (compile failures, dispatch errors). Useful for integrating with GUI logs or custom error handling.

### Device Information

```c
typedef enum ccl_device_info {
    CCL_DEVICE_INFO_NAME,                    // Device name (string)
    CCL_DEVICE_INFO_MAX_THREADS_PER_THREADGROUP, // Max threads (uint64_t)
    CCL_DEVICE_INFO_THREAD_EXECUTION_WIDTH,  // SIMD width (uint64_t)
    CCL_DEVICE_INFO_MAX_BUFFER_LENGTH,       // Max buffer size (uint64_t)
    CCL_DEVICE_INFO_SUPPORTS_GPU_ONLY_BUFFERS, // Private storage support (bool)
    CCL_DEVICE_INFO_MAX_COMPUTE_UNITS        // Compute units (uint64_t)
} ccl_device_info;

ccl_error ccl_get_device_info(
    ccl_context *ctx,
    ccl_device_info info,
    void *out_value,      // Pointer to output (type depends on info)
    size_t *out_size      // In: size of buffer, Out: actual size needed
);
```

Query device capabilities and properties. Useful for choosing optimal problem sizes and threadgroup configurations.

**Example:**
```c
char deviceName[256];
size_t size = sizeof(deviceName);
ccl_get_device_info(ctx, CCL_DEVICE_INFO_NAME, deviceName, &size);
printf("Device: %s\n", deviceName);

uint64_t maxThreads;
size = sizeof(maxThreads);
ccl_get_device_info(ctx, CCL_DEVICE_INFO_MAX_THREADS_PER_THREADGROUP, 
                   &maxThreads, &size);
```

**Note**: Some properties (like max threads per threadgroup) are per-pipeline in Metal, so the values returned are reasonable defaults. Actual values may vary based on the specific kernel.

### Metal Capabilities

Query Metal-specific capabilities and feature support:

```c
ccl_metal_capabilities caps;
ccl_get_metal_capabilities(ctx, &caps);

if (caps.supports_metal_4) {
    printf("Metal 4 supported!\n");
}
if (caps.supports_function_tables) {
    printf("Function tables available (max size: %u)\n", caps.max_function_table_size);
}
if (caps.supports_raytracing) {
    printf("Ray tracing supported\n");
}
if (caps.supports_simdgroup_matrix) {
    printf("SIMD-group matrix operations available\n");
}
```

### Function Tables (Metal 3+)

Function tables enable GPU-side function pointer dispatch:

```c
// Create a function table
ccl_function_table *table;
ccl_create_function_table(ctx, 16, &table);  // 16 function slots

// Add kernels to the table
ccl_function_table_set(table, kernel1, 0);
ccl_function_table_set(table, kernel2, 1);

// Use the table in your shader (via buffer binding)
// The shader can dynamically select which function to call

ccl_destroy_function_table(table);
```

### Binary Archives (Metal 3+)

Cache compiled pipelines to disk for faster startup:

```c
// Create archive and add kernels
ccl_binary_archive *archive;
ccl_create_binary_archive(ctx, &archive);
ccl_binary_archive_add_kernel(archive, kernel1);
ccl_binary_archive_add_kernel(archive, kernel2);

// Serialize to save
uint8_t *data = malloc(1024 * 1024);
size_t size = 1024 * 1024;
ccl_binary_archive_serialize(archive, data, &size);
// Save data to disk...

ccl_destroy_binary_archive(archive);
```

### Ray Tracing (Metal 3+)

Create acceleration structures and ray tracing pipelines:

```c
// Create acceleration structure
ccl_acceleration_structure *as;
ccl_create_acceleration_structure(ctx, geometry_count, &as);

// Create ray tracing pipeline
ccl_raytracing_pipeline *rt_pipeline;
const char *source = "...";  // Ray tracing shader source
ccl_create_raytracing_pipeline_from_source(
    ctx, source, "raygen_function", "intersection_function",
    &rt_pipeline, log, sizeof(log)
);

ccl_destroy_raytracing_pipeline(rt_pipeline);
ccl_destroy_acceleration_structure(as);
```

### Indirect Command Buffers (Metal 3+)

Reduce CPU overhead with GPU-driven command generation:

```c
// Create ICB
ccl_indirect_command_buffer *icb;
ccl_create_indirect_command_buffer(ctx, 100, &icb);  // Max 100 commands

// Encode commands into ICB
ccl_indirect_command_buffer_encode_compute(
    icb, kernel, 1, global_size, local_size, buffers, num_buffers
);

// Execute ICB
ccl_fence *fence;
ccl_execute_indirect_command_buffer(ctx, icb, 1, &fence);
ccl_fence_wait(fence);

ccl_destroy_indirect_command_buffer(icb);
```

## Examples

### Basic Vector Addition

See `examples/vec_add_metal.c` for a simple example.

### Advanced Matrix Multiplication

See `examples/matrix_mult_metal.c` for a comprehensive example demonstrating:
- 2D dispatch
- Uniforms API
- Async dispatch with fences
- Pipeline caching
- Multiple concurrent dispatches

### Device Information

See `examples/device_info_metal.c` for an example of querying device capabilities.

### SIMD-Group Matrix Operations

See `shaders/simdgroup_matrix.metal` for example kernels demonstrating SIMD-group matrix operations (Metal 3+, Apple7+ GPUs).

## Building

```bash
mkdir build && cd build
cmake ..
make
```

### Requirements

- macOS with Metal support
- CMake 3.15+
- Xcode Command Line Tools

## Error Handling

All functions return `ccl_error`:
- `CCL_OK`: Success
- `CCL_ERROR_INVALID_ARGUMENT`: Invalid parameters
- `CCL_ERROR_DEVICE_FAILED`: GPU/device error
- `CCL_ERROR_COMPILE_FAILED`: Kernel compilation failed (check log_buffer)
- `CCL_ERROR_DISPATCH_FAILED`: Dispatch execution failed
- `CCL_ERROR_UNSUPPORTED_BACKEND`: Backend not available

## Performance Tips

1. **Use GPU_ONLY buffers** for data that never needs CPU access
2. **Reuse kernels** - pipeline caching automatically reuses compiled kernels
3. **Use async dispatch** when launching multiple kernels
4. **Set uniforms** instead of creating tiny buffers for constants
5. **Use 2D/3D dispatch** for naturally multi-dimensional problems
6. **Precompile libraries** - use `ccl_create_kernel_from_library` for faster startup and better code obfuscation
7. **Use binary archives** (Metal 3+) to cache pipelines to disk and reduce compilation time
8. **Leverage function tables** (Metal 3+) for dynamic kernel selection without CPU overhead
9. **Use indirect command buffers** (Metal 3+) for GPU-driven pipelines with many similar dispatches
10. **Check capabilities** - use `ccl_get_metal_capabilities` to gate Metal 3/4 features appropriately

## Thread Safety

- **Contexts**: Not thread-safe. Use from a single thread.
- **Buffers/Kernels**: Can be used from any thread, but must outlive all dispatches.
- **Fences**: Can be checked/wait from any thread.


