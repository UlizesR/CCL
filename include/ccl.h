// include/ccl.h
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ccl_backend {
    CCL_BACKEND_METAL = 0,
    CCL_BACKEND_GL_COMPUTE = 1,
    CCL_BACKEND_OPENCL = 2
} ccl_backend;

typedef enum ccl_error {
    CCL_OK = 0,
    CCL_ERROR_UNSUPPORTED_BACKEND,
    CCL_ERROR_BACKEND_INIT_FAILED,
    CCL_ERROR_DEVICE_FAILED,
    CCL_ERROR_COMPILE_FAILED,
    CCL_ERROR_INVALID_ARGUMENT,
    CCL_ERROR_DISPATCH_FAILED,
    CCL_ERROR_NOT_SUPPORTED  // Feature not supported on this device/backend
} ccl_error;

// Device info query types
typedef enum ccl_device_info {
    CCL_DEVICE_INFO_NAME = 0,                    // Device name (string)
    CCL_DEVICE_INFO_MAX_THREADS_PER_THREADGROUP, // Max threads per threadgroup (uint64_t)
    CCL_DEVICE_INFO_THREAD_EXECUTION_WIDTH,      // SIMD width (uint64_t)
    CCL_DEVICE_INFO_MAX_BUFFER_LENGTH,           // Max buffer size in bytes (uint64_t)
    CCL_DEVICE_INFO_SUPPORTS_GPU_ONLY_BUFFERS,   // Supports private storage (bool)
    CCL_DEVICE_INFO_MAX_COMPUTE_UNITS            // Number of compute units (uint64_t)
} ccl_device_info;

// Metal capability flags structure
typedef struct ccl_metal_capabilities {
    bool supports_metal_3;           // Metal 3 support
    bool supports_metal_4;           // Metal 4 support
    bool supports_function_tables;   // GPU function pointers
    bool supports_raytracing;        // Ray tracing support
    bool supports_binary_archives;   // Pipeline binary archives
    bool supports_simdgroup_matrix;  // SIMD-group matrix operations
    bool supports_indirect_command_buffers; // Indirect command buffers
    bool supports_argument_buffers;   // Argument buffers
    bool supports_gpu_dynamic_libraries; // GPU Dynamic Libraries (Metal 4+)
    uint32_t max_function_table_size; // Max entries in function table
    uint32_t max_argument_buffer_length; // Max argument buffer size (Metal 3: 128KB, Metal 4: potentially larger)
} ccl_metal_capabilities;

// Forward declarations
// NOTE: CCL contexts are not thread-safe. Access a context from a single thread
// unless otherwise documented. Multiple contexts can be used concurrently from
// different threads, but each context should be used by only one thread.
typedef struct ccl_context ccl_context;

// Log callback function type
typedef void (*ccl_log_fn)(const char *msg, void *user_data);
typedef struct ccl_buffer  ccl_buffer;
typedef struct ccl_kernel  ccl_kernel;
typedef struct ccl_fence   ccl_fence;

// Buffer access flags
typedef enum ccl_buffer_flags {
    CCL_BUFFER_READ      = 1 << 0,
    CCL_BUFFER_WRITE     = 1 << 1,
    CCL_BUFFER_READWRITE = CCL_BUFFER_READ | CCL_BUFFER_WRITE
} ccl_buffer_flags;

// Buffer usage hints (for performance optimization)
typedef enum ccl_buffer_usage {
    CCL_BUFFER_USAGE_DEFAULT = 0,  // Shared memory (CPU/GPU accessible)
    CCL_BUFFER_USAGE_GPU_ONLY,      // Private memory (GPU-only, faster) Note: GPU_ONLY buffers can only be initialized at creation time via initial_data. Subsequent ccl_buffer_upload/download calls will fail.
    CCL_BUFFER_USAGE_CPU_TO_GPU,    // Optimized for CPU→GPU transfers
    CCL_BUFFER_USAGE_GPU_TO_CPU     // Optimized for GPU→CPU transfers
} ccl_buffer_usage;

// Context lifecycle
ccl_error ccl_create_context(ccl_backend backend, ccl_context **out_ctx);
void      ccl_destroy_context(ccl_context *ctx);

// Device information queries
// Query device capabilities and properties. Returns CCL_ERROR_NOT_SUPPORTED
// if the info type is not available for this backend/device.
ccl_error ccl_get_device_info(
    ccl_context *ctx,
    ccl_device_info info,
    void *out_value,      // Pointer to output (type depends on info)
    size_t *out_size      // In: size of out_value buffer, Out: actual size needed
);

// Log callback for error reporting
void ccl_set_log_callback(ccl_context *ctx, ccl_log_fn fn, void *user_data);

// Command buffer batching (group multiple dispatches into one command buffer)
// Begin a batch: subsequent dispatches will be added to the same command buffer
// End a batch: commit the command buffer and optionally return a fence
//
// IMPORTANT: While a batch is active:
//   - All ccl_dispatch_* calls share a single command buffer and encoder
//   - ccl_dispatch_*_async will NOT return fences (sets *out_fence = NULL)
//   - The only fence available is from ccl_end_batch
//   - Calling ccl_begin_batch twice without ccl_end_batch is an error
//
// Example:
//   ccl_begin_batch(ctx);
//   ccl_dispatch_nd(ctx, kernel1, ...);  // Added to batch
//   ccl_dispatch_nd(ctx, kernel2, ...);  // Added to batch
//   ccl_fence *f = NULL;
//   ccl_end_batch(ctx, &f);  // Commits all at once
//   ccl_fence_wait(f);
ccl_error ccl_begin_batch(ccl_context *ctx);
ccl_error ccl_end_batch(ccl_context *ctx, ccl_fence **out_fence);

// Buffers
ccl_error ccl_create_buffer(
    ccl_context *ctx,
    size_t size,
    ccl_buffer_flags flags,
    const void *initial_data,
    ccl_buffer **out_buf
);

// Extended buffer creation with usage hints
ccl_error ccl_create_buffer_ex(
    ccl_context *ctx,
    size_t size,
    ccl_buffer_flags flags,
    ccl_buffer_usage usage,
    const void *initial_data,
    ccl_buffer **out_buf
);

void ccl_destroy_buffer(ccl_buffer *buf);

ccl_error ccl_buffer_upload(
    ccl_buffer *buf,
    size_t offset,
    const void *data,
    size_t size
);

ccl_error ccl_buffer_download(
    ccl_buffer *buf,
    size_t offset,
    void *data,
    size_t size
);

// Extended upload/download with context (supports GPU_ONLY buffers via blit transfers)
ccl_error ccl_buffer_upload_ex(
    ccl_context *ctx,
    ccl_buffer *buf,
    size_t offset,
    const void *data,
    size_t size
);

ccl_error ccl_buffer_download_ex(
    ccl_context *ctx,
    ccl_buffer *buf,
    size_t offset,
    void *data,
    size_t size
);

// Kernels (compute programs)
ccl_error ccl_create_kernel_from_source(
    ccl_context *ctx,
    const char *source,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,        // optional
    size_t log_buffer_size   // 0 if no log
);

// Create kernel from precompiled library (faster startup, better obfuscation)
ccl_error ccl_create_kernel_from_library(
    ccl_context *ctx,
    const uint8_t *lib_data,
    size_t lib_size,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,        // optional
    size_t log_buffer_size   // 0 if no log
);

void ccl_destroy_kernel(ccl_kernel *kernel);

// Set uniform/constant bytes for a kernel (small parameters)
// Uniforms are persistent across dispatches until explicitly cleared.
// Use this to set small parameters (scalars, structs) that will be applied
// to all subsequent dispatches of this kernel.
//
// IMPORTANT: Uniforms and buffers share the same index space. If you set
// a uniform at index N and also pass a buffer at index N, the buffer will
// override the uniform (buffers are set after uniforms in dispatch).
// Avoid reusing indices between uniforms and buffers.
ccl_error ccl_set_bytes(
    ccl_kernel *kernel,
    uint32_t index,        // buffer index
    const void *data,
    size_t size
);

// Clear all uniforms for a kernel
void ccl_clear_bytes(ccl_kernel *kernel);

// N-dimensional dispatch (1D, 2D, or 3D)
ccl_error ccl_dispatch_nd(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t dim,                    // 1, 2, or 3
    const size_t global_size[3],   // [x, y, z] dimensions
    const size_t local_size[3],    // threadgroup size per dimension (0 = auto)
    ccl_buffer **buffers,
    size_t num_buffers
);

// Async N-dimensional dispatch
ccl_error ccl_dispatch_nd_async(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t dim,
    const size_t global_size[3],
    const size_t local_size[3],
    ccl_buffer **buffers,
    size_t num_buffers,
    ccl_fence **out_fence   // optional, NULL to fire-and-forget
);

// 1D dispatch (convenience wrapper around ccl_dispatch_nd)
ccl_error ccl_dispatch_1d(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t global_size,
    size_t local_size,   // 0 = choose automatically (Metal max threads per TG)
    ccl_buffer **buffers,
    size_t num_buffers
);

// Async 1D dispatch (convenience wrapper)
ccl_error ccl_dispatch_1d_async(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t global_size,
    size_t local_size,
    ccl_buffer **buffers,
    size_t num_buffers,
    ccl_fence **out_fence   // optional, NULL to fire-and-forget
);

// Fence operations
bool ccl_fence_is_complete(ccl_fence *fence);  // Non-blocking check
void ccl_fence_wait(ccl_fence *fence);         // Blocking wait
void ccl_fence_destroy(ccl_fence *fence);

// Get error message from a completed fence (returns NULL if no error or not yet completed)
// The returned string is valid until the fence is destroyed
const char *ccl_fence_get_error_message(ccl_fence *fence);

// Debug labels (for GPU capture tools)
void ccl_set_context_label(ccl_context *ctx, const char *label);
void ccl_set_buffer_label(ccl_buffer *buf, const char *label);
void ccl_set_kernel_label(ccl_kernel *kernel, const char *label);

// Metal 3/4 Capability Detection
// Query Metal-specific capabilities. Returns CCL_ERROR_NOT_SUPPORTED if not using Metal backend.
ccl_error ccl_get_metal_capabilities(
    ccl_context *ctx,
    ccl_metal_capabilities *out_caps
);

// Function Tables (Metal 3+)
// Function tables enable GPU-side function pointer dispatch for dynamic kernel selection
typedef struct ccl_function_table ccl_function_table;

// Create a function table for GPU function pointer dispatch
// size: Number of function entries in the table
// initial_kernel: Optional kernel to use for creating the table's pipeline.
//                 If NULL, the table will be created from the first kernel added via ccl_function_table_set.
//                 All kernels added to the table must be compatible with the table's pipeline.
ccl_error ccl_create_function_table(
    ccl_context *ctx,
    uint32_t size,
    ccl_kernel *initial_kernel,  // Optional: if NULL, table created lazily on first set
    ccl_function_table **out_table
);

// Set a function in the table at the given index
// kernel: The kernel to add to the table
// index: Table index (must be < size passed to ccl_create_function_table)
ccl_error ccl_function_table_set(
    ccl_function_table *table,
    ccl_kernel *kernel,
    uint32_t index
);

// Destroy a function table
void ccl_destroy_function_table(ccl_function_table *table);

// Binary Archives (Metal 3+)
// Binary archives cache compiled pipelines for faster startup
typedef struct ccl_binary_archive ccl_binary_archive;

// Create a binary archive for pipeline caching
ccl_error ccl_create_binary_archive(
    ccl_context *ctx,
    ccl_binary_archive **out_archive
);

// Add a compiled kernel to the archive
ccl_error ccl_binary_archive_add_kernel(
    ccl_binary_archive *archive,
    ccl_kernel *kernel
);

// Serialize archive to data (for saving to disk)
// Returns size needed in out_size if out_data is NULL
ccl_error ccl_binary_archive_serialize(
    ccl_binary_archive *archive,
    uint8_t *out_data,
    size_t *out_size
);

// Load archive from serialized data
ccl_error ccl_binary_archive_load(
    ccl_context *ctx,
    const uint8_t *data,
    size_t size,
    ccl_binary_archive **out_archive
);

// Destroy a binary archive
void ccl_destroy_binary_archive(ccl_binary_archive *archive);

// Ray Tracing (Metal 3+)
// Acceleration structures and ray tracing pipelines
typedef struct ccl_acceleration_structure ccl_acceleration_structure;
typedef struct ccl_raytracing_pipeline ccl_raytracing_pipeline;

// Create an acceleration structure for ray tracing
// geometry_count: Number of geometries
// Returns CCL_ERROR_NOT_SUPPORTED if ray tracing is not available
ccl_error ccl_create_acceleration_structure(
    ccl_context *ctx,
    uint32_t geometry_count,
    ccl_acceleration_structure **out_as
);

// Destroy an acceleration structure
void ccl_destroy_acceleration_structure(ccl_acceleration_structure *as);

// Create a ray tracing pipeline from source
ccl_error ccl_create_raytracing_pipeline_from_source(
    ccl_context *ctx,
    const char *source,
    const char *raygen_function,
    const char *intersection_function,  // optional, NULL if not used
    ccl_raytracing_pipeline **out_pipeline,
    char *log_buffer,
    size_t log_buffer_size
);

// Destroy a ray tracing pipeline
void ccl_destroy_raytracing_pipeline(ccl_raytracing_pipeline *pipeline);

// Indirect Command Buffers (Metal 3+)
typedef struct ccl_indirect_command_buffer ccl_indirect_command_buffer;

// Create an indirect command buffer
// max_commands: Maximum number of commands in the buffer
ccl_error ccl_create_indirect_command_buffer(
    ccl_context *ctx,
    uint32_t max_commands,
    ccl_indirect_command_buffer **out_icb
);

// Encode a compute command into the ICB
ccl_error ccl_indirect_command_buffer_encode_compute(
    ccl_indirect_command_buffer *icb,
    ccl_kernel *kernel,
    size_t dim,
    const size_t global_size[3],
    const size_t local_size[3],
    ccl_buffer **buffers,
    size_t num_buffers
);

// Execute an indirect command buffer
ccl_error ccl_execute_indirect_command_buffer(
    ccl_context *ctx,
    ccl_indirect_command_buffer *icb,
    uint32_t command_count,
    ccl_fence **out_fence
);

// Destroy an indirect command buffer
void ccl_destroy_indirect_command_buffer(ccl_indirect_command_buffer *icb);

// GPU Dynamic Libraries (Metal 4+)
// GPU Dynamic Libraries allow creating libraries that can be linked at runtime on the GPU
typedef struct ccl_gpu_dynamic_library ccl_gpu_dynamic_library;

// Create a GPU dynamic library from compiled library data
// This is a Metal 4 feature that enables runtime library linking on the GPU
ccl_error ccl_create_gpu_dynamic_library(
    ccl_context *ctx,
    const uint8_t *lib_data,
    size_t lib_size,
    ccl_gpu_dynamic_library **out_lib
);

// Create a kernel from a GPU dynamic library
// This allows using functions from dynamically linked libraries
ccl_error ccl_create_kernel_from_gpu_dynamic_library(
    ccl_context *ctx,
    ccl_gpu_dynamic_library *dyn_lib,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,
    size_t log_buffer_size
);

// Destroy a GPU dynamic library
void ccl_destroy_gpu_dynamic_library(ccl_gpu_dynamic_library *dyn_lib);

#ifdef __cplusplus
}
#endif

