/**
 * MTL Compute Library
 * A C/Objective-C library for Metal compute shaders
 */

#ifndef MTL_COMPUTE_H
#define MTL_COMPUTE_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque types for Metal objects
typedef struct MTLComputeDevice MTLComputeDevice;
typedef struct MTLComputeBuffer MTLComputeBuffer;
typedef struct MTLComputePipeline MTLComputePipeline;
typedef struct MTLComputeCommandList MTLComputeCommandList;
typedef struct MTLComputeSampler MTLComputeSampler;

// Error codes
typedef enum {
    MTL_SUCCESS = 0,
    MTL_ERROR_NO_DEVICE = -1,
    MTL_ERROR_SHADER_COMPILATION = -2,
    MTL_ERROR_PIPELINE_CREATION = -3,
    MTL_ERROR_BUFFER_CREATION = -4,
    MTL_ERROR_COMMAND_ENCODING = -5,
    MTL_ERROR_EXECUTION = -6,
    MTL_ERROR_INVALID_PARAMETER = -7,
    MTL_ERROR_IO = -8,
    MTL_ERROR_UNSUPPORTED = -9
} MTLComputeError;

// Buffer storage modes
typedef enum {
    MTL_STORAGE_SHARED = 0,    // CPU and GPU accessible
    MTL_STORAGE_PRIVATE = 1,   // GPU only
    MTL_STORAGE_MANAGED = 2    // Synchronized between CPU and GPU
} MTLComputeStorageMode;

/**
 * Initialize the Metal compute device
 * @return Pointer to device context, or NULL on failure
 */
MTLComputeDevice* mtl_compute_device_create(void);

/**
 * Release the Metal compute device
 * @param device Device context to release
 */
void mtl_compute_device_destroy(MTLComputeDevice* device);

/**
 * Get device name
 * @param device Device context
 * @return Device name string (do not free)
 */
const char* mtl_compute_device_get_name(MTLComputeDevice* device);

/**
 * Get raw MTLDevice (escape hatch for advanced users)
 * @param device Device context
 * @return Opaque pointer to id<MTLDevice> (cast to void*)
 */
void* mtl_compute_device_get_mtl_device(MTLComputeDevice* device);

/**
 * Check if device supports managed storage mode
 * @param device Device context
 * @return true if managed mode is supported (macOS only)
 */
bool mtl_compute_device_is_managed_supported(MTLComputeDevice* device);

/**
 * Compile a compute shader from source
 * @param device Device context
 * @param source Metal shader source code
 * @param function_name Name of the kernel function
 * @param error Output error code
 * @param error_log Optional buffer to receive detailed error message (can be NULL)
 * @param error_log_size Size of error_log buffer
 * @return Pipeline object, or NULL on failure
 */
MTLComputePipeline* mtl_compute_pipeline_create(
    MTLComputeDevice* device,
    const char* source,
    const char* function_name,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
);

/**
 * Compile a compute shader from file
 * @param device Device context
 * @param filepath Path to .metal shader file
 * @param function_name Name of the kernel function
 * @param error Output error code
 * @param error_log Optional buffer to receive detailed error message (can be NULL)
 * @param error_log_size Size of error_log buffer
 * @return Pipeline object, or NULL on failure
 */
MTLComputePipeline* mtl_compute_pipeline_create_from_file(
    MTLComputeDevice* device,
    const char* filepath,
    const char* function_name,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
);

// Compile options for pipeline creation (MSL 1.6.x)
typedef struct {
    const char* const* preprocessor_macros;  // NULL-terminated array, e.g. {"USE_SIMD=1", "K=16", NULL}
    const char* language_version;            // e.g. "3.0", "4.0", NULL for default
    bool fast_math_enabled;                  // Enable fast-math optimizations
    const char* const* additional_includes;  // NULL-terminated array of include paths
} MTLComputeShaderOptions;

/**
 * Compile a compute shader with compile options (MSL 1.6.x)
 * @param device Device context
 * @param source Metal shader source code
 * @param function_name Name of the kernel function
 * @param options Compile options (preprocessor macros, language version, etc.)
 * @param error Output error code
 * @param error_log Optional buffer to receive detailed error message (can be NULL)
 * @param error_log_size Size of error_log buffer
 * @return Pipeline object, or NULL on failure
 */
MTLComputePipeline* mtl_compute_pipeline_create_ex(
    MTLComputeDevice* device,
    const char* source,
    const char* function_name,
    const MTLComputeShaderOptions* options,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
);

/**
 * Load precompiled pipeline from .metallib file (offline compilation)
 * @param device Device context
 * @param metallib_path Path to .metallib file
 * @param function_name Name of the kernel function
 * @param error Output error code
 * @param error_log Optional buffer to receive detailed error message (can be NULL)
 * @param error_log_size Size of error_log buffer
 * @return Pipeline object, or NULL on failure
 */
MTLComputePipeline* mtl_compute_pipeline_create_from_metallib(
    MTLComputeDevice* device,
    const char* metallib_path,
    const char* function_name,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
);

// Function constants for pipeline specialization
typedef struct {
    size_t index;
    enum { MTL_CONSTANT_BOOL, MTL_CONSTANT_INT, MTL_CONSTANT_FLOAT } type;
    union {
        bool bool_value;
        int int_value;
        float float_value;
    } value;
} MTLComputeFunctionConstant;

/**
 * Compile a compute shader with function constants
 * @param device Device context
 * @param source Metal shader source code
 * @param function_name Name of the kernel function
 * @param constants Array of function constant values
 * @param constant_count Number of constants
 * @param error Output error code
 * @param error_log Optional buffer to receive detailed error message (can be NULL)
 * @param error_log_size Size of error_log buffer
 * @return Pipeline object, or NULL on failure
 */
MTLComputePipeline* mtl_compute_pipeline_create_with_constants(
    MTLComputeDevice* device,
    const char* source,
    const char* function_name,
    const MTLComputeFunctionConstant* constants,
    size_t constant_count,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
);

/**
 * Release a compute pipeline
 * @param pipeline Pipeline to release
 */
void mtl_compute_pipeline_destroy(MTLComputePipeline* pipeline);

/**
 * Get maximum total threads per threadgroup
 * @param pipeline Pipeline object
 * @return Maximum threads per threadgroup
 */
size_t mtl_compute_pipeline_max_threads_per_threadgroup(MTLComputePipeline* pipeline);

/**
 * Calculate optimal threadgroup size for 1D dispatch
 * @param pipeline Pipeline object
 * @param total_threads Total number of threads to dispatch
 * @param threads_per_group Output: optimal threads per threadgroup
 * @param num_groups Output: number of threadgroups needed
 */
void mtl_compute_auto_threadgroup_1d(
    MTLComputePipeline* pipeline,
    size_t total_threads,
    size_t* threads_per_group,
    size_t* num_groups
);

/**
 * Calculate optimal threadgroup size for 2D dispatch (image processing)
 * @param pipeline Pipeline object
 * @param grid_width Total width in threads
 * @param grid_height Total height in threads
 * @param threadgroup_width Output: optimal threadgroup width (typically 16)
 * @param threadgroup_height Output: optimal threadgroup height (typically 16)
 */
void mtl_compute_auto_threadgroup_2d(
    MTLComputePipeline* pipeline,
    size_t grid_width,
    size_t grid_height,
    size_t* threadgroup_width,
    size_t* threadgroup_height
);

/**
 * Validate if threadgroup size matches kernel's [[required_threads_per_threadgroup]]
 * @param pipeline Pipeline object
 * @param tg_width Threadgroup width
 * @param tg_height Threadgroup height
 * @param tg_depth Threadgroup depth
 * @return true if valid or no requirement, false if mismatch
 */
bool mtl_compute_pipeline_validate_threadgroup_size(
    MTLComputePipeline* pipeline,
    size_t tg_width,
    size_t tg_height,
    size_t tg_depth
);

/**
 * Pipeline resource reflection (MSL 5.2.1)
 */
typedef struct {
    uint32_t buffer_count;
    uint32_t texture_count;
    uint32_t sampler_count;
    uint32_t threadgroup_memory_length;
} MTLComputeResourceInfo;

/**
 * Get pipeline reflection info (what resources it expects)
 * @param pipeline Pipeline object
 * @param resource_info Output resource info
 * @return Error code
 */
MTLComputeError mtl_compute_pipeline_get_resource_info(
    MTLComputePipeline* pipeline,
    MTLComputeResourceInfo* resource_info
);

/**
 * Create a buffer
 * @param device Device context
 * @param size Buffer size in bytes
 * @param mode Storage mode
 * @return Buffer object, or NULL on failure
 */
MTLComputeBuffer* mtl_compute_buffer_create(
    MTLComputeDevice* device,
    size_t size,
    MTLComputeStorageMode mode
);

/**
 * Create a buffer with initial data
 * @param device Device context
 * @param data Initial data to copy
 * @param size Buffer size in bytes
 * @param mode Storage mode
 * @return Buffer object, or NULL on failure
 */
MTLComputeBuffer* mtl_compute_buffer_create_with_data(
    MTLComputeDevice* device,
    const void* data,
    size_t size,
    MTLComputeStorageMode mode
);

/**
 * Release a buffer
 * @param buffer Buffer to release
 */
void mtl_compute_buffer_destroy(MTLComputeBuffer* buffer);

/**
 * Get buffer contents (for CPU-accessible buffers)
 * @param buffer Buffer object
 * @return Pointer to buffer contents, or NULL if not accessible
 */
void* mtl_compute_buffer_contents(MTLComputeBuffer* buffer);

/**
 * Get buffer size
 * @param buffer Buffer object
 * @return Size in bytes
 */
size_t mtl_compute_buffer_size(MTLComputeBuffer* buffer);

/**
 * Mark buffer as modified (for managed buffers)
 * @param buffer Buffer object
 */
void mtl_compute_buffer_did_modify(MTLComputeBuffer* buffer);

/**
 * Synchronize buffer from GPU to CPU (for managed buffers)
 * @param buffer Buffer object
 * @param device Device context
 */
void mtl_compute_buffer_synchronize(MTLComputeBuffer* buffer, MTLComputeDevice* device);

/**
 * Upload data to buffer (handles staging for private buffers automatically)
 * @param device Device context
 * @param buffer Target buffer
 * @param src Source data
 * @param size Size in bytes
 * @return Error code
 */
MTLComputeError mtl_compute_buffer_upload(
    MTLComputeDevice* device,
    MTLComputeBuffer* buffer,
    const void* src,
    size_t size
);

/**
 * Download data from buffer (handles synchronization automatically)
 * @param device Device context
 * @param buffer Source buffer
 * @param dst Destination buffer
 * @param size Size in bytes
 * @return Error code
 */
MTLComputeError mtl_compute_buffer_download(
    MTLComputeDevice* device,
    MTLComputeBuffer* buffer,
    void* dst,
    size_t size
);

// ============================================================================
// SAMPLERS - Complete the buffer/texture/sampler resource triad
// ============================================================================

/**
 * Sampler filtering modes
 */
typedef enum {
    MTL_SAMPLER_FILTER_NEAREST = 0,
    MTL_SAMPLER_FILTER_LINEAR = 1
} MTLComputeSamplerFilter;

/**
 * Sampler address modes
 */
typedef enum {
    MTL_SAMPLER_ADDRESS_CLAMP_TO_EDGE = 0,
    MTL_SAMPLER_ADDRESS_REPEAT = 1,
    MTL_SAMPLER_ADDRESS_MIRRORED_REPEAT = 2,
    MTL_SAMPLER_ADDRESS_CLAMP_TO_ZERO = 3
} MTLComputeSamplerAddressMode;

/**
 * Sampler descriptor
 */
typedef struct {
    MTLComputeSamplerFilter min_filter;
    MTLComputeSamplerFilter mag_filter;
    MTLComputeSamplerFilter mip_filter;
    MTLComputeSamplerAddressMode address_mode_u;
    MTLComputeSamplerAddressMode address_mode_v;
    MTLComputeSamplerAddressMode address_mode_w;
    bool normalized_coordinates;
} MTLComputeSamplerDesc;

/**
 * Create a sampler
 * @param device Device context
 * @param desc Sampler descriptor
 * @return Sampler object, or NULL on failure
 */
MTLComputeSampler* mtl_compute_sampler_create(
    MTLComputeDevice* device,
    const MTLComputeSamplerDesc* desc
);

/**
 * Release sampler
 * @param sampler Sampler to release
 */
void mtl_compute_sampler_destroy(MTLComputeSampler* sampler);

// ============================================================================
// UNIFIED DISPATCH - Combine buffers and textures
// ============================================================================

// Forward declare texture type
typedef struct MTLComputeTexture MTLComputeTexture;

/**
 * Unified dispatch descriptor for mixed buffer/texture/sampler workloads
 * 
 * This is the canonical dispatch format - all other dispatch functions build this internally.
 * 
 * Field rules:
 * - pipeline: REQUIRED (must not be NULL)
 * - buffers/textures/samplers: can be NULL if corresponding count is 0
 * - buffer_count/texture_count/sampler_count: 0 = no resources of that type
 * - grid_width/height/depth: REQUIRED (must be > 0)
 * - threadgroup_width/height/depth: 0 = auto-select optimal size based on pipeline
 *   (if you specify 0, library will call mtl_compute_auto_threadgroup_* internally)
 */
typedef struct {
    MTLComputePipeline* pipeline;
    MTLComputeBuffer** buffers;
    size_t buffer_count;
    MTLComputeTexture** textures;
    size_t texture_count;
    MTLComputeSampler** samplers;
    size_t sampler_count;
    size_t grid_width;           // > 0 required
    size_t grid_height;          // > 0 required
    size_t grid_depth;           // > 0 required
    size_t threadgroup_width;    // 0 = auto
    size_t threadgroup_height;   // 0 = auto
    size_t threadgroup_depth;    // 0 = auto
} MTLComputeDispatchDesc;

/**
 * Execute a compute shader with unified descriptor (buffers + textures)
 * @param device Device context
 * @param desc Dispatch descriptor
 * @return Error code
 */
MTLComputeError mtl_compute_dispatch_desc(
    MTLComputeDevice* device,
    const MTLComputeDispatchDesc* desc
);

// ============================================================================
// ENCODER-BASED API - For batching multiple dispatches
// ============================================================================

/**
 * Begin encoding compute commands (for batching)
 * @param device Device context
 * @param out_command_list Output command list
 * @return Error code
 */
MTLComputeError mtl_compute_begin(
    MTLComputeDevice* device,
    MTLComputeCommandList** out_command_list
);

/**
 * Encode a dispatch into command list
 * @param command_list Command list from mtl_compute_begin
 * @param desc Dispatch descriptor
 * @return Error code
 */
MTLComputeError mtl_compute_encode_dispatch(
    MTLComputeCommandList* command_list,
    const MTLComputeDispatchDesc* desc
);

/**
 * End encoding and submit all commands (synchronous - waits for completion)
 * @param command_list Command list to submit
 * @return Error code
 */
MTLComputeError mtl_compute_end_submit(MTLComputeCommandList* command_list);

/**
 * End encoding and submit without waiting (asynchronous)
 * @param command_list Command list to submit
 * @return Error code
 */
MTLComputeError mtl_compute_end_submit_nowait(MTLComputeCommandList* command_list);

// ============================================================================
// IMMEDIATE DISPATCH - Original API (convenience wrappers)
// ============================================================================

/**
 * Execute a compute shader (synchronous, waits for completion)
 * @param device Device context
 * @param pipeline Compute pipeline
 * @param buffers Array of buffer arguments
 * @param buffer_count Number of buffers
 * @param grid_width Total number of threads in X dimension
 * @param grid_height Total number of threads in Y dimension
 * @param grid_depth Total number of threads in Z dimension
 * @param threadgroup_width Threads per threadgroup in X
 * @param threadgroup_height Threads per threadgroup in Y
 * @param threadgroup_depth Threads per threadgroup in Z
 * @return Error code
 */
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
);

/**
 * Execute a compute shader without waiting (asynchronous)
 * @param device Device context
 * @param pipeline Compute pipeline
 * @param buffers Array of buffer arguments
 * @param buffer_count Number of buffers
 * @param grid_width Total number of threads in X dimension
 * @param grid_height Total number of threads in Y dimension
 * @param grid_depth Total number of threads in Z dimension
 * @param threadgroup_width Threads per threadgroup in X
 * @param threadgroup_height Threads per threadgroup in Y
 * @param threadgroup_depth Threads per threadgroup in Z
 * @return Error code
 */
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
);

/**
 * Legacy alias for mtl_compute_dispatch_sync (backwards compatibility)
 */
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
);

/**
 * Execute a compute shader with 1D dispatch
 * @param device Device context
 * @param pipeline Compute pipeline
 * @param buffers Array of buffer arguments
 * @param buffer_count Number of buffers
 * @param total_threads Total number of threads
 * @param threads_per_group Threads per threadgroup
 * @return Error code
 */
MTLComputeError mtl_compute_dispatch_1d(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t total_threads,
    size_t threads_per_group
);

/**
 * Get error description
 * @param error Error code
 * @return Human-readable error description (stable pointer)
 */
const char* mtl_compute_error_string(MTLComputeError error);

// ============================================================================
// ARGUMENT BUFFERS - MSL 2.13 (layout-aware)
// ============================================================================

typedef struct MTLComputeArgumentBuffer MTLComputeArgumentBuffer;

/**
 * Argument kinds for layout-aware argument buffers
 */
typedef enum {
    MTL_ARG_BUFFER = 0,
    MTL_ARG_TEXTURE = 1,
    MTL_ARG_SAMPLER = 2
} MTLComputeArgKind;

/**
 * Argument descriptor for layout-aware argument buffers
 * Describes a single resource slot in the argument buffer.
 */
typedef struct {
    MTLComputeArgKind kind;  // Type of resource
    uint32_t index;          // Index in shader (matching [[buffer(N)]], [[texture(N)]], etc.)
} MTLComputeArgDesc;

/**
 * Create an argument buffer with explicit layout (MSL 2.13 style)
 * @param device Device context
 * @param layout Array of argument descriptors
 * @param layout_count Number of arguments in layout
 * @return Argument buffer object, or NULL if unsupported
 */
MTLComputeArgumentBuffer* mtl_compute_argbuf_create_layout(
    MTLComputeDevice* device,
    const MTLComputeArgDesc* layout,
    size_t layout_count
);

/**
 * Create a simple argument buffer (legacy, slot-based)
 * @param device Device context
 * @param max_buffers Maximum number of buffers
 * @param max_textures Maximum number of textures
 * @return Argument buffer object, or NULL if unsupported
 */
MTLComputeArgumentBuffer* mtl_compute_argbuf_create(
    MTLComputeDevice* device,
    size_t max_buffers,
    size_t max_textures
);

/**
 * Bind a buffer into the argument buffer
 * @param argbuf Argument buffer
 * @param index Index in the argument buffer
 * @param buffer Buffer to bind
 * @return Error code
 */
MTLComputeError mtl_compute_argbuf_set_buffer(
    MTLComputeArgumentBuffer* argbuf,
    uint32_t index,
    MTLComputeBuffer* buffer
);

/**
 * Bind a texture into the argument buffer
 * @param argbuf Argument buffer
 * @param index Index in the argument buffer
 * @param texture Texture to bind
 * @return Error code
 */
MTLComputeError mtl_compute_argbuf_set_texture(
    MTLComputeArgumentBuffer* argbuf,
    uint32_t index,
    MTLComputeTexture* texture
);

/**
 * Bind a sampler into the argument buffer
 * @param argbuf Argument buffer
 * @param index Index in the argument buffer
 * @param sampler Sampler to bind
 * @return Error code
 */
MTLComputeError mtl_compute_argbuf_set_sampler(
    MTLComputeArgumentBuffer* argbuf,
    uint32_t index,
    MTLComputeSampler* sampler
);

/**
 * Get the underlying buffer for binding (use this in dispatch)
 * @param argbuf Argument buffer
 * @return Buffer object
 */
MTLComputeBuffer* mtl_compute_argbuf_as_buffer(MTLComputeArgumentBuffer* argbuf);

/**
 * Release argument buffer
 * @param argbuf Argument buffer to release
 */
void mtl_compute_argbuf_destroy(MTLComputeArgumentBuffer* argbuf);

// ============================================================================
// FUNCTION TABLES - MSL 2.15, 5.1.4, 5.1.5
// ============================================================================

typedef struct MTLComputeFunctionTable MTLComputeFunctionTable;

/**
 * Create a function table for visible/stitchable functions (Metal 3+)
 * @param device Device context
 * @param pipeline Pipeline containing visible functions
 * @param max_functions Maximum number of functions in table
 * @return Function table object, or NULL if unsupported
 */
MTLComputeFunctionTable* mtl_compute_function_table_create(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    size_t max_functions
);

/**
 * Set a visible function at an index in the table
 * @param table Function table
 * @param index Index in the table
 * @param visible_function_name Name of the [[visible]] function
 * @return Error code (MTL_ERROR_UNSUPPORTED if not available)
 */
MTLComputeError mtl_compute_function_table_set(
    MTLComputeFunctionTable* table,
    uint32_t index,
    const char* visible_function_name
);

/**
 * Get the underlying buffer for binding (use this in dispatch)
 * @param table Function table
 * @return Buffer object
 */
MTLComputeBuffer* mtl_compute_function_table_as_buffer(MTLComputeFunctionTable* table);

/**
 * Release function table
 * @param table Function table to release
 */
void mtl_compute_function_table_destroy(MTLComputeFunctionTable* table);

// ============================================================================
// INDIRECT COMMAND BUFFERS - MSL 6.16
// ============================================================================

typedef struct MTLComputeIndirectCommandBuffer MTLComputeIndirectCommandBuffer;

/**
 * Create an indirect command buffer for pre-recorded dispatches (MSL 6.16)
 * @param device Device context
 * @param max_commands Maximum number of commands
 * @return ICB object, or NULL if unsupported
 */
MTLComputeIndirectCommandBuffer* mtl_compute_icb_create(
    MTLComputeDevice* device,
    size_t max_commands
);

/**
 * Encode a dispatch into the indirect command buffer
 * @param icb Indirect command buffer
 * @param command_index Index in the ICB
 * @param desc Dispatch descriptor
 * @return Error code
 */
MTLComputeError mtl_compute_icb_encode_dispatch(
    MTLComputeIndirectCommandBuffer* icb,
    uint32_t command_index,
    const MTLComputeDispatchDesc* desc
);

/**
 * Execute the indirect command buffer
 * @param device Device context
 * @param icb Indirect command buffer
 * @param num_commands Number of commands to execute
 * @return Error code
 */
MTLComputeError mtl_compute_icb_execute(
    MTLComputeDevice* device,
    MTLComputeIndirectCommandBuffer* icb,
    size_t num_commands
);

/**
 * Reset indirect command buffer for re-recording (avoids reallocation)
 * @param icb Indirect command buffer
 * @return Error code
 */
MTLComputeError mtl_compute_icb_reset(MTLComputeIndirectCommandBuffer* icb);

/**
 * Release indirect command buffer
 * @param icb ICB to release
 */
void mtl_compute_icb_destroy(MTLComputeIndirectCommandBuffer* icb);

// ============================================================================
// SIMD-GROUP & TENSOR HELPERS - MSL 6.7, 2.21
// ============================================================================

/**
 * Tensor descriptor for Metal tensor types (MSL 2.21)
 * 
 * Describes tensor layout for kernels using MSL tensor types.
 * All fields required (> 0).
 */
typedef struct {
    uint32_t width, height, depth;          // Dimensions (all must be > 0)
    uint32_t stride_x, stride_y, stride_z;  // Strides in elements
    uint32_t element_size_bytes;            // sizeof(element type)
} MTLComputeTensorDesc;

/**
 * Create a buffer with tensor descriptor metadata
 * @param device Device context
 * @param desc Tensor descriptor
 * @param data_size Size of tensor data in bytes
 * @param mode Storage mode
 * @return Buffer object with descriptor prepended
 */
MTLComputeBuffer* mtl_compute_buffer_create_for_tensor(
    MTLComputeDevice* device,
    const MTLComputeTensorDesc* desc,
    size_t data_size,
    MTLComputeStorageMode mode
);

/**
 * Memory ordering/barrier flags (MSL 6.9)
 */
typedef enum {
    MTL_MEMORY_SCOPE_DEVICE = 0,
    MTL_MEMORY_SCOPE_THREADGROUP = 1,
    MTL_MEMORY_SCOPE_TEXTURE = 2,
    MTL_MEMORY_SCOPE_SIMDGROUP = 3
} MTLComputeMemoryScope;

// ============================================================================
// DEVICE CAPABILITIES
// ============================================================================

/**
 * Device capabilities - runtime detection of hardware + library support
 * 
 * All features return true only if BOTH:
 * 1. Hardware/OS supports it (runtime Metal API check)
 * 2. Library was compiled with support (compile-time feature flag)
 * 
 * Functions return MTL_ERROR_UNSUPPORTED when capabilities are false.
 */
typedef struct {
    // Metal version support (hardware)
    bool supports_metal_3;
    bool supports_metal_4;
    
    // Core features
    bool supports_managed_storage;           // macOS only
    bool supports_non_uniform_threadgroups;
    
    // Advanced features (Metal 3/4) - hardware + library
    bool supports_shared_events;             // Async GPU-CPU sync
    bool supports_binary_archives;           // Pipeline caching
    bool supports_heaps;                     // Memory heaps
    bool supports_indirect_dispatch;         // GPU-driven dispatch
    bool supports_function_pointers;         // [[visible]] functions
    bool supports_argument_buffers;          // Argument buffers (tier 2)
    bool supports_indirect_command_buffers;  // ICB encode/execute
    bool supports_simdgroup_matrix;          // SIMD matrix ops
    bool supports_raytracing;                // Ray intersection (rare)
    
    // Device limits
    size_t max_threadgroup_memory;           // Bytes per threadgroup
    size_t max_threads_per_threadgroup;      // Total threads per group
    size_t recommended_max_working_set_size; // Bytes for optimal perf
    
    // Device info
    const char* device_name;                 // e.g. "Apple M2 Pro"
} MTLComputeDeviceCapabilities;

/**
 * Query device capabilities
 * @param device Device context
 * @param caps Output capabilities structure
 */
void mtl_compute_device_get_capabilities(
    MTLComputeDevice* device,
    MTLComputeDeviceCapabilities* caps
);

/**
 * Print comprehensive feature report (for debugging/introspection)
 * @param device Device context
 */
void mtl_compute_device_print_features(MTLComputeDevice* device);

/**
 * Create device by index (for multi-GPU systems)
 * @param device_index Index of the device (0 = default)
 * @param out_device Output device pointer
 * @return Error code
 */
MTLComputeError mtl_compute_device_create_with_index(
    uint32_t device_index,
    MTLComputeDevice** out_device
);

// ============================================================================
// PIPELINE LIBRARY & CACHING
// ============================================================================

typedef struct MTLComputePipelineLibrary MTLComputePipelineLibrary;

/**
 * Create a pipeline library for batch pipeline management
 * @param device Device context
 * @param descriptor_path Optional path to library descriptor file
 * @return Pipeline library object, or NULL on failure
 */
MTLComputePipelineLibrary* mtl_compute_pipeline_library_create(
    MTLComputeDevice* device,
    const char* descriptor_path
);

/**
 * Add a pipeline to the library
 * @param library Pipeline library
 * @param name Name for this pipeline
 * @param pipeline Pipeline to add
 * @return Error code
 */
MTLComputeError mtl_compute_pipeline_library_add(
    MTLComputePipelineLibrary* library,
    const char* name,
    MTLComputePipeline* pipeline
);

/**
 * Get a pipeline from the library by name
 * @param library Pipeline library
 * @param name Pipeline name
 * @return Pipeline object, or NULL if not found
 */
MTLComputePipeline* mtl_compute_pipeline_library_get(
    MTLComputePipelineLibrary* library,
    const char* name
);

/**
 * Release pipeline library
 * @param library Library to release
 */
void mtl_compute_pipeline_library_destroy(MTLComputePipelineLibrary* library);

// ============================================================================
// VALIDATION & DEBUGGING
// ============================================================================

/**
 * Validate a dispatch descriptor against a pipeline
 * @param pipeline Pipeline to validate against
 * @param desc Dispatch descriptor
 * @param error_log Optional buffer for validation errors
 * @param error_log_size Size of error log buffer
 * @return MTL_SUCCESS if valid, error code with details in log otherwise
 */
MTLComputeError mtl_compute_validate_dispatch(
    MTLComputePipeline* pipeline,
    const MTLComputeDispatchDesc* desc,
    char* error_log,
    size_t error_log_size
);

// ============================================================================
// DEBUG LABELS & LOGGING - MSL 5.1.12
// ============================================================================

/**
 * Set debug label on command list (appears in Xcode GPU debugger)
 * @param command_list Command list
 * @param label Debug label string
 */
void mtl_compute_command_list_set_label(
    MTLComputeCommandList* command_list,
    const char* label
);

/**
 * Set debug label on pipeline (appears in Xcode GPU debugger)
 * @param pipeline Pipeline
 * @param label Debug label string
 */
void mtl_compute_pipeline_set_label(
    MTLComputePipeline* pipeline,
    const char* label
);

/**
 * Set debug label on buffer
 * @param buffer Buffer
 * @param label Debug label string
 */
void mtl_compute_buffer_set_label(
    MTLComputeBuffer* buffer,
    const char* label
);

// ============================================================================
// ADVANCED FEATURES - Always available, runtime capability-checked
// ============================================================================

// Shared events for async GPU-CPU synchronization
typedef struct MTLComputeSharedEvent MTLComputeSharedEvent;

/**
 * Create a shared event for GPU-CPU synchronization
 * @param device Device context
 * @return Shared event object, or NULL if unsupported
 */
MTLComputeSharedEvent* mtl_compute_event_create(MTLComputeDevice* device);

/**
 * Wait for event to reach value (blocking)
 * @param event Shared event
 * @param value Value to wait for
 * @param timeout_ns Timeout in nanoseconds (0 = infinite)
 * @return true if completed, false if timeout or unsupported
 */
bool mtl_compute_event_wait(MTLComputeSharedEvent* event, uint64_t value, uint64_t timeout_ns);

/**
 * Check if event has reached value (non-blocking)
 * @param event Shared event
 * @param value Value to check
 * @return true if reached
 */
bool mtl_compute_event_check(MTLComputeSharedEvent* event, uint64_t value);

/**
 * Release shared event
 * @param event Event to release
 */
void mtl_compute_event_destroy(MTLComputeSharedEvent* event);

/**
 * Dispatch compute shader asynchronously with event signaling
 * @param device Device context
 * @param pipeline Compute pipeline
 * @param buffers Array of buffers
 * @param buffer_count Number of buffers
 * @param total_threads Total threads
 * @param threads_per_group Threads per threadgroup
 * @param event Event to signal when complete (can be NULL)
 * @param signal_value Value to signal
 * @return Error code (MTL_ERROR_UNSUPPORTED if events not available)
 */
MTLComputeError mtl_compute_dispatch_async(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t total_threads,
    size_t threads_per_group,
    MTLComputeSharedEvent* event,
    uint64_t signal_value
);

// Performance monitoring (enhanced)
typedef struct {
    double gpu_time_ms;
    double cpu_time_ms;
    uint64_t threads_executed;
    uint64_t threadgroups_executed;
    size_t memory_used_bytes;
    size_t threadgroup_memory_used;
    size_t execution_width;              // SIMD width actually used
    double throughput_gflops;
} MTLComputePerformanceStats;

/**
 * Dispatch with performance profiling
 * @param device Device context
 * @param pipeline Compute pipeline
 * @param buffers Array of buffers
 * @param buffer_count Number of buffers
 * @param total_threads Total threads
 * @param threads_per_group Threads per threadgroup
 * @param stats Output performance statistics
 * @return Error code
 */
MTLComputeError mtl_compute_dispatch_profiled(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t total_threads,
    size_t threads_per_group,
    MTLComputePerformanceStats* stats
);

// Indirect dispatch parameters
typedef struct {
    uint32_t threadgroups_x;
    uint32_t threadgroups_y;
    uint32_t threadgroups_z;
} MTLIndirectDispatchParams;

/**
 * GPU-driven indirect dispatch
 * @param device Device context
 * @param pipeline Compute pipeline
 * @param buffers Array of buffers
 * @param buffer_count Number of buffers
 * @param indirect_buffer Buffer containing dispatch params
 * @param indirect_offset Offset in indirect buffer
 * @param threads_per_group Threads per threadgroup
 * @return Error code (MTL_ERROR_UNSUPPORTED if not available)
 */
MTLComputeError mtl_compute_dispatch_indirect(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    MTLComputeBuffer* indirect_buffer,
    size_t indirect_offset,
    size_t threads_per_group
);

// Memory heaps for efficient sub-allocation
typedef struct MTLComputeHeap MTLComputeHeap;

// Compute pass for batched execution
typedef struct MTLComputePass MTLComputePass;

/**
 * Create a memory heap for sub-allocations
 * @param device Device context
 * @param size Heap size in bytes
 * @param mode Storage mode
 * @return Heap object, or NULL if unsupported
 */
MTLComputeHeap* mtl_compute_heap_create(
    MTLComputeDevice* device,
    size_t size,
    MTLComputeStorageMode mode
);

/**
 * Allocate buffer from heap
 * @param heap Heap object
 * @param size Buffer size
 * @return Buffer object, or NULL on failure
 */
MTLComputeBuffer* mtl_compute_buffer_alloc_from_heap(MTLComputeHeap* heap, size_t size);

/**
 * Get heap memory usage
 * @param heap Heap object
 * @param used Output bytes used
 * @param capacity Output total capacity
 */
void mtl_compute_heap_get_usage(MTLComputeHeap* heap, size_t* used, size_t* capacity);

/**
 * Release heap
 * @param heap Heap to release
 */
void mtl_compute_heap_destroy(MTLComputeHeap* heap);

// ============================================================================
// COMPUTE PASSES - Frame-based batch execution
// ============================================================================

/**
 * Create a compute pass (reusable batch of dispatches)
 * @param device Device context
 * @param max_dispatches Maximum number of dispatches in pass
 * @return Pass object, or NULL on failure
 */
MTLComputePass* mtl_compute_pass_create(MTLComputeDevice* device, size_t max_dispatches);

/**
 * Add a dispatch to the pass
 * @param pass Pass object
 * @param desc Dispatch descriptor (copied internally)
 * @return Error code
 */
MTLComputeError mtl_compute_pass_add_dispatch(
    MTLComputePass* pass,
    const MTLComputeDispatchDesc* desc
);

/**
 * Execute all dispatches in the pass
 * @param pass Pass object
 * @return Error code
 */
MTLComputeError mtl_compute_pass_execute(MTLComputePass* pass);

/**
 * Clear all dispatches from pass (for reuse)
 * @param pass Pass object
 */
void mtl_compute_pass_clear(MTLComputePass* pass);

/**
 * Release pass
 * @param pass Pass to release
 */
void mtl_compute_pass_destroy(MTLComputePass* pass);

// ============================================================================
// TENSOR UTILITIES - High-level abstractions
// ============================================================================

/**
 * Helper: Fill a tensor descriptor with row-major strides
 * @param desc Tensor descriptor to fill (dims must be set)
 */
void mtl_compute_tensor_make_row_major(MTLComputeTensorDesc* desc);

/**
 * Helper: Get data pointer from tensor buffer (skips descriptor header)
 * @param tensor_buffer Buffer created with mtl_compute_buffer_create_for_tensor
 * @return Pointer to tensor data, or NULL on failure
 */
void* mtl_compute_tensor_data_ptr(MTLComputeBuffer* tensor_buffer);

/**
 * Auto-tune threadgroup size for a pipeline
 * Benchmarks different threadgroup sizes and returns the fastest
 * @param device Device context
 * @param pipeline Pipeline to tune
 * @param total_threads Total thread count for workload
 * @param optimal_threadgroup_size Output: best threadgroup size found
 */
MTLComputeError mtl_compute_auto_tune(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    size_t total_threads,
    size_t* out_width,
    size_t* out_height,
    size_t* out_depth
);

/**
 * Tensor operation: Fill tensor with constant value
 * @param device Device context
 * @param tensor Tensor buffer
 * @param value Fill value
 * @return Error code
 */
MTLComputeError mtl_tensor_fill(
    MTLComputeDevice* device,
    MTLComputeBuffer* tensor,
    float value
);

/**
 * Tensor operation: SAXPY (y = alpha*x + y)
 * @param device Device context
 * @param alpha Scalar multiplier
 * @param x Input tensor X
 * @param y Input/output tensor Y
 * @param result Output tensor
 * @return Error code
 */
MTLComputeError mtl_tensor_saxpy(
    MTLComputeDevice* device,
    float alpha,
    MTLComputeBuffer* x,
    MTLComputeBuffer* y,
    MTLComputeBuffer* result
);

#ifdef __cplusplus
}
#endif

#endif // MTL_COMPUTE_H

