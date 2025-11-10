/**
 * MTLComp Internal Header
 * 
 * Private header containing internal struct definitions and helpers.
 * This is NOT exposed to users - it's only for the implementation modules.
 */

#ifndef MTL_INTERNAL_H
#define MTL_INTERNAL_H

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "mtl_compute.h"
#include "mtl_texture.h"

// ============================================================================
// INTERNAL STRUCTURES - Opaque to users
// ============================================================================

struct MTLComputeDevice {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    
    // Cached capabilities (computed once at creation)
    MTLComputeDeviceCapabilities caps;
    
    // Advanced resources (optional, created on demand)
    id<MTLSharedEvent> defaultSharedEvent;
    id<MTLBinaryArchive> defaultArchive;
};

struct MTLComputeBuffer {
    id<MTLBuffer> buffer;
};

struct MTLComputePipeline {
    id<MTLComputePipelineState> pipelineState;
    id<MTLFunction> function;
};

struct MTLComputeCommandList {
    id<MTLCommandBuffer> commandBuffer;
    id<MTLComputeCommandEncoder> computeEncoder;
    MTLComputeDevice* device;
};

struct MTLComputeTexture {
    id<MTLTexture> texture;
    size_t width;
    size_t height;
    size_t depth;
    int format;      // MTLComputePixelFormat
    int type;        // MTLComputeTextureType
};

struct MTLComputeSampler {
    id<MTLSamplerState> samplerState;
};

struct MTLComputeSharedEvent {
    id<MTLSharedEvent> event;
};

struct MTLComputeHeap {
    id<MTLHeap> heap;
};

struct MTLComputePipelineLibrary {
    NSMutableDictionary* pipelines;  // Maps NSString -> NSValue(pointer)
};

struct MTLComputeArgumentBuffer {
    id<MTLBuffer> backing_buffer;
    id<MTLArgumentEncoder> encoder;
    MTLComputeBuffer* wrapper;  // Wrapper to return via _as_buffer()
};

struct MTLComputeFunctionTable {
    id<MTLVisibleFunctionTable> table;
    MTLComputeBuffer* wrapper;
};

struct MTLComputeIndirectCommandBuffer {
    id<MTLIndirectCommandBuffer> icb;
    MTLComputeDevice* device;
    size_t max_commands;
};

// ============================================================================
// CENTRAL LOGGING
// ============================================================================

// Logger callback type
typedef void (*mtl_log_fn)(const char* message);

// Global logger (defaults to fprintf(stderr))
extern mtl_log_fn g_mtl_log;

// Set custom logger
void mtl_set_logger(mtl_log_fn fn);

// Internal logging macro
#define MTL_LOG(fmt, ...) do { \
    if (g_mtl_log) { \
        char buf[512]; \
        snprintf(buf, sizeof(buf), fmt, ##__VA_ARGS__); \
        g_mtl_log(buf); \
    } \
} while(0)

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

// Convert storage mode
static inline MTLResourceOptions mtl_convert_storage_mode(MTLComputeStorageMode mode) {
    switch (mode) {
        case MTL_STORAGE_SHARED:
            return MTLResourceStorageModeShared;
        case MTL_STORAGE_PRIVATE:
            return MTLResourceStorageModePrivate;
        case MTL_STORAGE_MANAGED:
#if TARGET_OS_OSX
            return MTLResourceStorageModeManaged;
#else
            return MTLResourceStorageModeShared;
#endif
        default:
            return MTLResourceStorageModeShared;
    }
}

// Copy error message to buffer
static inline void mtl_copy_error_log(const char* message, char* error_log, size_t error_log_size) {
    if (error_log && error_log_size > 0 && message) {
        strncpy(error_log, message, error_log_size - 1);
        error_log[error_log_size - 1] = '\0';
    }
}

// Detect device capabilities at runtime
void mtl_detect_capabilities(MTLComputeDevice* ctx);

// Set resources on encoder (implemented in mtl_dispatch.m)
void mtl_set_buffers(id<MTLComputeCommandEncoder> encoder, MTLComputeBuffer** buffers, size_t count);
void mtl_set_textures(id<MTLComputeCommandEncoder> encoder, MTLComputeTexture** textures, size_t count);
void mtl_set_samplers(id<MTLComputeCommandEncoder> encoder, MTLComputeSampler** samplers, size_t count);

// Unified encoder - THE single dispatch path
MTLComputeError mtl_encode_core(id<MTLComputeCommandEncoder> enc, const MTLComputeDispatchDesc* desc);

// Command buffer helpers (shared across modules)
id<MTLCommandBuffer> mtl_create_command_buffer(MTLComputeDevice* device);
id<MTLComputeCommandEncoder> mtl_create_compute_encoder(id<MTLCommandBuffer> commandBuffer);

#endif // MTL_INTERNAL_H

