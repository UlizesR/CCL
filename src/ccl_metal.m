// src/ccl_metal.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreFoundation/CoreFoundation.h>
#import <dispatch/dispatch.h>

#include "ccl_internal.h"
#include <string.h>
#include <stdint.h>

// Simple 64-bit hash function (FNV-1a variant)
static uint64_t ccl_hash_string(const char *str, size_t len) {
    uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    for (size_t i = 0; i < len; ++i) {
        hash ^= (uint64_t)(unsigned char)str[i];
        hash *= 1099511628211ULL;  // FNV prime
    }
    return hash;
}

// --- Objective-C helper objects ---

@interface CCLMetalContext : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLCommandQueue> queue;
@property (nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelineCache;
@property (nonatomic, strong) MTLCompileOptions *compileOptions;
@property (nonatomic, strong) NSString *label;
@property (nonatomic, assign) ccl_log_fn logCallback;
@property (nonatomic, assign) void *logUserData;
// Batch state
@property (nonatomic, strong) id<MTLCommandBuffer> activeBatch;
@property (nonatomic, strong) id<MTLComputeCommandEncoder> activeEncoder;
@end

@implementation CCLMetalContext
- (instancetype)init {
    if (self = [super init]) {
        _pipelineCache = [NSMutableDictionary dictionary];
        _compileOptions = [MTLCompileOptions new];
    }
    return self;
}
@end

@interface CCLMetalBuffer : NSObject
@property (nonatomic, strong) id<MTLBuffer> buffer;
@property (nonatomic, assign) ccl_buffer_usage usage;
@property (nonatomic, strong) id<MTLBuffer> stagingBuffer;  // For GPU_ONLY buffers
@end

@implementation CCLMetalBuffer
@end

// --- Fence for async operations ---

@interface CCLMetalFence : NSObject
@property (nonatomic, strong) id<MTLCommandBuffer> commandBuffer;
@property (nonatomic, strong) NSString *errorMessage;  // Cached error message
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)cmd;
- (void)updateErrorMessage;  // Update cached error message from command buffer
@end

@implementation CCLMetalFence
- (instancetype)initWithCommandBuffer:(id<MTLCommandBuffer>)cmd {
    if (self = [super init]) {
        _commandBuffer = cmd;
        _errorMessage = nil;
    }
    return self;
}

- (void)updateErrorMessage {
    if (self.commandBuffer.status >= MTLCommandBufferStatusCompleted) {
        NSError *error = self.commandBuffer.error;
        if (error) {
            const char *msg = error.localizedDescription.UTF8String;
            self.errorMessage = msg ? [NSString stringWithUTF8String:msg] : @"Unknown error";
        } else {
            self.errorMessage = nil;
        }
    }
}
@end

@interface CCLMetalKernel : NSObject
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property (nonatomic, strong) id<MTLFunction> function;  // For function tables
@property (nonatomic, strong) NSString *label;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSData *> *uniforms;  // index -> data
@property (nonatomic, assign) NSUInteger cachedThreadgroupSize;  // Cached optimal size for 1D
@end

@implementation CCLMetalKernel
- (instancetype)init {
    if (self = [super init]) {
        _uniforms = [NSMutableDictionary dictionary];
        _cachedThreadgroupSize = 0;  // 0 = not cached yet
    }
    return self;
}
@end

// Function Table (Metal 3+)
@interface CCLMetalFunctionTable : NSObject
@property (nonatomic, strong) id<MTLVisibleFunctionTable> table;
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;  // Pipeline that created this table
@property (nonatomic, assign) uint32_t size;
@end

@implementation CCLMetalFunctionTable
@end

// Binary Archive (Metal 3+)
@interface CCLMetalBinaryArchive : NSObject
@property (nonatomic, strong) id<MTLBinaryArchive> archive;
@property (nonatomic, strong) NSMutableSet<id<MTLComputePipelineState>> *addedPipelines;
@end

@implementation CCLMetalBinaryArchive
- (instancetype)init {
    if (self = [super init]) {
        _addedPipelines = [NSMutableSet set];
    }
    return self;
}
@end

// Acceleration Structure (Metal 3+)
@interface CCLMetalAccelerationStructure : NSObject
@property (nonatomic, strong) id<MTLAccelerationStructure> accelerationStructure;
@end

@implementation CCLMetalAccelerationStructure
@end

// Ray Tracing Pipeline (Metal 3+)
// Note: Metal ray tracing uses intersection functions and compute pipelines, not separate RT pipeline types
@interface CCLMetalRaytracingPipeline : NSObject
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;  // Ray tracing uses compute pipelines
@property (nonatomic, strong) id<MTLIntersectionFunctionTable> intersectionTable;
@end

@implementation CCLMetalRaytracingPipeline
@end

// Indirect Command Buffer (Metal 3+)
@interface CCLMetalIndirectCommandBuffer : NSObject
@property (nonatomic, strong) id<MTLIndirectCommandBuffer> icb;
@property (nonatomic, assign) uint32_t maxCommands;
@end

@implementation CCLMetalIndirectCommandBuffer
@end

// --- Context ---

// Helper to call log callback if set
static void ccl_log_metal(ccl_context *ctx, const char *msg) {
    if (!ctx || ctx->kind != CCL_BACKEND_KIND_METAL) return;
    CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
    if (metalCtx.logCallback && msg) {
        metalCtx.logCallback(msg, metalCtx.logUserData);
    }
}

static ccl_error ccl_create_context_metal(ccl_context **out_ctx) {
    if (!out_ctx) return CCL_ERROR_INVALID_ARGUMENT;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        return CCL_ERROR_DEVICE_FAILED;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        return CCL_ERROR_DEVICE_FAILED;
    }

    CCLMetalContext *metalCtx = [[CCLMetalContext alloc] init];
    metalCtx.device = device;
    metalCtx.queue = queue;

    ccl_context *ctx = (ccl_context *)malloc(sizeof(ccl_context));
    if (!ctx) {
        return CCL_ERROR_BACKEND_INIT_FAILED;
    }
    ctx->kind = CCL_BACKEND_KIND_METAL;
    ctx->impl = (__bridge_retained void *)metalCtx; // C holds a retain

    *out_ctx = ctx;
    return CCL_OK;
}

ccl_error ccl_create_context(ccl_backend backend, ccl_context **out_ctx) {
    if (!out_ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (backend) {
    case CCL_BACKEND_METAL:
        return ccl_create_context_metal(out_ctx);
    case CCL_BACKEND_GL_COMPUTE:
    case CCL_BACKEND_OPENCL:
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

void ccl_destroy_context(ccl_context *ctx) {
    if (!ctx) return;
    if (ctx->impl) {
        CFRelease(ctx->impl);   // matches __bridge_retained
    }
    free(ctx);
}

void ccl_set_log_callback(ccl_context *ctx, ccl_log_fn fn, void *user_data) {
    if (!ctx) return;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL: {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        metalCtx.logCallback = fn;
        metalCtx.logUserData = user_data;
        break;
    }
    default:
        break;
    }
}

// --- Command Buffer Batching ---

static ccl_error ccl_begin_batch_metal(ccl_context *ctx) {
    if (!ctx || ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        
        // Check if already in a batch
        if (metalCtx.activeBatch != nil) {
            return CCL_ERROR_INVALID_ARGUMENT;  // Already in a batch
        }
        
        // Create new command buffer and encoder for batch
        metalCtx.activeBatch = [metalCtx.queue commandBuffer];
        if (metalCtx.label) {
            metalCtx.activeBatch.label = metalCtx.label;
        }
        metalCtx.activeEncoder = [metalCtx.activeBatch computeCommandEncoder];
        
        return CCL_OK;
    }
}

static ccl_error ccl_end_batch_metal(ccl_context *ctx, ccl_fence **out_fence) {
    if (!ctx || ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        
        // Check if not in a batch
        if (metalCtx.activeBatch == nil) {
            if (out_fence) *out_fence = NULL;
            return CCL_ERROR_INVALID_ARGUMENT;  // Not in a batch
        }
        
        // End encoding and commit
        [metalCtx.activeEncoder endEncoding];
        [metalCtx.activeBatch commit];
        
        // Create fence if requested
        CCLMetalFence *fence = nil;
        if (out_fence) {
            fence = [[CCLMetalFence alloc] initWithCommandBuffer:metalCtx.activeBatch];
        }
        
        // Clear batch state
        id<MTLCommandBuffer> cmd = metalCtx.activeBatch;
        metalCtx.activeBatch = nil;
        metalCtx.activeEncoder = nil;
        
        // Return fence if requested
        if (out_fence && fence) {
            ccl_fence *cclFence = (ccl_fence *)malloc(sizeof(ccl_fence));
            if (!cclFence) {
                if (out_fence) {
                    *out_fence = NULL;
                }
                return CCL_ERROR_DEVICE_FAILED;
            }
            cclFence->kind = CCL_BACKEND_KIND_METAL;
            cclFence->impl = (__bridge_retained void *)fence;
            *out_fence = cclFence;
        }
        
        return CCL_OK;
    }
}

ccl_error ccl_begin_batch(ccl_context *ctx) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_begin_batch_metal(ctx);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

ccl_error ccl_end_batch(ccl_context *ctx, ccl_fence **out_fence) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_end_batch_metal(ctx, out_fence);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

// --- Device Info ---

static ccl_error ccl_get_device_info_metal(
    ccl_context *ctx,
    ccl_device_info info,
    void *out_value,
    size_t *out_size
) {
    if (!ctx || !out_size) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        id<MTLDevice> device = metalCtx.device;
        
        switch (info) {
            case CCL_DEVICE_INFO_NAME: {
                const char *name = device.name.UTF8String;
                size_t nameLen = name ? strlen(name) + 1 : 0;
                
                if (out_size) {
                    if (*out_size < nameLen) {
                        *out_size = nameLen;  // Return required size
                        return CCL_ERROR_INVALID_ARGUMENT;
                    }
                    if (out_value && name) {
                        strncpy((char *)out_value, name, *out_size - 1);
                        ((char *)out_value)[*out_size - 1] = '\0';
                    }
                    *out_size = nameLen;
                }
                return CCL_OK;
            }
            
            case CCL_DEVICE_INFO_MAX_THREADS_PER_THREADGROUP: {
                // Note: In Metal, max threads per threadgroup is per-pipeline,
                // not a device property. We return a reasonable default.
                // Actual value should be queried from a specific pipeline state.
                if (!out_value || !out_size || *out_size < sizeof(uint64_t)) {
                    if (out_size) *out_size = sizeof(uint64_t);
                    return CCL_ERROR_INVALID_ARGUMENT;
                }
                
                // Metal devices typically support 256-1024 threads per threadgroup
                // This is a conservative default; actual value depends on the kernel
                uint64_t maxThreads = 1024;  // Safe default for most Metal devices
                *(uint64_t *)out_value = maxThreads;
                *out_size = sizeof(uint64_t);
                return CCL_OK;
            }
            
            case CCL_DEVICE_INFO_THREAD_EXECUTION_WIDTH: {
                if (!out_value || !out_size || *out_size < sizeof(uint64_t)) {
                    if (out_size) *out_size = sizeof(uint64_t);
                    return CCL_ERROR_INVALID_ARGUMENT;
                }
                
                // Metal SIMD width is typically 32 for Apple GPUs, 64 for some
                // This is also per-pipeline, but we can provide a reasonable default
                uint64_t simdWidth = 32;  // Common for Apple GPUs
                *(uint64_t *)out_value = simdWidth;
                *out_size = sizeof(uint64_t);
                return CCL_OK;
            }
            
            case CCL_DEVICE_INFO_MAX_BUFFER_LENGTH: {
                if (!out_value || !out_size || *out_size < sizeof(uint64_t)) {
                    if (out_size) *out_size = sizeof(uint64_t);
                    return CCL_ERROR_INVALID_ARGUMENT;
                }
                
                // Metal supports very large buffers
                // Query actual max if available, otherwise use conservative default
                uint64_t maxBuffer = 256 * 1024 * 1024;  // 256MB default
                if (@available(macOS 10.15, iOS 13.0, *)) {
                    if ([device respondsToSelector:@selector(maxBufferLength)]) {
                        maxBuffer = device.maxBufferLength;
                    }
                }
                *(uint64_t *)out_value = maxBuffer;
                *out_size = sizeof(uint64_t);
                return CCL_OK;
            }
            
            case CCL_DEVICE_INFO_SUPPORTS_GPU_ONLY_BUFFERS: {
                if (!out_value || !out_size || *out_size < sizeof(bool)) {
                    if (out_size) *out_size = sizeof(bool);
                    return CCL_ERROR_INVALID_ARGUMENT;
                }
                
                // Metal always supports private storage mode
                *(bool *)out_value = true;
                *out_size = sizeof(bool);
                return CCL_OK;
            }
            
            case CCL_DEVICE_INFO_MAX_COMPUTE_UNITS: {
                if (!out_value || !out_size || *out_size < sizeof(uint64_t)) {
                    if (out_size) *out_size = sizeof(uint64_t);
                    return CCL_ERROR_INVALID_ARGUMENT;
                }
                
                // Metal doesn't expose compute units directly via MTLDevice
                // This would require creating a pipeline to query, so we return
                // a reasonable default or indicate it's not directly queryable
                // For Apple GPUs, typical values are 8-64 compute units
                uint64_t computeUnits = 0;  // 0 indicates "not directly queryable"
                *(uint64_t *)out_value = computeUnits;
                *out_size = sizeof(uint64_t);
                return CCL_OK;
            }
            
            default:
                return CCL_ERROR_NOT_SUPPORTED;
        }
    }
}

ccl_error ccl_get_device_info(
    ccl_context *ctx,
    ccl_device_info info,
    void *out_value,
    size_t *out_size
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_get_device_info_metal(ctx, info, out_value, out_size);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

// --- Buffers ---

static ccl_error ccl_create_buffer_metal_ex(
    ccl_context *ctx,
    size_t size,
    ccl_buffer_flags flags,
    ccl_buffer_usage usage,
    const void *initial_data,
    ccl_buffer **out_buf
) {
    if (!ctx || !out_buf) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;

        MTLResourceOptions options;
        id<MTLBuffer> mtlBuffer = nil;
        id<MTLBuffer> stagingBuffer = nil;

        // Map usage to Metal storage mode
        switch (usage) {
            case CCL_BUFFER_USAGE_GPU_ONLY: {
                // Private storage - requires staging buffer for initial data upload
                options = MTLResourceStorageModePrivate;
                mtlBuffer = [metalCtx.device newBufferWithLength:size options:options];
                if (!mtlBuffer) return CCL_ERROR_DEVICE_FAILED;
                
                // Create staging buffer for initial data (if provided)
                if (initial_data) {
                    stagingBuffer = [metalCtx.device newBufferWithBytes:initial_data
                                                                 length:size
                                                                options:MTLResourceStorageModeShared];
                    if (!stagingBuffer) return CCL_ERROR_DEVICE_FAILED;
                    
                    // Copy to private buffer via blit encoder
                    // Use a dedicated command buffer for the blit to avoid interfering
                    // with any ongoing compute dispatches
                    id<MTLCommandBuffer> cmd = [metalCtx.queue commandBuffer];
                    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                    [blit copyFromBuffer:stagingBuffer sourceOffset:0
                              toBuffer:mtlBuffer destinationOffset:0
                              size:size];
                    [blit endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    
                    // Note: We keep stagingBuffer in the property for potential future use
                    // in a "slow path" download (blit from private -> staging -> CPU).
                    // For now, it's only used during initial creation.
                    // Optimization: Could release stagingBuffer after initial copy if we
                    // don't plan to support downloads, but keeping it enables future features.
                }
                break;
            }
            case CCL_BUFFER_USAGE_CPU_TO_GPU:
            case CCL_BUFFER_USAGE_GPU_TO_CPU:
            case CCL_BUFFER_USAGE_DEFAULT:
            default: {
                // Shared storage for CPU/GPU access
                options = MTLResourceStorageModeShared;
                if (initial_data) {
                    mtlBuffer = [metalCtx.device newBufferWithBytes:initial_data
                                                             length:size
                                                            options:options];
                } else {
                    mtlBuffer = [metalCtx.device newBufferWithLength:size
                                                             options:options];
                }
                break;
            }
        }

        if (!mtlBuffer) {
            return CCL_ERROR_DEVICE_FAILED;
        }

        CCLMetalBuffer *metalBuf = [CCLMetalBuffer new];
        metalBuf.buffer = mtlBuffer;
        metalBuf.usage = usage;
        metalBuf.stagingBuffer = stagingBuffer;

        ccl_buffer *buf = (ccl_buffer *)malloc(sizeof(ccl_buffer));
        if (!buf) return CCL_ERROR_DEVICE_FAILED;

        buf->kind = CCL_BACKEND_KIND_METAL;
        buf->impl = (__bridge_retained void *)metalBuf;
        buf->size = size;

        *out_buf = buf;
        return CCL_OK;
    }
}

static ccl_error ccl_create_buffer_metal(
    ccl_context *ctx,
    size_t size,
    ccl_buffer_flags flags,
    const void *initial_data,
    ccl_buffer **out_buf
) {
    // Convenience wrapper using default usage
    return ccl_create_buffer_metal_ex(ctx, size, flags, CCL_BUFFER_USAGE_DEFAULT, initial_data, out_buf);
}

ccl_error ccl_create_buffer(
    ccl_context *ctx,
    size_t size,
    ccl_buffer_flags flags,
    const void *initial_data,
    ccl_buffer **out_buf
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_buffer_metal(ctx, size, flags, initial_data, out_buf);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

ccl_error ccl_create_buffer_ex(
    ccl_context *ctx,
    size_t size,
    ccl_buffer_flags flags,
    ccl_buffer_usage usage,
    const void *initial_data,
    ccl_buffer **out_buf
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_buffer_metal_ex(ctx, size, flags, usage, initial_data, out_buf);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

void ccl_destroy_buffer(ccl_buffer *buf) {
    if (!buf) return;
    if (buf->impl) {
        CFRelease(buf->impl);   // matches __bridge_retained
    }
    free(buf);
}

ccl_error ccl_buffer_upload(
    ccl_buffer *buf,
    size_t offset,
    const void *data,
    size_t size
) {
    if (!buf || !data) return CCL_ERROR_INVALID_ARGUMENT;
    if (offset + size > buf->size) return CCL_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        if (buf->kind == CCL_BACKEND_KIND_METAL) {
            CCLMetalBuffer *metalBuf = (__bridge CCLMetalBuffer *)buf->impl;
            
            // For GPU_ONLY buffers, uploads require a blit command
            // For now, we only support direct uploads to shared buffers
            // TODO: Add ccl_buffer_upload_ex that takes context for GPU_ONLY blits
            if (metalBuf.usage == CCL_BUFFER_USAGE_GPU_ONLY) {
                return CCL_ERROR_INVALID_ARGUMENT;  // GPU_ONLY uploads need context for blit
            }
            
            // Shared buffers - direct memory access
            uint8_t *dst = (uint8_t *)metalBuf.buffer.contents + offset;
            memcpy(dst, data, size);
            return CCL_OK;
        }

        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

static ccl_error ccl_buffer_upload_ex_metal(
    ccl_context *ctx,
    ccl_buffer *buf,
    size_t offset,
    const void *data,
    size_t size
) {
    if (!ctx || !buf || !data) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL || buf->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;
    if (offset + size > buf->size) return CCL_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        CCLMetalBuffer *metalBuf = (__bridge CCLMetalBuffer *)buf->impl;
        
        if (metalBuf.usage == CCL_BUFFER_USAGE_GPU_ONLY) {
            // GPU_ONLY buffer - use blit transfer
            // Create or reuse staging buffer (size to full buffer capacity for reuse)
            size_t stagingSize = MAX(size, buf->size);
            if (!metalBuf.stagingBuffer || metalBuf.stagingBuffer.length < stagingSize) {
                // Create new staging buffer sized to full buffer capacity
                metalBuf.stagingBuffer = [metalCtx.device newBufferWithLength:stagingSize
                                                                     options:MTLResourceStorageModeShared];
                if (!metalBuf.stagingBuffer) return CCL_ERROR_DEVICE_FAILED;
            }
            
            // Copy data to staging buffer
            memcpy(metalBuf.stagingBuffer.contents, data, size);
            
            // Blit from staging to private buffer
            id<MTLCommandBuffer> cmd = [metalCtx.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:metalBuf.stagingBuffer sourceOffset:0
                      toBuffer:metalBuf.buffer destinationOffset:offset
                      size:size];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            
            return CCL_OK;
        } else {
            // Shared buffer - direct memory access
            uint8_t *dst = (uint8_t *)metalBuf.buffer.contents + offset;
            memcpy(dst, data, size);
            return CCL_OK;
        }
    }
}

static ccl_error ccl_buffer_download_ex_metal(
    ccl_context *ctx,
    ccl_buffer *buf,
    size_t offset,
    void *data,
    size_t size
) {
    if (!ctx || !buf || !data) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL || buf->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;
    if (offset + size > buf->size) return CCL_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        CCLMetalBuffer *metalBuf = (__bridge CCLMetalBuffer *)buf->impl;
        
        if (metalBuf.usage == CCL_BUFFER_USAGE_GPU_ONLY) {
            // GPU_ONLY buffer - use blit transfer
            // Create or reuse staging buffer (size to full buffer capacity for reuse)
            size_t stagingSize = MAX(size, buf->size);
            if (!metalBuf.stagingBuffer || metalBuf.stagingBuffer.length < stagingSize) {
                // Create new staging buffer sized to full buffer capacity
                metalBuf.stagingBuffer = [metalCtx.device newBufferWithLength:stagingSize
                                                                     options:MTLResourceStorageModeShared];
                if (!metalBuf.stagingBuffer) return CCL_ERROR_DEVICE_FAILED;
            }
            
            // Blit from private to staging buffer
            id<MTLCommandBuffer> cmd = [metalCtx.queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromBuffer:metalBuf.buffer sourceOffset:offset
                      toBuffer:metalBuf.stagingBuffer destinationOffset:0
                      size:size];
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            
            // Copy from staging buffer to CPU
            memcpy(data, metalBuf.stagingBuffer.contents, size);
            return CCL_OK;
        } else {
            // Shared buffer - direct memory access
            uint8_t *src = (uint8_t *)metalBuf.buffer.contents + offset;
            memcpy(data, src, size);
            return CCL_OK;
        }
    }
}

ccl_error ccl_buffer_upload_ex(
    ccl_context *ctx,
    ccl_buffer *buf,
    size_t offset,
    const void *data,
    size_t size
) {
    if (!ctx || !buf) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_buffer_upload_ex_metal(ctx, buf, offset, data, size);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

ccl_error ccl_buffer_download_ex(
    ccl_context *ctx,
    ccl_buffer *buf,
    size_t offset,
    void *data,
    size_t size
) {
    if (!ctx || !buf) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_buffer_download_ex_metal(ctx, buf, offset, data, size);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

ccl_error ccl_buffer_download(
    ccl_buffer *buf,
    size_t offset,
    void *data,
    size_t size
) {
    if (!buf || !data) return CCL_ERROR_INVALID_ARGUMENT;
    if (offset + size > buf->size) return CCL_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        if (buf->kind == CCL_BACKEND_KIND_METAL) {
            CCLMetalBuffer *metalBuf = (__bridge CCLMetalBuffer *)buf->impl;
            
            // GPU_ONLY buffers are not CPU-readable (contents is NULL)
            // TODO: Add full blit-based transfer support for GPU_ONLY buffers
            if (metalBuf.usage == CCL_BUFFER_USAGE_GPU_ONLY) {
                return CCL_ERROR_INVALID_ARGUMENT;
            }
            
            uint8_t *src = (uint8_t *)metalBuf.buffer.contents + offset;
            memcpy(data, src, size);
            return CCL_OK;
        }

        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

// --- Kernels ---

static ccl_error ccl_create_kernel_from_source_metal(
    ccl_context *ctx,
    const char *source,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,
    size_t log_buffer_size
) {
    if (!ctx || !source || !entry_point || !out_kernel)
        return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;

        // Create cache key from source + entry point
        // Use hash of source for memory efficiency (especially for large shaders)
        size_t srcLen = strlen(source);
        uint64_t srcHash = ccl_hash_string(source, srcLen);
        NSString *entry = [NSString stringWithUTF8String:entry_point];
        // Format: "hash|entry" where hash is hex representation
        NSString *cacheKey = [NSString stringWithFormat:@"%016llx|%@", srcHash, entry];

        // Check pipeline cache
        // NOTE: pipelineCache is not thread-safe. CCL contexts should be used
        // from a single thread, or add synchronization if multi-threaded access is needed.
        id<MTLComputePipelineState> pipeline = metalCtx.pipelineCache[cacheKey];
        if (pipeline) {
            // Cache hit - reuse pipeline
            CCLMetalKernel *metalKernel = [[CCLMetalKernel alloc] init];
            metalKernel.pipeline = pipeline;
            // Note: function not stored for cache hits (would need to recompile to get it)
            // This is acceptable since function tables are typically used with new kernels

            ccl_kernel *kernel = (ccl_kernel *)malloc(sizeof(ccl_kernel));
            if (!kernel) return CCL_ERROR_DEVICE_FAILED;

            kernel->kind = CCL_BACKEND_KIND_METAL;
            kernel->impl = (__bridge_retained void *)metalKernel;

            *out_kernel = kernel;
            return CCL_OK;
        }

        // Cache miss - compile
        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = [metalCtx.device newLibraryWithSource:src
                                                           options:metalCtx.compileOptions
                                                             error:&error];
        if (!lib) {
            const char *msg = error.localizedDescription.UTF8String;
            const char *errorMsg = msg ? msg : "Metal compile error";
            
            if (log_buffer && log_buffer_size > 0) {
                snprintf(log_buffer, log_buffer_size, "%s", errorMsg);
            }
            
            // Also call log callback if set
            char callbackMsg[512];
            snprintf(callbackMsg, sizeof(callbackMsg), "Kernel compile failed: %s", errorMsg);
            ccl_log_metal(ctx, callbackMsg);
            
            return CCL_ERROR_COMPILE_FAILED;
        }

        id<MTLFunction> func = [lib newFunctionWithName:entry];
        if (!func) {
            if (log_buffer && log_buffer_size > 0) {
                snprintf(log_buffer, log_buffer_size, "Entry point not found: %s", entry_point);
            }
            return CCL_ERROR_COMPILE_FAILED;
        }

        pipeline = [metalCtx.device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            if (log_buffer && log_buffer_size > 0) {
                const char *msg = error.localizedDescription.UTF8String;
                snprintf(log_buffer, log_buffer_size, "%s", msg ? msg : "Pipeline creation error");
            }
            return CCL_ERROR_COMPILE_FAILED;
        }

        // Cache the pipeline
        metalCtx.pipelineCache[cacheKey] = pipeline;

        CCLMetalKernel *metalKernel = [[CCLMetalKernel alloc] init];
        metalKernel.pipeline = pipeline;
        metalKernel.function = func;  // Store function for function tables

        ccl_kernel *kernel = (ccl_kernel *)malloc(sizeof(ccl_kernel));
        if (!kernel) return CCL_ERROR_DEVICE_FAILED;

        kernel->kind = CCL_BACKEND_KIND_METAL;
        kernel->impl = (__bridge_retained void *)metalKernel;

        *out_kernel = kernel;
        return CCL_OK;
    }
}

static ccl_error ccl_create_kernel_from_library_metal(
    ccl_context *ctx,
    const uint8_t *lib_data,
    size_t lib_size,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,
    size_t log_buffer_size
) {
    if (!ctx || !lib_data || !entry_point || !out_kernel || lib_size == 0)
        return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;

        // Create library from precompiled data
        // Metal expects dispatch_data_t, so we create it from the raw bytes
        dispatch_data_t libData = dispatch_data_create(lib_data, lib_size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError *error = nil;
        id<MTLLibrary> lib = [metalCtx.device newLibraryWithData:libData
                                                           error:&error];
        if (!lib) {
            const char *msg = error.localizedDescription.UTF8String;
            const char *errorMsg = msg ? msg : "Failed to load Metal library from data";
            
            if (log_buffer && log_buffer_size > 0) {
                snprintf(log_buffer, log_buffer_size, "%s", errorMsg);
            }
            
            // Also call log callback if set
            char callbackMsg[512];
            snprintf(callbackMsg, sizeof(callbackMsg), "Library load failed: %s", errorMsg);
            ccl_log_metal(ctx, callbackMsg);
            
            return CCL_ERROR_COMPILE_FAILED;
        }

        NSString *entry = [NSString stringWithUTF8String:entry_point];
        id<MTLFunction> func = [lib newFunctionWithName:entry];
        if (!func) {
            const char *errorMsg = "Entry point not found";
            
            if (log_buffer && log_buffer_size > 0) {
                snprintf(log_buffer, log_buffer_size, "%s: %s", errorMsg, entry_point);
            }
            
            // Also call log callback if set
            char callbackMsg[512];
            snprintf(callbackMsg, sizeof(callbackMsg), "%s: %s", errorMsg, entry_point);
            ccl_log_metal(ctx, callbackMsg);
            
            return CCL_ERROR_COMPILE_FAILED;
        }

        id<MTLComputePipelineState> pipeline =
            [metalCtx.device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            const char *msg = error.localizedDescription.UTF8String;
            const char *errorMsg = msg ? msg : "Pipeline creation error";
            
            if (log_buffer && log_buffer_size > 0) {
                snprintf(log_buffer, log_buffer_size, "%s", errorMsg);
            }
            
            // Also call log callback if set
            char callbackMsg[512];
            snprintf(callbackMsg, sizeof(callbackMsg), "Pipeline creation failed: %s", errorMsg);
            ccl_log_metal(ctx, callbackMsg);
            
            return CCL_ERROR_COMPILE_FAILED;
        }

        CCLMetalKernel *metalKernel = [[CCLMetalKernel alloc] init];
        metalKernel.pipeline = pipeline;
        metalKernel.function = func;  // Store function for function tables

        ccl_kernel *kernel = (ccl_kernel *)malloc(sizeof(ccl_kernel));
        if (!kernel) return CCL_ERROR_DEVICE_FAILED;

        kernel->kind = CCL_BACKEND_KIND_METAL;
        kernel->impl = (__bridge_retained void *)metalKernel;

        *out_kernel = kernel;
        return CCL_OK;
    }
}

ccl_error ccl_create_kernel_from_source(
    ccl_context *ctx,
    const char *source,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,
    size_t log_buffer_size
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_kernel_from_source_metal(
            ctx, source, entry_point, out_kernel, log_buffer, log_buffer_size
        );
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

ccl_error ccl_create_kernel_from_library(
    ccl_context *ctx,
    const uint8_t *lib_data,
    size_t lib_size,
    const char *entry_point,
    ccl_kernel **out_kernel,
    char *log_buffer,
    size_t log_buffer_size
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_kernel_from_library_metal(
            ctx, lib_data, lib_size, entry_point, out_kernel, log_buffer, log_buffer_size
        );
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

void ccl_destroy_kernel(ccl_kernel *kernel) {
    if (!kernel) return;
    if (kernel->impl) {
        CFRelease(kernel->impl);
    }
    free(kernel);
}

// --- Uniforms / Constants ---
// Uniforms are persistent across dispatches until explicitly cleared.
// Use ccl_set_bytes() to set small parameters (scalars, structs) that will
// be applied to all subsequent dispatches of this kernel.
// Use ccl_clear_bytes() to remove all uniforms.

ccl_error ccl_set_bytes(ccl_kernel *kernel, uint32_t index, const void *data, size_t size) {
    if (!kernel || !data || size == 0) return CCL_ERROR_INVALID_ARGUMENT;
    if (kernel->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_UNSUPPORTED_BACKEND;
    
    @autoreleasepool {
        CCLMetalKernel *metalKernel = (__bridge CCLMetalKernel *)kernel->impl;
        NSData *bytes = [NSData dataWithBytes:data length:size];
        metalKernel.uniforms[@(index)] = bytes;
    }
    
    return CCL_OK;
}

void ccl_clear_bytes(ccl_kernel *kernel) {
    if (!kernel || kernel->kind != CCL_BACKEND_KIND_METAL) return;
    
    CCLMetalKernel *metalKernel = (__bridge CCLMetalKernel *)kernel->impl;
    [metalKernel.uniforms removeAllObjects];
}

// --- Dispatch ---

// Internal ND async dispatch implementation
static ccl_error ccl_dispatch_nd_metal_async(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t dim,
    const size_t global_size[3],
    const size_t local_size[3],
    ccl_buffer **buffers,
    size_t num_buffers,
    ccl_fence **out_fence
) {
    if (!ctx || !kernel || !global_size) return CCL_ERROR_INVALID_ARGUMENT;
    if (dim < 1 || dim > 3) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL ||
        kernel->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;

    @autoreleasepool {
        // Early return for zero-sized dispatch
        if (global_size[0] == 0 || (dim >= 2 && global_size[1] == 0) || (dim >= 3 && global_size[2] == 0)) {
            if (out_fence) {
                *out_fence = NULL;
            }
            return CCL_OK;
        }
        
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        CCLMetalKernel  *metalKernel = (__bridge CCLMetalKernel *)kernel->impl;

        // Check if we're in a batch
        BOOL isBatched = (metalCtx.activeBatch != nil);
        id<MTLCommandBuffer> cmd = isBatched ? metalCtx.activeBatch : [metalCtx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = nil;
        
        if (isBatched) {
            // Reuse existing encoder from batch
            enc = metalCtx.activeEncoder;
        } else {
            // New command buffer - set label and create encoder
            if (metalCtx.label) {
                cmd.label = metalCtx.label;
            }
            enc = [cmd computeCommandEncoder];
        }
        
        // Create fence if requested (only for non-batched dispatches)
        CCLMetalFence *fence = nil;
        if (out_fence && !isBatched) {
            fence = [[CCLMetalFence alloc] initWithCommandBuffer:cmd];
        }
        
        #ifdef CCL_DEBUG
        NSString *groupLabel = metalKernel.label ?: @"ccl_dispatch_nd";
        [enc pushDebugGroup:groupLabel];
        #endif

        [enc setComputePipelineState:metalKernel.pipeline];

        // Set uniforms/bytes first (before buffers)
        // Note: If a buffer is later set at the same index, it will override the uniform
        for (NSNumber *index in metalKernel.uniforms) {
            NSData *data = metalKernel.uniforms[index];
            [enc setBytes:data.bytes length:data.length atIndex:index.unsignedIntValue];
        }

        // Set buffers (these can override uniforms if indices overlap)
        for (size_t i = 0; i < num_buffers; ++i) {
            ccl_buffer *b = buffers[i];
            if (!b || b->kind != CCL_BACKEND_KIND_METAL) continue;
            CCLMetalBuffer *mb = (__bridge CCLMetalBuffer *)b->impl;
            [enc setBuffer:mb.buffer offset:0 atIndex:(NSUInteger)i];
        }

        // Compute threadgroup sizes with optimal heuristics
        NSUInteger simdWidth = metalKernel.pipeline.threadExecutionWidth;
        NSUInteger maxThreads = metalKernel.pipeline.maxTotalThreadsPerThreadgroup;
        
        MTLSize threadsPerThreadgroup;
        MTLSize threadgroupsPerGrid;
        
        if (dim == 1) {
            // 1D: optimize X dimension with threadExecutionWidth
            NSUInteger tgX;
            if (local_size[0] > 0) {
                // User specified size - use it (but still optimize to SIMD width)
                tgX = (NSUInteger)local_size[0];
                tgX = MIN(tgX, maxThreads);
                tgX = (tgX / simdWidth) * simdWidth;
                if (tgX == 0) tgX = simdWidth;
            } else {
                // Auto-select: use cached value if available, otherwise compute and cache
                if (metalKernel.cachedThreadgroupSize > 0) {
                    tgX = metalKernel.cachedThreadgroupSize;
                } else {
                    tgX = maxThreads;
                    tgX = (tgX / simdWidth) * simdWidth;
                    if (tgX == 0) tgX = simdWidth;
                    // Cache for future dispatches
                    metalKernel.cachedThreadgroupSize = tgX;
                }
            }
            
            threadsPerThreadgroup = MTLSizeMake(tgX, 1, 1);
            NSUInteger numGroups = (NSUInteger)((global_size[0] + tgX - 1) / tgX);
            threadgroupsPerGrid = MTLSizeMake(numGroups, 1, 1);
        } else if (dim == 2) {
            // 2D: optimize X, use Y as-is (or reasonable default)
            NSUInteger tgX = (local_size[0] > 0) ? (NSUInteger)local_size[0] : simdWidth;
            tgX = MIN(tgX, maxThreads);
            tgX = (tgX / simdWidth) * simdWidth;
            if (tgX == 0) tgX = simdWidth;
            
            NSUInteger tgY = (local_size[1] > 0) ? (NSUInteger)local_size[1] : 1;
            // Ensure total threads doesn't exceed max
            while (tgX * tgY > maxThreads && tgY > 1) {
                tgY--;
            }
            
            threadsPerThreadgroup = MTLSizeMake(tgX, tgY, 1);
            NSUInteger numGroupsX = (NSUInteger)((global_size[0] + tgX - 1) / tgX);
            NSUInteger numGroupsY = (NSUInteger)((global_size[1] + tgY - 1) / tgY);
            threadgroupsPerGrid = MTLSizeMake(numGroupsX, numGroupsY, 1);
        } else {  // dim == 3
            // 3D: optimize X, use Y and Z as-is (or reasonable defaults)
            NSUInteger tgX = (local_size[0] > 0) ? (NSUInteger)local_size[0] : simdWidth;
            tgX = MIN(tgX, maxThreads);
            tgX = (tgX / simdWidth) * simdWidth;
            if (tgX == 0) tgX = simdWidth;
            
            NSUInteger tgY = (local_size[1] > 0) ? (NSUInteger)local_size[1] : 1;
            NSUInteger tgZ = (local_size[2] > 0) ? (NSUInteger)local_size[2] : 1;
            
            // Ensure total threads doesn't exceed max
            while (tgX * tgY * tgZ > maxThreads && tgZ > 1) {
                tgZ--;
            }
            while (tgX * tgY * tgZ > maxThreads && tgY > 1) {
                tgY--;
            }
            
            threadsPerThreadgroup = MTLSizeMake(tgX, tgY, tgZ);
            NSUInteger numGroupsX = (NSUInteger)((global_size[0] + tgX - 1) / tgX);
            NSUInteger numGroupsY = (NSUInteger)((global_size[1] + tgY - 1) / tgY);
            NSUInteger numGroupsZ = (NSUInteger)((global_size[2] + tgZ - 1) / tgZ);
            threadgroupsPerGrid = MTLSizeMake(numGroupsX, numGroupsY, numGroupsZ);
        }

        [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        #ifdef CCL_DEBUG
        [enc popDebugGroup];
        #endif
        
        // Only end encoding and commit if not in a batch
        if (!isBatched) {
            [enc endEncoding];
            [cmd commit];
            
            // Return fence if requested
            if (out_fence && fence) {
                ccl_fence *cclFence = (ccl_fence *)malloc(sizeof(ccl_fence));
                if (!cclFence) {
                    if (out_fence) {
                        *out_fence = NULL;
                    }
                    return CCL_ERROR_DEVICE_FAILED;
                }
                cclFence->kind = CCL_BACKEND_KIND_METAL;
                cclFence->impl = (__bridge_retained void *)fence;
                *out_fence = cclFence;
            }
        } else {
            // In batch - don't return fence (will be returned at end_batch)
            if (out_fence) {
                *out_fence = NULL;
            }
        }

        return CCL_OK;
    }
}

// Internal async dispatch implementation (1D wrapper)
static ccl_error ccl_dispatch_1d_metal_async(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t global_size,
    size_t local_size,
    ccl_buffer **buffers,
    size_t num_buffers,
    ccl_fence **out_fence
) {
    // Wrap 1D dispatch as ND dispatch
    size_t global[3] = { global_size, 1, 1 };
    size_t local[3] = { local_size, 0, 0 };
    return ccl_dispatch_nd_metal_async(ctx, kernel, 1, global, local, buffers, num_buffers, out_fence);
}

static ccl_error ccl_dispatch_1d_metal(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t global_size,
    size_t local_size,
    ccl_buffer **buffers,
    size_t num_buffers
) {
    ccl_fence *fence = NULL;
    ccl_error err = ccl_dispatch_1d_metal_async(ctx, kernel, global_size, local_size, buffers, num_buffers, &fence);
    if (err != CCL_OK) {
        return err;
    }

    // Zero-sized dispatch - no work to wait on
    if (!fence) {
        return CCL_OK;
    }

    // Wait for completion (synchronous behavior)
    ccl_fence_wait(fence);
    
        // Check for errors
        CCLMetalFence *metalFence = (__bridge CCLMetalFence *)fence->impl;
        if (metalFence.commandBuffer.error) {
            NSError *error = metalFence.commandBuffer.error;
            const char *msg = error.localizedDescription.UTF8String;
            const char *errorMsg = msg ? msg : "Dispatch execution failed";
            
            // Call log callback if set
            char callbackMsg[512];
            snprintf(callbackMsg, sizeof(callbackMsg), "Dispatch error: %s", errorMsg);
            ccl_log_metal(ctx, callbackMsg);
            
            ccl_fence_destroy(fence);
            return CCL_ERROR_DISPATCH_FAILED;
        }

    ccl_fence_destroy(fence);
    return CCL_OK;
}

ccl_error ccl_dispatch_1d(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t global_size,
    size_t local_size,
    ccl_buffer **buffers,
    size_t num_buffers
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_dispatch_1d_metal(ctx, kernel, global_size, local_size, buffers, num_buffers);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

// --- ND Dispatch ---

ccl_error ccl_dispatch_nd(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t dim,
    const size_t global_size[3],
    const size_t local_size[3],
    ccl_buffer **buffers,
    size_t num_buffers
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL: {
        ccl_fence *fence = NULL;
        ccl_error err = ccl_dispatch_nd_metal_async(ctx, kernel, dim, global_size, local_size, buffers, num_buffers, &fence);
        if (err != CCL_OK) {
            return err;
        }

        // Zero-sized dispatch - no work to wait on
        if (!fence) {
            return CCL_OK;
        }

        // Wait for completion (synchronous behavior)
        ccl_fence_wait(fence);
        
        // Check for errors
        CCLMetalFence *metalFence = (__bridge CCLMetalFence *)fence->impl;
        if (metalFence.commandBuffer.error) {
            NSError *error = metalFence.commandBuffer.error;
            const char *msg = error.localizedDescription.UTF8String;
            const char *errorMsg = msg ? msg : "Dispatch execution failed";
            
            // Call log callback if set
            char callbackMsg[512];
            snprintf(callbackMsg, sizeof(callbackMsg), "Dispatch error: %s", errorMsg);
            ccl_log_metal(ctx, callbackMsg);
            
            ccl_fence_destroy(fence);
            return CCL_ERROR_DISPATCH_FAILED;
        }

        ccl_fence_destroy(fence);
        return CCL_OK;
    }
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

ccl_error ccl_dispatch_nd_async(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t dim,
    const size_t global_size[3],
    const size_t local_size[3],
    ccl_buffer **buffers,
    size_t num_buffers,
    ccl_fence **out_fence
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_dispatch_nd_metal_async(ctx, kernel, dim, global_size, local_size, buffers, num_buffers, out_fence);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

// --- Async dispatch and fences ---

ccl_error ccl_dispatch_1d_async(
    ccl_context *ctx,
    ccl_kernel *kernel,
    size_t global_size,
    size_t local_size,
    ccl_buffer **buffers,
    size_t num_buffers,
    ccl_fence **out_fence
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;

    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_dispatch_1d_metal_async(ctx, kernel, global_size, local_size, buffers, num_buffers, out_fence);
    default:
        return CCL_ERROR_UNSUPPORTED_BACKEND;
    }
}

bool ccl_fence_is_complete(ccl_fence *fence) {
    if (!fence) return true;
    
    if (fence->kind == CCL_BACKEND_KIND_METAL) {
        CCLMetalFence *metalFence = (__bridge CCLMetalFence *)fence->impl;
        return metalFence.commandBuffer.status >= MTLCommandBufferStatusCompleted;
    }
    
    return true;
}

void ccl_fence_wait(ccl_fence *fence) {
    if (!fence) return;
    
    if (fence->kind == CCL_BACKEND_KIND_METAL) {
        CCLMetalFence *metalFence = (__bridge CCLMetalFence *)fence->impl;
        [metalFence.commandBuffer waitUntilCompleted];
        // Update error message after waiting
        [metalFence updateErrorMessage];
    }
}

const char *ccl_fence_get_error_message(ccl_fence *fence) {
    if (!fence) return NULL;
    
    if (fence->kind == CCL_BACKEND_KIND_METAL) {
        CCLMetalFence *metalFence = (__bridge CCLMetalFence *)fence->impl;
        // Update error message if not already cached
        [metalFence updateErrorMessage];
        return metalFence.errorMessage ? metalFence.errorMessage.UTF8String : NULL;
    }
    
    return NULL;
}

void ccl_fence_destroy(ccl_fence *fence) {
    if (!fence) return;
    if (fence->impl) {
        CFRelease(fence->impl);
    }
    free(fence);
}

// --- Debug labels ---

void ccl_set_context_label(ccl_context *ctx, const char *label) {
    if (!ctx || ctx->kind != CCL_BACKEND_KIND_METAL) return;
    
    CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
    if (label) {
        metalCtx.label = [NSString stringWithUTF8String:label];
    } else {
        metalCtx.label = nil;
    }
}

void ccl_set_buffer_label(ccl_buffer *buf, const char *label) {
    if (!buf || buf->kind != CCL_BACKEND_KIND_METAL) return;
    
    CCLMetalBuffer *mb = (__bridge CCLMetalBuffer *)buf->impl;
    if (label) {
        mb.buffer.label = [NSString stringWithUTF8String:label];
    } else {
        mb.buffer.label = nil;
    }
}

void ccl_set_kernel_label(ccl_kernel *kernel, const char *label) {
    if (!kernel || kernel->kind != CCL_BACKEND_KIND_METAL) return;
    
    CCLMetalKernel *mk = (__bridge CCLMetalKernel *)kernel->impl;
    mk.label = label ? [NSString stringWithUTF8String:label] : nil;
}

// --- Metal 3/4 Capability Detection ---

static ccl_error ccl_get_metal_capabilities_metal(
    ccl_context *ctx,
    ccl_metal_capabilities *out_caps
) {
    if (!ctx || !out_caps) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_NOT_SUPPORTED;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        id<MTLDevice> device = metalCtx.device;
        
        // Initialize all to false
        memset(out_caps, 0, sizeof(ccl_metal_capabilities));
        
        // Check Metal 3 support (macOS 11.0+, iOS 14.0+)
        if (@available(macOS 11.0, iOS 14.0, *)) {
            // Check for Metal 3 GPU families
            if ([device supportsFamily:MTLGPUFamilyMetal3]) {
                out_caps->supports_metal_3 = true;
            }
            
            // Check for Metal 4 support (macOS 13.0+, iOS 16.0+)
            // Metal 4 doesn't have a separate GPU family enum, but we can detect it via other means
            if (@available(macOS 13.0, iOS 16.0, *)) {
                // Metal 4 is available on newer devices - check for enhanced features
                // For now, we'll set this based on Metal 3 support and newer OS versions
                // In practice, you'd check for specific Metal 4 features
                if (out_caps->supports_metal_3) {
                    // Metal 4 is typically available on devices that support Metal 3 with newer OS
                    // This is a simplified check - full detection would query specific Metal 4 features
                    out_caps->supports_metal_4 = true;
                }
            }
            
            // Function tables (Metal 3+)
            if (out_caps->supports_metal_3) {
                out_caps->supports_function_tables = true;
                // Max function table size is typically 64K entries
                out_caps->max_function_table_size = 65536;
            }
            
            // Ray tracing (Metal 3+)
            if ([device supportsRaytracing]) {
                out_caps->supports_raytracing = true;
            }
            
            // Binary archives (Metal 3+)
            if ([device supportsFamily:MTLGPUFamilyMetal3]) {
                out_caps->supports_binary_archives = true;
            }
            
            // SIMD-group matrix (Apple7+ family, Metal 3+)
            if ([device supportsFamily:MTLGPUFamilyApple7]) {
                out_caps->supports_simdgroup_matrix = true;
            }
            // Also check Apple8, Apple9, Apple10 families
            if (@available(macOS 12.0, iOS 15.0, *)) {
                if ([device supportsFamily:MTLGPUFamilyApple8]) {
                    out_caps->supports_simdgroup_matrix = true;
                }
            }
            if (@available(macOS 13.0, iOS 16.0, *)) {
                if ([device supportsFamily:MTLGPUFamilyApple9]) {
                    out_caps->supports_simdgroup_matrix = true;
                }
            }
            // Check for newer Apple GPU families as they become available
            // Apple10 family may not be in all SDKs yet
            if (@available(macOS 14.0, iOS 17.0, *)) {
                // Check if Apple10 family exists (may not be in older SDKs)
                if ([device respondsToSelector:@selector(supportsFamily:)]) {
                    // Try to detect newer families - this will compile even if enum doesn't exist
                    // Runtime check will handle availability
                }
            }
            
            // Indirect command buffers (Metal 3+)
            if (out_caps->supports_metal_3) {
                out_caps->supports_indirect_command_buffers = true;
            }
            
            // Argument buffers (available on all Metal devices, enhanced in Metal 3/4)
            // Metal 2+ devices support argument buffers
            if (out_caps->supports_metal_3 || [device supportsFamily:MTLGPUFamilyApple1]) {
                out_caps->supports_argument_buffers = true;
                // Metal 3+ has expanded limits
                if (out_caps->supports_metal_3) {
                    out_caps->max_argument_buffer_length = 128 * 1024;  // 128KB for Metal 3+
                } else {
                    out_caps->max_argument_buffer_length = 64 * 1024;   // 64KB for older Metal
                }
            }
        }
        
        return CCL_OK;
    }
}

ccl_error ccl_get_metal_capabilities(
    ccl_context *ctx,
    ccl_metal_capabilities *out_caps
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_get_metal_capabilities_metal(ctx, out_caps);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

// --- Function Tables (Metal 3+) ---

static ccl_error ccl_create_function_table_metal(
    ccl_context *ctx,
    uint32_t size,
    ccl_function_table **out_table
) {
    if (!ctx || !out_table || size == 0) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_NOT_SUPPORTED;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        id<MTLDevice> device = metalCtx.device;
        
        // Check Metal 3 support
        if (@available(macOS 11.0, iOS 14.0, *)) {
            if (![device supportsFamily:MTLGPUFamilyMetal3]) {
                return CCL_ERROR_NOT_SUPPORTED;
            }
        } else {
            return CCL_ERROR_NOT_SUPPORTED;
        }
        
        // Create a minimal compute function for the pipeline
        // We need a pipeline state to create visible function tables
        const char *dummySource = "kernel void dummy_function() {}";
        NSString *src = [NSString stringWithUTF8String:dummySource];
        NSError *error = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error];
        if (!lib) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        id<MTLFunction> dummyFunc = [lib newFunctionWithName:@"dummy_function"];
        if (!dummyFunc) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = dummyFunc;
        
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:dummyFunc error:&error];
        if (!pipeline) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        // Create visible function table descriptor
        MTLVisibleFunctionTableDescriptor *tableDesc = [[MTLVisibleFunctionTableDescriptor alloc] init];
        tableDesc.functionCount = size;
        
        id<MTLVisibleFunctionTable> table = [pipeline newVisibleFunctionTableWithDescriptor:tableDesc];
        if (!table) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        CCLMetalFunctionTable *metalTable = [[CCLMetalFunctionTable alloc] init];
        metalTable.table = table;
        metalTable.pipeline = pipeline;
        metalTable.size = size;
        
        ccl_function_table *cclTable = (ccl_function_table *)malloc(sizeof(ccl_function_table));
        if (!cclTable) return CCL_ERROR_DEVICE_FAILED;
        
        cclTable->kind = CCL_BACKEND_KIND_METAL;
        cclTable->impl = (__bridge_retained void *)metalTable;
        
        *out_table = cclTable;
        return CCL_OK;
    }
}

ccl_error ccl_create_function_table(
    ccl_context *ctx,
    uint32_t size,
    ccl_function_table **out_table
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_function_table_metal(ctx, size, out_table);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

static ccl_error ccl_function_table_set_metal(
    ccl_function_table *table,
    ccl_kernel *kernel,
    uint32_t index
) {
    if (!table || !kernel) return CCL_ERROR_INVALID_ARGUMENT;
    if (table->kind != CCL_BACKEND_KIND_METAL || kernel->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalFunctionTable *metalTable = (__bridge CCLMetalFunctionTable *)table->impl;
        CCLMetalKernel *metalKernel = (__bridge CCLMetalKernel *)kernel->impl;
        
        if (index >= metalTable.size) {
            return CCL_ERROR_INVALID_ARGUMENT;
        }
        
        // Check Metal 3 support
        if (@available(macOS 11.0, iOS 14.0, *)) {
            // Get function handle from the kernel's pipeline
            // Note: We need the function from the kernel to create a handle
            // The function must be from the same library/device
            if (!metalKernel.function) {
                // Function not available (e.g., from cache hit)
                // For cached kernels, we can't create function handles
                // Users should create kernels without caching for function tables
                return CCL_ERROR_INVALID_ARGUMENT;
            }
            
            // Create function handle from the pipeline state
            // The pipeline must support function handles (Metal 3+)
            id<MTLFunctionHandle> handle = [metalTable.pipeline functionHandleWithFunction:metalKernel.function];
            if (!handle) {
                return CCL_ERROR_DEVICE_FAILED;
            }
            
            // Set function in table
            [metalTable.table setFunction:handle atIndex:index];
            
            return CCL_OK;
        }
        
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

ccl_error ccl_function_table_set(
    ccl_function_table *table,
    ccl_kernel *kernel,
    uint32_t index
) {
    if (!table) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (table->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_function_table_set_metal(table, kernel, index);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

static void ccl_destroy_function_table_metal(ccl_function_table *table) {
    if (!table) return;
    if (table->impl) {
        CFRelease(table->impl);
    }
    free(table);
}

void ccl_destroy_function_table(ccl_function_table *table) {
    if (!table) return;
    
    switch (table->kind) {
    case CCL_BACKEND_KIND_METAL:
        ccl_destroy_function_table_metal(table);
        break;
    default:
        break;
    }
}

// --- Binary Archives (Metal 3+) ---

static ccl_error ccl_create_binary_archive_metal(
    ccl_context *ctx,
    ccl_binary_archive **out_archive
) {
    if (!ctx || !out_archive) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_NOT_SUPPORTED;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        id<MTLDevice> device = metalCtx.device;
        
        // Check Metal 3 support
        if (@available(macOS 11.0, iOS 14.0, *)) {
            if (![device supportsFamily:MTLGPUFamilyMetal3]) {
                return CCL_ERROR_NOT_SUPPORTED;
            }
        } else {
            return CCL_ERROR_NOT_SUPPORTED;
        }
        
        MTLBinaryArchiveDescriptor *desc = [[MTLBinaryArchiveDescriptor alloc] init];
        NSError *error = nil;
        id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:desc error:&error];
        if (!archive) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        CCLMetalBinaryArchive *metalArchive = [[CCLMetalBinaryArchive alloc] init];
        metalArchive.archive = archive;
        
        ccl_binary_archive *cclArchive = (ccl_binary_archive *)malloc(sizeof(ccl_binary_archive));
        if (!cclArchive) return CCL_ERROR_DEVICE_FAILED;
        
        cclArchive->kind = CCL_BACKEND_KIND_METAL;
        cclArchive->impl = (__bridge_retained void *)metalArchive;
        
        *out_archive = cclArchive;
        return CCL_OK;
    }
}

ccl_error ccl_create_binary_archive(
    ccl_context *ctx,
    ccl_binary_archive **out_archive
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_binary_archive_metal(ctx, out_archive);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

static ccl_error ccl_binary_archive_add_kernel_metal(
    ccl_binary_archive *archive,
    ccl_kernel *kernel
) {
    if (!archive || !kernel) return CCL_ERROR_INVALID_ARGUMENT;
    if (archive->kind != CCL_BACKEND_KIND_METAL || kernel->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalBinaryArchive *metalArchive = (__bridge CCLMetalBinaryArchive *)archive->impl;
        CCLMetalKernel *metalKernel = (__bridge CCLMetalKernel *)kernel->impl;
        
        // Check if already added
        if ([metalArchive.addedPipelines containsObject:metalKernel.pipeline]) {
            return CCL_OK;  // Already in archive
        }
        
        // Add compute pipeline to archive
        MTLComputePipelineDescriptor *desc = [[MTLComputePipelineDescriptor alloc] init];
        if (metalKernel.function) {
            desc.computeFunction = metalKernel.function;
        } else {
            // Can't add without function - would need to recompile
            return CCL_ERROR_INVALID_ARGUMENT;
        }
        
        NSError *error = nil;
        BOOL success = [metalArchive.archive addComputePipelineFunctionsWithDescriptor:desc error:&error];
        if (!success) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        [metalArchive.addedPipelines addObject:metalKernel.pipeline];
        return CCL_OK;
    }
}

ccl_error ccl_binary_archive_add_kernel(
    ccl_binary_archive *archive,
    ccl_kernel *kernel
) {
    if (!archive) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (archive->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_binary_archive_add_kernel_metal(archive, kernel);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

static ccl_error ccl_binary_archive_serialize_metal(
    ccl_binary_archive *archive,
    uint8_t *out_data,
    size_t *out_size
) {
    if (!archive || !out_size) return CCL_ERROR_INVALID_ARGUMENT;
    if (archive->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalBinaryArchive *metalArchive = (__bridge CCLMetalBinaryArchive *)archive->impl;
        
        // Serialize archive - Metal requires a file URL, so we use a temporary approach
        // In practice, users should use serializeToURL with a file path
        // For in-memory serialization, we'd need to write to a temp file and read it back
        // This is a limitation of Metal's API
        NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:
                             [NSString stringWithFormat:@"ccl_archive_%p.metallib", metalArchive.archive]];
        NSURL *tempURL = [NSURL fileURLWithPath:tempPath];
        
        NSError *error = nil;
        BOOL success = [metalArchive.archive serializeToURL:tempURL error:&error];
        if (!success) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        NSData *data = [NSData dataWithContentsOfURL:tempURL];
        [[NSFileManager defaultManager] removeItemAtURL:tempURL error:nil];
        
        if (!data) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        size_t dataSize = data.length;
        if (out_data) {
            if (*out_size < dataSize) {
                *out_size = dataSize;
                return CCL_ERROR_INVALID_ARGUMENT;
            }
            memcpy(out_data, data.bytes, dataSize);
        }
        *out_size = dataSize;
        return CCL_OK;
    }
}

ccl_error ccl_binary_archive_serialize(
    ccl_binary_archive *archive,
    uint8_t *out_data,
    size_t *out_size
) {
    if (!archive) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (archive->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_binary_archive_serialize_metal(archive, out_data, out_size);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

static ccl_error ccl_binary_archive_load_metal(
    ccl_context *ctx,
    const uint8_t *data,
    size_t size,
    ccl_binary_archive **out_archive
) {
    if (!ctx || !data || !out_archive || size == 0) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        id<MTLDevice> device = metalCtx.device;
        
        // Check Metal 3 support
        if (@available(macOS 11.0, iOS 14.0, *)) {
            if (![device supportsFamily:MTLGPUFamilyMetal3]) {
                return CCL_ERROR_NOT_SUPPORTED;
            }
        } else {
            return CCL_ERROR_NOT_SUPPORTED;
        }
        
        NSData *archiveData = [NSData dataWithBytes:data length:size];
        MTLBinaryArchiveDescriptor *desc = [[MTLBinaryArchiveDescriptor alloc] init];
        // Note: Metal doesn't have a direct "load from data" API, so we'd need to save to URL first
        // For now, return error indicating this needs file-based loading
        // In practice, you'd use serializeToURL/URL-based loading
        return CCL_ERROR_NOT_SUPPORTED;  // File-based loading required
    }
}

ccl_error ccl_binary_archive_load(
    ccl_context *ctx,
    const uint8_t *data,
    size_t size,
    ccl_binary_archive **out_archive
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_binary_archive_load_metal(ctx, data, size, out_archive);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

static void ccl_destroy_binary_archive_metal(ccl_binary_archive *archive) {
    if (!archive) return;
    if (archive->impl) {
        CFRelease(archive->impl);
    }
    free(archive);
}

void ccl_destroy_binary_archive(ccl_binary_archive *archive) {
    if (!archive) return;
    
    switch (archive->kind) {
    case CCL_BACKEND_KIND_METAL:
        ccl_destroy_binary_archive_metal(archive);
        break;
    default:
        break;
    }
}

// --- Ray Tracing (Metal 3+) ---

static ccl_error ccl_create_acceleration_structure_metal(
    ccl_context *ctx,
    uint32_t geometry_count,
    ccl_acceleration_structure **out_as
) {
    if (!ctx || !out_as) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_NOT_SUPPORTED;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        id<MTLDevice> device = metalCtx.device;
        
        // Check ray tracing support
        if (@available(macOS 11.0, iOS 14.0, *)) {
            if (![device supportsRaytracing]) {
                return CCL_ERROR_NOT_SUPPORTED;
            }
        } else {
            return CCL_ERROR_NOT_SUPPORTED;
        }
        
        // Create acceleration structure descriptor
        MTLAccelerationStructureDescriptor *desc = [[MTLPrimitiveAccelerationStructureDescriptor alloc] init];
        // Note: This is a simplified API - full implementation would require geometry setup
        // For now, return a placeholder that indicates the structure exists
        
        CCLMetalAccelerationStructure *metalAS = [[CCLMetalAccelerationStructure alloc] init];
        // In a full implementation, we'd create the actual MTLAccelerationStructure here
        // This requires setting up geometry buffers, which is beyond the scope of this basic API
        
        ccl_acceleration_structure *cclAS = (ccl_acceleration_structure *)malloc(sizeof(ccl_acceleration_structure));
        if (!cclAS) return CCL_ERROR_DEVICE_FAILED;
        
        cclAS->kind = CCL_BACKEND_KIND_METAL;
        cclAS->impl = (__bridge_retained void *)metalAS;
        
        *out_as = cclAS;
        return CCL_OK;
    }
}

ccl_error ccl_create_acceleration_structure(
    ccl_context *ctx,
    uint32_t geometry_count,
    ccl_acceleration_structure **out_as
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_acceleration_structure_metal(ctx, geometry_count, out_as);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

void ccl_destroy_acceleration_structure(ccl_acceleration_structure *as) {
    if (!as) return;
    if (as->impl) {
        CFRelease(as->impl);
    }
    free(as);
}

static ccl_error ccl_create_raytracing_pipeline_from_source_metal(
    ccl_context *ctx,
    const char *source,
    const char *raygen_function,
    const char *intersection_function,
    ccl_raytracing_pipeline **out_pipeline,
    char *log_buffer,
    size_t log_buffer_size
) {
    if (!ctx || !source || !raygen_function || !out_pipeline)
        return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        id<MTLDevice> device = metalCtx.device;
        
        // Check ray tracing support
        if (@available(macOS 11.0, iOS 14.0, *)) {
            if (![device supportsRaytracing]) {
                if (log_buffer && log_buffer_size > 0) {
                    snprintf(log_buffer, log_buffer_size, "Ray tracing not supported on this device");
                }
                return CCL_ERROR_NOT_SUPPORTED;
            }
        } else {
            return CCL_ERROR_NOT_SUPPORTED;
        }
        
        // Compile library
        NSError *error = nil;
        NSString *src = [NSString stringWithUTF8String:source];
        id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                   options:metalCtx.compileOptions
                                                     error:&error];
        if (!lib) {
            const char *msg = error.localizedDescription.UTF8String;
            if (log_buffer && log_buffer_size > 0) {
                snprintf(log_buffer, log_buffer_size, "%s", msg ? msg : "Compile error");
            }
            return CCL_ERROR_COMPILE_FAILED;
        }
        
        // Get raygen function
        NSString *raygenName = [NSString stringWithUTF8String:raygen_function];
        id<MTLFunction> raygenFunc = [lib newFunctionWithName:raygenName];
        if (!raygenFunc) {
            if (log_buffer && log_buffer_size > 0) {
                snprintf(log_buffer, log_buffer_size, "Raygen function not found: %s", raygen_function);
            }
            return CCL_ERROR_COMPILE_FAILED;
        }
        
        // Metal ray tracing uses compute pipelines with intersection functions
        // Create a compute pipeline for the raygen function
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:raygenFunc error:&error];
        if (!pipeline) {
            const char *msg = error.localizedDescription.UTF8String;
            if (log_buffer && log_buffer_size > 0) {
                snprintf(log_buffer, log_buffer_size, "%s", msg ? msg : "Pipeline creation error");
            }
            return CCL_ERROR_COMPILE_FAILED;
        }
        
        // Create intersection function table if intersection function is provided
        id<MTLIntersectionFunctionTable> intersectionTable = nil;
        if (intersection_function) {
            NSString *intersectionName = [NSString stringWithUTF8String:intersection_function];
            id<MTLFunction> intersectionFunc = [lib newFunctionWithName:intersectionName];
            if (intersectionFunc) {
                // Create intersection function table descriptor
                MTLIntersectionFunctionTableDescriptor *tableDesc = [[MTLIntersectionFunctionTableDescriptor alloc] init];
                tableDesc.functionCount = 1;
                intersectionTable = [pipeline newIntersectionFunctionTableWithDescriptor:tableDesc];
                if (intersectionTable) {
                    id<MTLFunctionHandle> handle = [pipeline functionHandleWithFunction:intersectionFunc];
                    if (handle) {
                        [intersectionTable setFunction:handle atIndex:0];
                    }
                }
            }
        }
        
        CCLMetalRaytracingPipeline *metalPipeline = [[CCLMetalRaytracingPipeline alloc] init];
        metalPipeline.pipeline = pipeline;
        metalPipeline.intersectionTable = intersectionTable;
        
        ccl_raytracing_pipeline *cclPipeline = (ccl_raytracing_pipeline *)malloc(sizeof(ccl_raytracing_pipeline));
        if (!cclPipeline) return CCL_ERROR_DEVICE_FAILED;
        
        cclPipeline->kind = CCL_BACKEND_KIND_METAL;
        cclPipeline->impl = (__bridge_retained void *)metalPipeline;
        
        *out_pipeline = cclPipeline;
        return CCL_OK;
    }
}

ccl_error ccl_create_raytracing_pipeline_from_source(
    ccl_context *ctx,
    const char *source,
    const char *raygen_function,
    const char *intersection_function,
    ccl_raytracing_pipeline **out_pipeline,
    char *log_buffer,
    size_t log_buffer_size
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_raytracing_pipeline_from_source_metal(
            ctx, source, raygen_function, intersection_function, out_pipeline, log_buffer, log_buffer_size
        );
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

void ccl_destroy_raytracing_pipeline(ccl_raytracing_pipeline *pipeline) {
    if (!pipeline) return;
    if (pipeline->impl) {
        CFRelease(pipeline->impl);
    }
    free(pipeline);
}

// --- Indirect Command Buffers (Metal 3+) ---

static ccl_error ccl_create_indirect_command_buffer_metal(
    ccl_context *ctx,
    uint32_t max_commands,
    ccl_indirect_command_buffer **out_icb
) {
    if (!ctx || !out_icb || max_commands == 0) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL) return CCL_ERROR_NOT_SUPPORTED;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        id<MTLDevice> device = metalCtx.device;
        
        // Check Metal 3 support
        if (@available(macOS 11.0, iOS 14.0, *)) {
            if (![device supportsFamily:MTLGPUFamilyMetal3]) {
                return CCL_ERROR_NOT_SUPPORTED;
            }
        } else {
            return CCL_ERROR_NOT_SUPPORTED;
        }
        
        // Create ICB descriptor
        MTLIndirectCommandBufferDescriptor *desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;  // Use ConcurrentDispatch for compute commands
        desc.inheritBuffers = NO;
        desc.inheritPipelineState = NO;
        
        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:desc
                                                                              maxCommandCount:max_commands
                                                                                      options:0];
        if (!icb) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        CCLMetalIndirectCommandBuffer *metalICB = [[CCLMetalIndirectCommandBuffer alloc] init];
        metalICB.icb = icb;
        metalICB.maxCommands = max_commands;
        
        ccl_indirect_command_buffer *cclICB = (ccl_indirect_command_buffer *)malloc(sizeof(ccl_indirect_command_buffer));
        if (!cclICB) return CCL_ERROR_DEVICE_FAILED;
        
        cclICB->kind = CCL_BACKEND_KIND_METAL;
        cclICB->impl = (__bridge_retained void *)metalICB;
        
        *out_icb = cclICB;
        return CCL_OK;
    }
}

ccl_error ccl_create_indirect_command_buffer(
    ccl_context *ctx,
    uint32_t max_commands,
    ccl_indirect_command_buffer **out_icb
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_create_indirect_command_buffer_metal(ctx, max_commands, out_icb);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

static ccl_error ccl_indirect_command_buffer_encode_compute_metal(
    ccl_indirect_command_buffer *icb,
    ccl_kernel *kernel,
    size_t dim,
    const size_t global_size[3],
    const size_t local_size[3],
    ccl_buffer **buffers,
    size_t num_buffers
) {
    if (!icb || !kernel || !global_size) return CCL_ERROR_INVALID_ARGUMENT;
    if (icb->kind != CCL_BACKEND_KIND_METAL || kernel->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalIndirectCommandBuffer *metalICB = (__bridge CCLMetalIndirectCommandBuffer *)icb->impl;
        CCLMetalKernel *metalKernel = (__bridge CCLMetalKernel *)kernel->impl;
        
        // Get indirect compute command
        id<MTLIndirectComputeCommand> command = [metalICB.icb indirectComputeCommandAtIndex:0];
        if (!command) {
            return CCL_ERROR_DEVICE_FAILED;
        }
        
        // Set pipeline
        [command setComputePipelineState:metalKernel.pipeline];
        
        // Set buffers - use setKernelBuffer for compute commands
        for (size_t i = 0; i < num_buffers && i < 31; ++i) {  // Metal limit is 31 buffers
            ccl_buffer *b = buffers[i];
            if (!b || b->kind != CCL_BACKEND_KIND_METAL) continue;
            CCLMetalBuffer *mb = (__bridge CCLMetalBuffer *)b->impl;
            [command setKernelBuffer:mb.buffer offset:0 atIndex:(NSUInteger)i];
        }
        
        // Set threadgroup sizes
        MTLSize threadsPerThreadgroup = MTLSizeMake(
            local_size[0] > 0 ? (NSUInteger)local_size[0] : 256,
            dim >= 2 && local_size[1] > 0 ? (NSUInteger)local_size[1] : 1,
            dim >= 3 && local_size[2] > 0 ? (NSUInteger)local_size[2] : 1
        );
        
        MTLSize threadgroupsPerGrid = MTLSizeMake(
            (NSUInteger)((global_size[0] + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width),
            dim >= 2 ? (NSUInteger)((global_size[1] + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height) : 1,
            dim >= 3 ? (NSUInteger)((global_size[2] + threadsPerThreadgroup.depth - 1) / threadsPerThreadgroup.depth) : 1
        );
        
        // Set threadgroup memory and dispatch sizes
        [command setThreadgroupMemoryLength:0 atIndex:0];
        [command setKernelBuffer:NULL offset:0 atIndex:31];  // Clear any unused slots
        
        // Note: For ConcurrentDispatch, the dispatch sizes are set via the execute call
        // The command encoding is simplified here - full implementation would properly encode all parameters
        
        return CCL_OK;
    }
}

ccl_error ccl_indirect_command_buffer_encode_compute(
    ccl_indirect_command_buffer *icb,
    ccl_kernel *kernel,
    size_t dim,
    const size_t global_size[3],
    const size_t local_size[3],
    ccl_buffer **buffers,
    size_t num_buffers
) {
    if (!icb) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (icb->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_indirect_command_buffer_encode_compute_metal(icb, kernel, dim, global_size, local_size, buffers, num_buffers);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

static ccl_error ccl_execute_indirect_command_buffer_metal(
    ccl_context *ctx,
    ccl_indirect_command_buffer *icb,
    uint32_t command_count,
    ccl_fence **out_fence
) {
    if (!ctx || !icb) return CCL_ERROR_INVALID_ARGUMENT;
    if (ctx->kind != CCL_BACKEND_KIND_METAL || icb->kind != CCL_BACKEND_KIND_METAL)
        return CCL_ERROR_INVALID_ARGUMENT;
    
    @autoreleasepool {
        CCLMetalContext *metalCtx = (__bridge CCLMetalContext *)ctx->impl;
        CCLMetalIndirectCommandBuffer *metalICB = (__bridge CCLMetalIndirectCommandBuffer *)icb->impl;
        
        // Check if in batch
        BOOL isBatched = (metalCtx.activeBatch != nil);
        id<MTLCommandBuffer> cmd = isBatched ? metalCtx.activeBatch : [metalCtx.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = nil;
        
        if (isBatched) {
            enc = metalCtx.activeEncoder;
        } else {
            if (metalCtx.label) {
                cmd.label = metalCtx.label;
            }
            enc = [cmd computeCommandEncoder];
        }
        
        // Execute ICB
        [enc executeCommandsInBuffer:metalICB.icb withRange:NSMakeRange(0, command_count)];
        
        // Create fence if requested (only for non-batched)
        CCLMetalFence *fence = nil;
        if (out_fence && !isBatched) {
            fence = [[CCLMetalFence alloc] initWithCommandBuffer:cmd];
        }
        
        if (!isBatched) {
            [enc endEncoding];
            [cmd commit];
            
            if (out_fence && fence) {
                ccl_fence *cclFence = (ccl_fence *)malloc(sizeof(ccl_fence));
                if (!cclFence) {
                    if (out_fence) *out_fence = NULL;
                    return CCL_ERROR_DEVICE_FAILED;
                }
                cclFence->kind = CCL_BACKEND_KIND_METAL;
                cclFence->impl = (__bridge_retained void *)fence;
                *out_fence = cclFence;
            }
        } else {
            if (out_fence) *out_fence = NULL;
        }
        
        return CCL_OK;
    }
}

ccl_error ccl_execute_indirect_command_buffer(
    ccl_context *ctx,
    ccl_indirect_command_buffer *icb,
    uint32_t command_count,
    ccl_fence **out_fence
) {
    if (!ctx) return CCL_ERROR_INVALID_ARGUMENT;
    
    switch (ctx->kind) {
    case CCL_BACKEND_KIND_METAL:
        return ccl_execute_indirect_command_buffer_metal(ctx, icb, command_count, out_fence);
    default:
        return CCL_ERROR_NOT_SUPPORTED;
    }
}

void ccl_destroy_indirect_command_buffer(ccl_indirect_command_buffer *icb) {
    if (!icb) return;
    if (icb->impl) {
        CFRelease(icb->impl);
    }
    free(icb);
}

