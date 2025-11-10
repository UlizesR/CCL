/**
 * MTL Resource Management
 * Buffers, samplers, heaps, and resource utilities
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "mtl_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// BUFFERS
// ============================================================================

MTLComputeBuffer* mtl_compute_buffer_create(MTLComputeDevice* device, size_t size, MTLComputeStorageMode mode) {
    if (!device || size == 0) {
        return NULL;
    }
    
    @autoreleasepool {
        MTLResourceOptions options = mtl_convert_storage_mode(mode);
        
        id<MTLBuffer> buffer = [device->device newBufferWithLength:size options:options];
        if (!buffer) {
            fprintf(stderr, "Failed to create buffer of size %zu\n", size);
            return NULL;
        }
        
        MTLComputeBuffer* mtlBuffer = (MTLComputeBuffer*)malloc(sizeof(MTLComputeBuffer));
        if (!mtlBuffer) {
            return NULL;
        }
        
        mtlBuffer->buffer = buffer;
        return mtlBuffer;
    }
}

MTLComputeBuffer* mtl_compute_buffer_create_with_data(
    MTLComputeDevice* device,
    const void* data,
    size_t size,
    MTLComputeStorageMode mode
) {
    if (!device || !data || size == 0) {
        return NULL;
    }
    
    MTLComputeBuffer* buffer = mtl_compute_buffer_create(device, size, mode);
    if (!buffer) {
        return NULL;
    }
    
    void* contents = mtl_compute_buffer_contents(buffer);
    if (contents) {
        memcpy(contents, data, size);
        if (mode == MTL_STORAGE_MANAGED) {
            mtl_compute_buffer_did_modify(buffer);
        }
    }
    
    return buffer;
}

void mtl_compute_buffer_destroy(MTLComputeBuffer* buffer) {
    if (buffer) {
        free(buffer);
    }
}

void* mtl_compute_buffer_contents(MTLComputeBuffer* buffer) {
    if (!buffer || !buffer->buffer) {
        return NULL;
    }
    @autoreleasepool {
        return [buffer->buffer contents];
    }
}

size_t mtl_compute_buffer_size(MTLComputeBuffer* buffer) {
    if (!buffer || !buffer->buffer) {
        return 0;
    }
    @autoreleasepool {
        return [buffer->buffer length];
    }
}

void mtl_compute_buffer_did_modify(MTLComputeBuffer* buffer) {
    if (!buffer || !buffer->buffer) {
        return;
    }
    @autoreleasepool {
#if TARGET_OS_OSX
        NSRange range = NSMakeRange(0, [buffer->buffer length]);
        [buffer->buffer didModifyRange:range];
#endif
    }
}

void mtl_compute_buffer_synchronize(MTLComputeBuffer* buffer, MTLComputeDevice* device) {
    if (!buffer || !buffer->buffer || !device) {
        return;
    }
    @autoreleasepool {
#if TARGET_OS_OSX
        id<MTLCommandBuffer> commandBuffer = [device->commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder synchronizeResource:buffer->buffer];
        [blitEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
#endif
    }
}

MTLComputeError mtl_compute_buffer_upload(
    MTLComputeDevice* device,
    MTLComputeBuffer* buffer,
    const void* src,
    size_t size
) {
    if (!device || !buffer || !src || size == 0) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    void* contents = mtl_compute_buffer_contents(buffer);
    if (!contents) {
        // Private buffer - would need staging
        return MTL_ERROR_UNSUPPORTED;
    }
    
    memcpy(contents, src, size);
    mtl_compute_buffer_did_modify(buffer);
    
    return MTL_SUCCESS;
}

MTLComputeError mtl_compute_buffer_download(
    MTLComputeDevice* device,
    MTLComputeBuffer* buffer,
    void* dst,
    size_t size
) {
    if (!device || !buffer || !dst || size == 0) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    mtl_compute_buffer_synchronize(buffer, device);
    
    void* contents = mtl_compute_buffer_contents(buffer);
    if (!contents) {
        return MTL_ERROR_UNSUPPORTED;
    }
    
    memcpy(dst, contents, size);
    
    return MTL_SUCCESS;
}

void mtl_compute_buffer_set_label(MTLComputeBuffer* buffer, const char* label) {
    if (!buffer || !label) {
        return;
    }
    
    @autoreleasepool {
        NSString* labelString = [NSString stringWithUTF8String:label];
        [buffer->buffer setLabel:labelString];
    }
}

// ============================================================================
// SAMPLERS
// ============================================================================

MTLComputeSampler* mtl_compute_sampler_create(
    MTLComputeDevice* device,
    const MTLComputeSamplerDesc* desc
) {
    if (!device || !desc) {
        return NULL;
    }
    
    @autoreleasepool {
        MTLSamplerDescriptor* samplerDesc = [[MTLSamplerDescriptor alloc] init];
        
        // Map filter modes
        samplerDesc.minFilter = (desc->min_filter == MTL_SAMPLER_FILTER_LINEAR) 
            ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        samplerDesc.magFilter = (desc->mag_filter == MTL_SAMPLER_FILTER_LINEAR)
            ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
        samplerDesc.mipFilter = (desc->mip_filter == MTL_SAMPLER_FILTER_LINEAR)
            ? MTLSamplerMipFilterLinear : MTLSamplerMipFilterNearest;
        
        // Map address modes
        MTLSamplerAddressMode addressU, addressV, addressW;
        switch (desc->address_mode_u) {
            case MTL_SAMPLER_ADDRESS_REPEAT:
                addressU = MTLSamplerAddressModeRepeat; break;
            case MTL_SAMPLER_ADDRESS_MIRRORED_REPEAT:
                addressU = MTLSamplerAddressModeMirrorRepeat; break;
            case MTL_SAMPLER_ADDRESS_CLAMP_TO_ZERO:
                addressU = MTLSamplerAddressModeClampToZero; break;
            default:
                addressU = MTLSamplerAddressModeClampToEdge; break;
        }
        switch (desc->address_mode_v) {
            case MTL_SAMPLER_ADDRESS_REPEAT:
                addressV = MTLSamplerAddressModeRepeat; break;
            case MTL_SAMPLER_ADDRESS_MIRRORED_REPEAT:
                addressV = MTLSamplerAddressModeMirrorRepeat; break;
            case MTL_SAMPLER_ADDRESS_CLAMP_TO_ZERO:
                addressV = MTLSamplerAddressModeClampToZero; break;
            default:
                addressV = MTLSamplerAddressModeClampToEdge; break;
        }
        switch (desc->address_mode_w) {
            case MTL_SAMPLER_ADDRESS_REPEAT:
                addressW = MTLSamplerAddressModeRepeat; break;
            case MTL_SAMPLER_ADDRESS_MIRRORED_REPEAT:
                addressW = MTLSamplerAddressModeMirrorRepeat; break;
            case MTL_SAMPLER_ADDRESS_CLAMP_TO_ZERO:
                addressW = MTLSamplerAddressModeClampToZero; break;
            default:
                addressW = MTLSamplerAddressModeClampToEdge; break;
        }
        
        samplerDesc.sAddressMode = addressU;
        samplerDesc.tAddressMode = addressV;
        samplerDesc.rAddressMode = addressW;
        samplerDesc.normalizedCoordinates = desc->normalized_coordinates;
        
        id<MTLSamplerState> samplerState = [device->device newSamplerStateWithDescriptor:samplerDesc];
        if (!samplerState) {
            return NULL;
        }
        
        MTLComputeSampler* sampler = (MTLComputeSampler*)malloc(sizeof(MTLComputeSampler));
        if (!sampler) {
            return NULL;
        }
        
        sampler->samplerState = samplerState;
        return sampler;
    }
}

void mtl_compute_sampler_destroy(MTLComputeSampler* sampler) {
    if (sampler) {
        free(sampler);
    }
}

// ============================================================================
// HEAPS
// ============================================================================

MTLComputeHeap* mtl_compute_heap_create(
    MTLComputeDevice* device,
    size_t size,
    MTLComputeStorageMode mode
) {
    if (!device || size == 0) {
        return NULL;
    }
    
    if (!device->caps.supports_heaps) {
        return NULL;
    }
    
    @autoreleasepool {
        MTLHeapDescriptor* descriptor = [[MTLHeapDescriptor alloc] init];
        descriptor.size = size;
        descriptor.storageMode = (MTLStorageMode)mtl_convert_storage_mode(mode);
        
        id<MTLHeap> heap = [device->device newHeapWithDescriptor:descriptor];
        if (!heap) {
            return NULL;
        }
        
        MTLComputeHeap* mtlHeap = (MTLComputeHeap*)malloc(sizeof(MTLComputeHeap));
        if (!mtlHeap) {
            return NULL;
        }
        
        mtlHeap->heap = heap;
        return mtlHeap;
    }
}

MTLComputeBuffer* mtl_compute_buffer_alloc_from_heap(MTLComputeHeap* heap, size_t size) {
    if (!heap || !heap->heap || size == 0) {
        return NULL;
    }
    
    @autoreleasepool {
        id<MTLBuffer> buffer = [heap->heap newBufferWithLength:size options:0];
        if (!buffer) {
            return NULL;
        }
        
        MTLComputeBuffer* mtlBuffer = (MTLComputeBuffer*)malloc(sizeof(MTLComputeBuffer));
        if (!mtlBuffer) {
            return NULL;
        }
        
        mtlBuffer->buffer = buffer;
        return mtlBuffer;
    }
}

void mtl_compute_heap_get_usage(MTLComputeHeap* heap, size_t* used, size_t* capacity) {
    if (!heap || !heap->heap) {
        if (used) *used = 0;
        if (capacity) *capacity = 0;
        return;
    }
    
    @autoreleasepool {
        if (used) *used = [heap->heap usedSize];
        if (capacity) *capacity = [heap->heap size];
    }
}

void mtl_compute_heap_destroy(MTLComputeHeap* heap) {
    if (heap) {
        free(heap);
    }
}

// ============================================================================
// TENSOR HELPERS
// ============================================================================

MTLComputeBuffer* mtl_compute_buffer_create_for_tensor(
    MTLComputeDevice* device,
    const MTLComputeTensorDesc* desc,
    size_t data_size,
    MTLComputeStorageMode mode
) {
    if (!device || !desc || data_size == 0) {
        return NULL;
    }
    
    // Allocate buffer: descriptor + data
    size_t total_size = sizeof(MTLComputeTensorDesc) + data_size;
    MTLComputeBuffer* buffer = mtl_compute_buffer_create(device, total_size, mode);
    if (!buffer) {
        return NULL;
    }
    
    // Write descriptor at the start
    void* contents = mtl_compute_buffer_contents(buffer);
    if (contents) {
        memcpy(contents, desc, sizeof(MTLComputeTensorDesc));
        mtl_compute_buffer_did_modify(buffer);
    }
    
    return buffer;
}

