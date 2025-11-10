/**
 * MTL Texture Support Implementation
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include "mtl_internal.h"
#include <stdio.h>
#include <stdlib.h>

// Helper: Convert MTLComputePixelFormat to Metal pixel format
static MTLPixelFormat convert_pixel_format(MTLComputePixelFormat format) {
    switch (format) {
        case MTL_PIXEL_FORMAT_RGBA8:
            return MTLPixelFormatRGBA8Unorm;
        case MTL_PIXEL_FORMAT_BGRA8:
            return MTLPixelFormatBGRA8Unorm;
        case MTL_PIXEL_FORMAT_RGBA32F:
            return MTLPixelFormatRGBA32Float;
        default:
            return MTLPixelFormatRGBA8Unorm;
    }
}

MTLComputeTexture* mtl_compute_texture_create(MTLComputeDevice* device, size_t width, size_t height, MTLComputePixelFormat format, const void* data) {
    if (!device || width == 0 || height == 0) {
        return NULL;
    }
    
    @autoreleasepool {
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:convert_pixel_format(format)
            width:width
            height:height
            mipmapped:NO];
        
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        id<MTLTexture> texture = [device->device newTextureWithDescriptor:descriptor];
        if (!texture) {
            fprintf(stderr, "Failed to create texture\n");
            return NULL;
        }
        
        // Upload data if provided
        if (data) {
            size_t bytesPerPixel = 4; // RGBA
            size_t bytesPerRow = width * bytesPerPixel;
            MTLRegion region = MTLRegionMake2D(0, 0, width, height);
            [texture replaceRegion:region
                       mipmapLevel:0
                         withBytes:data
                       bytesPerRow:bytesPerRow];
        }
        
        MTLComputeTexture* mtlTexture = (MTLComputeTexture*)malloc(sizeof(MTLComputeTexture));
        if (!mtlTexture) {
            return NULL;
        }
        
        mtlTexture->texture = texture;
        mtlTexture->width = width;
        mtlTexture->height = height;
        mtlTexture->depth = 1;
        mtlTexture->format = format;
        mtlTexture->type = MTL_TEXTURE_TYPE_2D;
        
        return mtlTexture;
    }
}

MTLComputeTexture* mtl_compute_texture_create_from_file(MTLComputeDevice* device, const char* filepath) {
    if (!device || !filepath) {
        return NULL;
    }
    
    @autoreleasepool {
        // Load image using CoreGraphics
        NSString* path = [NSString stringWithUTF8String:filepath];
        NSURL* url = [NSURL fileURLWithPath:path];
        
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (!source) {
            fprintf(stderr, "Failed to load image from: %s\n", filepath);
            return NULL;
        }
        
        CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        CFRelease(source);
        
        if (!cgImage) {
            fprintf(stderr, "Failed to decode image\n");
            return NULL;
        }
        
        size_t width = CGImageGetWidth(cgImage);
        size_t height = CGImageGetHeight(cgImage);
        
        // Create pixel buffer
        size_t bytesPerPixel = 4;
        size_t bytesPerRow = width * bytesPerPixel;
        size_t dataSize = height * bytesPerRow;
        unsigned char* pixels = (unsigned char*)malloc(dataSize);
        
        if (!pixels) {
            CGImageRelease(cgImage);
            return NULL;
        }
        
        // Create context and draw image
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(
            pixels,
            width,
            height,
            8,
            bytesPerRow,
            colorSpace,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
        );
        
        CGColorSpaceRelease(colorSpace);
        
        if (!context) {
            free(pixels);
            CGImageRelease(cgImage);
            fprintf(stderr, "Failed to create bitmap context\n");
            return NULL;
        }
        
        // Draw the image into the context
        CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
        CGContextRelease(context);
        CGImageRelease(cgImage);
        
        // Create Metal texture with the pixel data
        MTLComputeTexture* texture = mtl_compute_texture_create(
            device, width, height, MTL_PIXEL_FORMAT_RGBA8, pixels
        );
        
        free(pixels);
        
        if (texture) {
            printf("Loaded image: %s (%zu x %zu)\n", filepath, width, height);
        }
        
        return texture;
    }
}

int mtl_compute_texture_save_to_file(MTLComputeTexture* texture, const char* filepath) {
    if (!texture || !filepath) {
        return -1;
    }
    
    @autoreleasepool {
        size_t width = texture->width;
        size_t height = texture->height;
        size_t bytesPerPixel = 4;
        size_t bytesPerRow = width * bytesPerPixel;
        size_t dataSize = height * bytesPerRow;
        
        // Allocate buffer for pixel data
        unsigned char* pixels = (unsigned char*)malloc(dataSize);
        if (!pixels) {
            return -1;
        }
        
        // Read texture data
        MTLRegion region = MTLRegionMake2D(0, 0, width, height);
        [texture->texture getBytes:pixels
                       bytesPerRow:bytesPerRow
                        fromRegion:region
                       mipmapLevel:0];
        
        // Create CGImage
        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(
            pixels,
            width,
            height,
            8,
            bytesPerRow,
            colorSpace,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
        );
        
        if (!context) {
            CGColorSpaceRelease(colorSpace);
            free(pixels);
            fprintf(stderr, "Failed to create bitmap context\n");
            return -1;
        }
        
        CGImageRef cgImage = CGBitmapContextCreateImage(context);
        CGContextRelease(context);
        CGColorSpaceRelease(colorSpace);
        
        if (!cgImage) {
            free(pixels);
            fprintf(stderr, "Failed to create CGImage\n");
            return -1;
        }
        
        // Write to file
        NSString* path = [NSString stringWithUTF8String:filepath];
        NSURL* url = [NSURL fileURLWithPath:path];
        
        UTType* pngType = [UTType typeWithIdentifier:@"public.png"];
        CGImageDestinationRef destination = CGImageDestinationCreateWithURL(
            (__bridge CFURLRef)url,
            (__bridge CFStringRef)[pngType identifier],
            1,
            NULL
        );
        
        if (!destination) {
            CGImageRelease(cgImage);
            free(pixels);
            fprintf(stderr, "Failed to create image destination\n");
            return -1;
        }
        
        CGImageDestinationAddImage(destination, cgImage, NULL);
        bool success = CGImageDestinationFinalize(destination);
        
        CFRelease(destination);
        CGImageRelease(cgImage);
        free(pixels);
        
        if (success) {
            printf("Saved image: %s\n", filepath);
            return 0;
        } else {
            fprintf(stderr, "Failed to save image\n");
            return -1;
        }
    }
}

void mtl_compute_texture_get_size(
    MTLComputeTexture* texture,
    size_t* width,
    size_t* height
) {
    if (texture) {
        if (width) *width = texture->width;
        if (height) *height = texture->height;
    }
}

void mtl_compute_texture_get_size_3d(
    MTLComputeTexture* texture,
    size_t* width,
    size_t* height,
    size_t* depth
) {
    if (texture) {
        if (width) *width = texture->width;
        if (height) *height = texture->height;
        if (depth) *depth = texture->depth;
    }
}

MTLComputeTextureType mtl_compute_texture_get_type(MTLComputeTexture* texture) {
    if (!texture) {
        return MTL_TEXTURE_TYPE_2D;
    }
    return (MTLComputeTextureType)texture->type;
}

// Create 3D texture
MTLComputeTexture* mtl_compute_texture_create_3d(
    MTLComputeDevice* device,
    size_t width,
    size_t height,
    size_t depth,
    MTLComputePixelFormat format,
    const void* data
) {
    if (!device || width == 0 || height == 0 || depth == 0) {
        return NULL;
    }
    
    @autoreleasepool {
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
        descriptor.textureType = MTLTextureType3D;
        descriptor.pixelFormat = convert_pixel_format(format);
        descriptor.width = width;
        descriptor.height = height;
        descriptor.depth = depth;
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        id<MTLTexture> texture = [device->device newTextureWithDescriptor:descriptor];
        if (!texture) {
            return NULL;
        }
        
        // Upload data if provided
        if (data) {
            size_t bytesPerPixel = 4;  // Assuming RGBA
            size_t bytesPerRow = width * bytesPerPixel;
            size_t bytesPerImage = bytesPerRow * height;
            MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, depth);
            [texture replaceRegion:region
                       mipmapLevel:0
                             slice:0
                         withBytes:data
                       bytesPerRow:bytesPerRow
                     bytesPerImage:bytesPerImage];
        }
        
        MTLComputeTexture* mtlTexture = (MTLComputeTexture*)malloc(sizeof(MTLComputeTexture));
        if (!mtlTexture) {
            return NULL;
        }
        
        mtlTexture->texture = texture;
        mtlTexture->width = width;
        mtlTexture->height = height;
        mtlTexture->depth = depth;
        mtlTexture->format = format;
        mtlTexture->type = MTL_TEXTURE_TYPE_3D;
        
        return mtlTexture;
    }
}

// Create 2D texture array
MTLComputeTexture* mtl_compute_texture_create_array(
    MTLComputeDevice* device,
    size_t width,
    size_t height,
    size_t array_length,
    MTLComputePixelFormat format,
    const void* data
) {
    if (!device || width == 0 || height == 0 || array_length == 0) {
        return NULL;
    }
    
    @autoreleasepool {
        MTLTextureDescriptor* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:convert_pixel_format(format)
                                                                                               width:width
                                                                                              height:height
                                                                                           mipmapped:NO];
        descriptor.textureType = MTLTextureType2DArray;
        descriptor.arrayLength = array_length;
        descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        id<MTLTexture> texture = [device->device newTextureWithDescriptor:descriptor];
        if (!texture) {
            return NULL;
        }
        
        // Upload data if provided (all slices)
        if (data) {
            size_t bytesPerPixel = 4;
            size_t bytesPerRow = width * bytesPerPixel;
            size_t bytesPerImage = bytesPerRow * height;
            
            for (size_t slice = 0; slice < array_length; slice++) {
                MTLRegion region = MTLRegionMake2D(0, 0, width, height);
                const uint8_t* slice_data = (const uint8_t*)data + (slice * bytesPerImage);
                [texture replaceRegion:region
                           mipmapLevel:0
                                 slice:slice
                             withBytes:slice_data
                           bytesPerRow:bytesPerRow
                         bytesPerImage:bytesPerImage];
            }
        }
        
        MTLComputeTexture* mtlTexture = (MTLComputeTexture*)malloc(sizeof(MTLComputeTexture));
        if (!mtlTexture) {
            return NULL;
        }
        
        mtlTexture->texture = texture;
        mtlTexture->width = width;
        mtlTexture->height = height;
        mtlTexture->depth = array_length;  // Store array length as depth
        mtlTexture->format = format;
        mtlTexture->type = MTL_TEXTURE_TYPE_2D_ARRAY;
        
        return mtlTexture;
    }
}

void mtl_compute_texture_get_data(
    MTLComputeTexture* texture,
    MTLComputeDevice* device __unused,
    void* data,
    size_t size
) {
    if (!texture || !data) {
        return;
    }
    
    @autoreleasepool {
        size_t bytesPerRow = texture->width * 4;
        size_t expectedSize = texture->height * bytesPerRow;
        
        if (size < expectedSize) {
            fprintf(stderr, "Buffer too small for texture data\n");
            return;
        }
        
        MTLRegion region = MTLRegionMake2D(0, 0, texture->width, texture->height);
        [texture->texture getBytes:data
                       bytesPerRow:bytesPerRow
                        fromRegion:region
                       mipmapLevel:0];
    }
}

void mtl_compute_texture_destroy(MTLComputeTexture* texture) {
    if (texture) {
        free(texture);
    }
}

void mtl_compute_texture_set_label(MTLComputeTexture* texture, const char* label) {
    if (!texture || !label) {
        return;
    }
    
    @autoreleasepool {
        NSString* labelString = [NSString stringWithUTF8String:label];
        [texture->texture setLabel:labelString];
    }
}

MTLComputeError mtl_compute_dispatch_texture(
    MTLComputeDevice* device,
    MTLComputePipeline* pipeline,
    MTLComputeTexture** textures,
    size_t texture_count,
    MTLComputeBuffer** buffers,
    size_t buffer_count,
    size_t grid_width,
    size_t grid_height,
    size_t threadgroup_width,
    size_t threadgroup_height
) {
    // Use the new unified dispatch system
    MTLComputeDispatchDesc desc = {
        .pipeline = pipeline,
        .buffers = buffers,
        .buffer_count = buffer_count,
        .textures = textures,
        .texture_count = texture_count,
        .grid_width = grid_width,
        .grid_height = grid_height,
        .grid_depth = 1,
        .threadgroup_width = threadgroup_width,
        .threadgroup_height = threadgroup_height,
        .threadgroup_depth = 1
    };
    
    return mtl_compute_dispatch_desc(device, &desc);
}

