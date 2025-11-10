/**
 * MTL Texture Support
 * Extension to MTL Compute Library for texture operations
 */

#ifndef MTL_TEXTURE_H
#define MTL_TEXTURE_H

#include "mtl_compute.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque type for Metal textures
typedef struct MTLComputeTexture MTLComputeTexture;

// Texture pixel formats
typedef enum {
    MTL_PIXEL_FORMAT_RGBA8 = 0,
    MTL_PIXEL_FORMAT_BGRA8 = 1,
    MTL_PIXEL_FORMAT_RGBA32F = 2,
    MTL_PIXEL_FORMAT_R32F = 3,
    MTL_PIXEL_FORMAT_RG32F = 4
} MTLComputePixelFormat;

// Texture types
typedef enum {
    MTL_TEXTURE_TYPE_2D = 0,
    MTL_TEXTURE_TYPE_3D = 1,
    MTL_TEXTURE_TYPE_2D_ARRAY = 2,
    MTL_TEXTURE_TYPE_CUBE = 3
} MTLComputeTextureType;

/**
 * Create a texture from raw pixel data
 * @param device Device context
 * @param width Texture width in pixels
 * @param height Texture height in pixels
 * @param format Pixel format
 * @param data Raw pixel data (can be NULL for empty texture)
 * @return Texture object, or NULL on failure
 */
MTLComputeTexture* mtl_compute_texture_create(
    MTLComputeDevice* device,
    size_t width,
    size_t height,
    MTLComputePixelFormat format,
    const void* data
);

/**
 * Load texture from image file (JPEG, PNG, etc.)
 * @param device Device context
 * @param filepath Path to image file
 * @return Texture object, or NULL on failure
 */
MTLComputeTexture* mtl_compute_texture_create_from_file(
    MTLComputeDevice* device,
    const char* filepath
);

/**
 * Save texture to image file (PNG)
 * @param texture Texture object
 * @param filepath Output path
 * @return 0 on success, -1 on failure
 */
int mtl_compute_texture_save_to_file(
    MTLComputeTexture* texture,
    const char* filepath
);

/**
 * Create a 3D texture
 * @param device Device context
 * @param width Texture width in pixels
 * @param height Texture height in pixels
 * @param depth Texture depth in slices
 * @param format Pixel format
 * @param data Raw pixel data (can be NULL for empty texture)
 * @return Texture object, or NULL on failure
 */
MTLComputeTexture* mtl_compute_texture_create_3d(
    MTLComputeDevice* device,
    size_t width,
    size_t height,
    size_t depth,
    MTLComputePixelFormat format,
    const void* data
);

/**
 * Create a 2D texture array
 * @param device Device context
 * @param width Texture width in pixels
 * @param height Texture height in pixels
 * @param array_length Number of array slices
 * @param format Pixel format
 * @param data Raw pixel data (can be NULL for empty texture)
 * @return Texture object, or NULL on failure
 */
MTLComputeTexture* mtl_compute_texture_create_array(
    MTLComputeDevice* device,
    size_t width,
    size_t height,
    size_t array_length,
    MTLComputePixelFormat format,
    const void* data
);

/**
 * Get texture dimensions
 * @param texture Texture object
 * @param width Output width
 * @param height Output height
 */
void mtl_compute_texture_get_size(
    MTLComputeTexture* texture,
    size_t* width,
    size_t* height
);

/**
 * Get 3D texture dimensions
 * @param texture Texture object
 * @param width Output width
 * @param height Output height
 * @param depth Output depth
 */
void mtl_compute_texture_get_size_3d(
    MTLComputeTexture* texture,
    size_t* width,
    size_t* height,
    size_t* depth
);

/**
 * Get texture type
 * @param texture Texture object
 * @return Texture type
 */
MTLComputeTextureType mtl_compute_texture_get_type(MTLComputeTexture* texture);

/**
 * Get raw pixel data from texture
 * @param texture Texture object
 * @param device Device context
 * @param data Output buffer (must be pre-allocated)
 * @param size Size of output buffer in bytes
 */
void mtl_compute_texture_get_data(
    MTLComputeTexture* texture,
    MTLComputeDevice* device,
    void* data,
    size_t size
);

/**
 * Release a texture
 * @param texture Texture to release
 */
void mtl_compute_texture_destroy(MTLComputeTexture* texture);

/**
 * Set debug label on texture
 * @param texture Texture
 * @param label Debug label string
 */
void mtl_compute_texture_set_label(MTLComputeTexture* texture, const char* label);

/**
 * Execute a compute shader with textures (DEPRECATED - use mtl_compute_dispatch_desc instead)
 * @param device Device context
 * @param pipeline Compute pipeline
 * @param textures Array of texture arguments
 * @param texture_count Number of textures
 * @param buffers Array of buffer arguments (can be NULL)
 * @param buffer_count Number of buffers
 * @param grid_width Grid width (usually texture width)
 * @param grid_height Grid height (usually texture height)
 * @param threadgroup_width Threadgroup width (e.g., 16)
 * @param threadgroup_height Threadgroup height (e.g., 16)
 * @return Error code
 */
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
);

#ifdef __cplusplus
}
#endif

#endif // MTL_TEXTURE_H

