/**
 * MTL Compute Core - Umbrella Header
 * 
 * A comprehensive C/Objective-C library for Metal compute shaders with
 * Metal 3/4 features built-in.
 * 
 * 
 * Usage:
 *   #include "mtl_compute_core.h"  // Everything
 * 
 * Or include individual headers:
 *   #include "mtl_compute.h"  // Core + advanced features
 *   #include "mtl_texture.h"  // Texture operations
 * 
 * Note: mtl_compute_advanced.h is now deprecated - all features moved to
 * mtl_compute.h with runtime capability checks.
 */

#ifndef MTL_COMPUTE_CORE_H
#define MTL_COMPUTE_CORE_H

// Core compute API - Includes ALL features (basic + advanced)
#include "mtl_compute.h"

// Texture support
#include "mtl_texture.h"

/**
 * Library Version Information
 */
#define MTL_COMPUTE_VERSION_MAJOR 2
#define MTL_COMPUTE_VERSION_MINOR 0
#define MTL_COMPUTE_VERSION_PATCH 0

#define MTL_COMPUTE_VERSION_STRING "2.0.0"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get library version string
 * @return Version string (e.g., "2.0.0")
 */
static inline const char* mtl_compute_version(void) {
    return MTL_COMPUTE_VERSION_STRING;
}

/**
 * API Tiers - All tiers use the same unified internal encoder:
 * 
 * TIER 1 - IMMEDIATE MODE (Simplest):
 *   mtl_compute_device_create()
 *   mtl_compute_pipeline_create()
 *   mtl_compute_buffer_create()
 *   mtl_compute_dispatch_sync()  // or mtl_compute_dispatch()
 *   
 *   Best for: Prototypes, simple kernels, learning
 * 
 * TIER 2 - DESCRIPTOR MODE (Unified Resources):
 *   MTLComputeDispatchDesc desc = {...};  // Buffers + textures together
 *   mtl_compute_dispatch_desc(&desc)
 *   
 *   Best for: Image processing, mixed resource types
 * 
 * TIER 3 - ENCODER MODE (Batching):
 *   mtl_compute_begin(&cmdList)
 *   mtl_compute_encode_dispatch(cmdList, &desc)  // Multiple times
 *   mtl_compute_end_submit(cmdList)  // or _nowait for async
 *   
 *   Best for: Multiple dispatches, async execution
 * 
 * TIER 4 - METAL 3/4 FEATURES (Always Available, Runtime-Checked):
 *   // Query first:
 *   MTLComputeDeviceCapabilities caps;
 *   mtl_compute_device_get_capabilities(device, &caps);
 *   
 *   // Async with events:
 *   if (caps.supports_shared_events)
 *       mtl_compute_dispatch_async(..., event, value)
 *   
 *   // Profiling:
 *   mtl_compute_dispatch_profiled(..., &stats)
 *   
 *   // Indirect:
 *   if (caps.supports_indirect_dispatch)
 *       mtl_compute_dispatch_indirect(..., indirect_buffer, ...)
 *   
 *   // Heaps:
 *   if (caps.supports_heaps)
 *       heap = mtl_compute_heap_create(...)
 *   
 *   Best for: Production, complex pipelines, modern hardware
 *   
 * Note: All functions return MTL_ERROR_UNSUPPORTED if feature unavailable.
 */

#ifdef __cplusplus
}
#endif

#endif // MTL_COMPUTE_CORE_H

