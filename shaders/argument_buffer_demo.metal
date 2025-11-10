/**
 * Argument Buffer Demo - MSL 2.13
 * Demonstrates layout-aware argument buffers
 */

#include <metal_stdlib>
using namespace metal;

// Argument buffer layout matching MTLComputeArgDesc setup
struct ComputeResources {
    device float* input_buffer [[id(0)]];
    device float* output_buffer [[id(1)]];
    texture2d<float, access::read> input_texture [[id(2)]];
    sampler tex_sampler [[id(3)]];
};

kernel void process_with_argbuf(
    constant ComputeResources& resources [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    uint idx = gid.y * grid_size.x + gid.x;
    
    // Use buffer from argument buffer
    float value = resources.input_buffer[idx];
    
    // Sample texture using sampler from argument buffer
    float2 uv = float2(gid) / float2(grid_size);
    float4 color = resources.input_texture.sample(resources.tex_sampler, uv);
    
    // Write result
    resources.output_buffer[idx] = value * color.r;
}

