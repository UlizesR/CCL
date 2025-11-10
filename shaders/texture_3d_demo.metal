/**
 * 3D Texture Demo
 * Demonstrates volumetric processing with 3D textures
 */

#include <metal_stdlib>
using namespace metal;

// 3D box blur for volumetric data
kernel void blur_3d_volume(
    texture3d<float, access::read> input [[texture(0)]],
    texture3d<float, access::write> output [[texture(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= input.get_width() || 
        gid.y >= input.get_height() || 
        gid.z >= input.get_depth()) {
        return;
    }
    
    // 3x3x3 box filter
    float4 sum = float4(0.0);
    int count = 0;
    
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint3 coord = uint3(int3(gid) + int3(dx, dy, dz));
                if (coord.x < input.get_width() && 
                    coord.y < input.get_height() && 
                    coord.z < input.get_depth()) {
                    sum += input.read(coord);
                    count++;
                }
            }
        }
    }
    
    output.write(sum / float(count), gid);
}

// Ray marching through 3D texture with sampler
kernel void raymarch_volume(
    texture3d<float, access::read> volume [[texture(0)]],
    texture2d<float, access::write> output [[texture(1)]],
    sampler volume_sampler [[sampler(0)]],
    constant float& density_scale [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint2 output_size = uint2(output.get_width(), output.get_height());
    if (gid.x >= output_size.x || gid.y >= output_size.y) {
        return;
    }
    
    // Simple ray marching
    float2 uv = float2(gid) / float2(output_size);
    float3 ray_dir = normalize(float3(uv * 2.0 - 1.0, -1.0));
    float3 ray_pos = float3(0.5, 0.5, 1.0);
    
    float accumulated = 0.0;
    for (int i = 0; i < 64; i++) {
        if (ray_pos.z < 0.0 || ray_pos.z > 1.0) break;
        
        float4 sample_val = volume.sample(volume_sampler, ray_pos);
        accumulated += sample_val.r * density_scale;
        ray_pos += ray_dir * 0.02;
    }
    
    output.write(float4(accumulated), gid);
}

