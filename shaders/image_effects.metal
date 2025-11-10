#include <metal_stdlib>
using namespace metal;

/**
 * Image Processing Compute Shaders
 * Various effects for processing images
 */

// Grayscale conversion
kernel void grayscale(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    float4 color = inTexture.read(gid);
    float gray = dot(color.rgb, float3(0.299, 0.587, 0.114));
    outTexture.write(float4(gray, gray, gray, color.a), gid);
}

// Sepia tone effect
kernel void sepia(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    float4 color = inTexture.read(gid);
    
    float r = color.r * 0.393 + color.g * 0.769 + color.b * 0.189;
    float g = color.r * 0.349 + color.g * 0.686 + color.b * 0.168;
    float b = color.r * 0.272 + color.g * 0.534 + color.b * 0.131;
    
    outTexture.write(float4(r, g, b, color.a), gid);
}

// Invert colors
kernel void invert(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    float4 color = inTexture.read(gid);
    outTexture.write(float4(1.0 - color.rgb, color.a), gid);
}

// Brightness and contrast adjustment
kernel void brightness_contrast(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant float& brightness [[buffer(0)]],
    constant float& contrast [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    float4 color = inTexture.read(gid);
    
    // Apply brightness
    color.rgb += brightness;
    
    // Apply contrast
    color.rgb = ((color.rgb - 0.5) * contrast) + 0.5;
    
    // Clamp
    color = clamp(color, 0.0, 1.0);
    
    outTexture.write(color, gid);
}

// Box blur
kernel void blur(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant int& radius [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    float4 sum = float4(0.0);
    int count = 0;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int2 coord = int2(gid) + int2(dx, dy);
            
            if (coord.x >= 0 && coord.x < int(inTexture.get_width()) &&
                coord.y >= 0 && coord.y < int(inTexture.get_height())) {
                sum += inTexture.read(uint2(coord));
                count++;
            }
        }
    }
    
    outTexture.write(sum / float(count), gid);
}

// Edge detection (Sobel operator)
kernel void edge_detect(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    int width = int(inTexture.get_width());
    int height = int(inTexture.get_height());
    int x = int(gid.x);
    int y = int(gid.y);
    
    // Sobel kernels
    float Gx = 0.0;
    float Gy = 0.0;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int2 coord = int2(x + dx, y + dy);
            
            if (coord.x >= 0 && coord.x < width && coord.y >= 0 && coord.y < height) {
                float4 color = inTexture.read(uint2(coord));
                float intensity = dot(color.rgb, float3(0.299, 0.587, 0.114));
                
                // Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
                float kx = float((dx == -1) ? -1 : (dx == 1) ? 1 : 0) * 
                          float((dy == 0) ? 2 : 1);
                
                // Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
                float ky = float((dy == -1) ? -1 : (dy == 1) ? 1 : 0) * 
                          float((dx == 0) ? 2 : 1);
                
                Gx += intensity * kx;
                Gy += intensity * ky;
            }
        }
    }
    
    float magnitude = sqrt(Gx * Gx + Gy * Gy);
    magnitude = clamp(magnitude, 0.0, 1.0);
    
    outTexture.write(float4(magnitude, magnitude, magnitude, 1.0), gid);
}

// Sharpen filter
kernel void sharpen(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant float& amount [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    float4 center = inTexture.read(gid);
    
    // Laplacian kernel for sharpening
    float4 sum = float4(0.0);
    int count = 0;
    
    int2 offsets[4] = {
        int2(-1, 0), int2(1, 0),
        int2(0, -1), int2(0, 1)
    };
    
    for (int i = 0; i < 4; i++) {
        int2 coord = int2(gid) + offsets[i];
        if (coord.x >= 0 && coord.x < int(inTexture.get_width()) &&
            coord.y >= 0 && coord.y < int(inTexture.get_height())) {
            sum += inTexture.read(uint2(coord));
            count++;
        }
    }
    
    float4 avg = sum / float(count);
    float4 sharpened = center + (center - avg) * amount;
    
    outTexture.write(clamp(sharpened, 0.0, 1.0), gid);
}

// Vignette effect
kernel void vignette(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant float& strength [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    float4 color = inTexture.read(gid);
    
    float width = float(inTexture.get_width());
    float height = float(inTexture.get_height());
    
    // Normalized coordinates (0 to 1)
    float2 uv = float2(gid) / float2(width, height);
    
    // Center coordinates (-0.5 to 0.5)
    float2 centered = uv - 0.5;
    
    // Distance from center
    float dist = length(centered);
    
    // Vignette factor
    float vignette = 1.0 - smoothstep(0.3, 0.7, dist * strength);
    
    color.rgb *= vignette;
    
    outTexture.write(color, gid);
}

// Pixelate effect
kernel void pixelate(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant int& pixelSize [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }
    
    // Find the pixel block this thread belongs to
    uint2 blockCoord = uint2(gid.x / pixelSize * pixelSize, 
                             gid.y / pixelSize * pixelSize);
    
    // Sample from the top-left corner of the block
    float4 color = inTexture.read(blockCoord);
    
    outTexture.write(color, gid);
}

