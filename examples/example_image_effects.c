/**
 * Example: Image Effects using Metal Compute Shaders
 * Demonstrates texture processing with various image effects
 */

#include "mtl_compute.h"
#include "mtl_texture.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_menu() {
    printf("\n=== Available Image Effects ===\n");
    printf("1. Grayscale\n");
    printf("2. Sepia\n");
    printf("3. Invert Colors\n");
    printf("4. Brightness & Contrast\n");
    printf("5. Blur\n");
    printf("6. Edge Detection\n");
    printf("7. Sharpen\n");
    printf("8. Vignette\n");
    printf("9. Pixelate\n");
    printf("0. Apply All Effects\n");
    printf("================================\n\n");
}

int apply_effect(
    MTLComputeDevice* device,
    MTLComputeTexture* input,
    MTLComputeTexture* output,
    const char* effect_name,
    const char* kernel_name,
    MTLComputeBuffer** params,
    size_t param_count
) {
    printf("Applying %s effect...\n", effect_name);
    
    size_t width, height;
    mtl_compute_texture_get_size(input, &width, &height);
    
    // Compile shader with detailed error logging
    MTLComputeError error;
    char error_log[1024] = {0};
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create_from_file(
        device, "shaders/image_effects.metal", kernel_name, &error, error_log, sizeof(error_log)
    );
    
    if (!pipeline) {
        fprintf(stderr, "Failed to create pipeline for %s: %s\n", 
                effect_name, mtl_compute_error_string(error));
        if (error_log[0]) {
            fprintf(stderr, "  Details: %s\n", error_log);
        }
        return -1;
    }
    
    // Execute shader
    MTLComputeTexture* textures[] = {input, output};
    clock_t start = clock();
    
    error = mtl_compute_dispatch_texture(
        device, pipeline, textures, 2,
        params, param_count,
        width, height,
        16, 16  // 16x16 threadgroups work well for images
    );
    
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;
    
    if (error != MTL_SUCCESS) {
        fprintf(stderr, "Failed to execute %s: %s\n", 
                effect_name, mtl_compute_error_string(error));
        mtl_compute_pipeline_destroy(pipeline);
        return -1;
    }
    
    printf("  Completed in %.2f ms\n", elapsed);
    
    mtl_compute_pipeline_destroy(pipeline);
    return 0;
}

int main(int argc, char** argv) {
    printf("=== Metal Compute: Image Effects Example ===\n\n");
    
    // Check arguments
    if (argc < 2) {
        printf("Usage: %s <input_image> [output_directory]\n", argv[0]);
        printf("Example: %s images/photo.jpg output/\n\n", argv[0]);
        return 1;
    }
    
    const char* input_path = argv[1];
    const char* output_dir = argc >= 3 ? argv[2] : "output";
    
    // Initialize Metal device
    printf("Initializing Metal device...\n");
    MTLComputeDevice* device = mtl_compute_device_create();
    if (!device) {
        fprintf(stderr, "Failed to create Metal device\n");
        return 1;
    }
    
    printf("Using device: %s\n\n", mtl_compute_device_get_name(device));
    
    // Load input image
    printf("Loading image: %s\n", input_path);
    MTLComputeTexture* input_texture = mtl_compute_texture_create_from_file(device, input_path);
    if (!input_texture) {
        fprintf(stderr, "Failed to load image\n");
        mtl_compute_device_destroy(device);
        return 1;
    }
    
    size_t width, height;
    mtl_compute_texture_get_size(input_texture, &width, &height);
    printf("Image size: %zu x %zu pixels\n", width, height);
    
    // Create output texture
    MTLComputeTexture* output_texture = mtl_compute_texture_create(
        device, width, height, MTL_PIXEL_FORMAT_RGBA8, NULL
    );
    
    if (!output_texture) {
        fprintf(stderr, "Failed to create output texture\n");
        mtl_compute_texture_destroy(input_texture);
        mtl_compute_device_destroy(device);
        return 1;
    }
    
    // Show menu and get selection
    print_menu();
    printf("Select an effect (0-9): ");
    int choice;
    if (scanf("%d", &choice) != 1) {
        choice = 0;
    }
    
    char output_path[512];
    
    // Apply selected effect(s)
    if (choice == 0) {
        // Apply all effects
        printf("\nApplying all effects...\n\n");
        
        // 1. Grayscale
        snprintf(output_path, sizeof(output_path), "%s/grayscale.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Grayscale", "grayscale", NULL, 0) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        
        // 2. Sepia
        snprintf(output_path, sizeof(output_path), "%s/sepia.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Sepia", "sepia", NULL, 0) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        
        // 3. Invert
        snprintf(output_path, sizeof(output_path), "%s/invert.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Invert", "invert", NULL, 0) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        
        // 4. Brightness & Contrast
        float brightness = 0.2f;
        float contrast = 1.5f;
        MTLComputeBuffer* buf_brightness = mtl_compute_buffer_create_with_data(
            device, &brightness, sizeof(float), MTL_STORAGE_SHARED
        );
        MTLComputeBuffer* buf_contrast = mtl_compute_buffer_create_with_data(
            device, &contrast, sizeof(float), MTL_STORAGE_SHARED
        );
        MTLComputeBuffer* params[] = {buf_brightness, buf_contrast};
        
        snprintf(output_path, sizeof(output_path), "%s/bright_contrast.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Brightness & Contrast", 
                        "brightness_contrast", params, 2) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        mtl_compute_buffer_destroy(buf_brightness);
        mtl_compute_buffer_destroy(buf_contrast);
        
        // 5. Blur
        int blur_radius = 5;
        MTLComputeBuffer* buf_radius = mtl_compute_buffer_create_with_data(
            device, &blur_radius, sizeof(int), MTL_STORAGE_SHARED
        );
        MTLComputeBuffer* blur_params[] = {buf_radius};
        
        snprintf(output_path, sizeof(output_path), "%s/blur.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Blur", 
                        "blur", blur_params, 1) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        mtl_compute_buffer_destroy(buf_radius);
        
        // 6. Edge Detection
        snprintf(output_path, sizeof(output_path), "%s/edges.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Edge Detection", 
                        "edge_detect", NULL, 0) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        
        // 7. Sharpen
        float sharpen_amount = 2.0f;
        MTLComputeBuffer* buf_sharpen = mtl_compute_buffer_create_with_data(
            device, &sharpen_amount, sizeof(float), MTL_STORAGE_SHARED
        );
        MTLComputeBuffer* sharpen_params[] = {buf_sharpen};
        
        snprintf(output_path, sizeof(output_path), "%s/sharpen.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Sharpen", 
                        "sharpen", sharpen_params, 1) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        mtl_compute_buffer_destroy(buf_sharpen);
        
        // 8. Vignette
        float vignette_strength = 2.0f;
        MTLComputeBuffer* buf_vignette = mtl_compute_buffer_create_with_data(
            device, &vignette_strength, sizeof(float), MTL_STORAGE_SHARED
        );
        MTLComputeBuffer* vignette_params[] = {buf_vignette};
        
        snprintf(output_path, sizeof(output_path), "%s/vignette.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Vignette", 
                        "vignette", vignette_params, 1) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        mtl_compute_buffer_destroy(buf_vignette);
        
        // 9. Pixelate
        int pixel_size = 10;
        MTLComputeBuffer* buf_pixel = mtl_compute_buffer_create_with_data(
            device, &pixel_size, sizeof(int), MTL_STORAGE_SHARED
        );
        MTLComputeBuffer* pixel_params[] = {buf_pixel};
        
        snprintf(output_path, sizeof(output_path), "%s/pixelate.png", output_dir);
        if (apply_effect(device, input_texture, output_texture, "Pixelate", 
                        "pixelate", pixel_params, 1) == 0) {
            mtl_compute_texture_save_to_file(output_texture, output_path);
        }
        mtl_compute_buffer_destroy(buf_pixel);
        
        printf("\n✓ All effects applied successfully!\n");
        printf("Output files saved to: %s/\n", output_dir);
        
    } else {
        // Apply single effect
        printf("\nProcessing...\n\n");
        
        MTLComputeBuffer** params = NULL;
        size_t param_count = 0;
        const char* effect_name = "";
        const char* kernel_name = "";
        const char* filename = "";
        
        switch (choice) {
            case 1:
                effect_name = "Grayscale";
                kernel_name = "grayscale";
                filename = "grayscale.png";
                break;
            case 2:
                effect_name = "Sepia";
                kernel_name = "sepia";
                filename = "sepia.png";
                break;
            case 3:
                effect_name = "Invert";
                kernel_name = "invert";
                filename = "invert.png";
                break;
            case 4: {
                effect_name = "Brightness & Contrast";
                kernel_name = "brightness_contrast";
                filename = "bright_contrast.png";
                float brightness = 0.2f;
                float contrast = 1.5f;
                MTLComputeBuffer* buf_b = mtl_compute_buffer_create_with_data(
                    device, &brightness, sizeof(float), MTL_STORAGE_SHARED
                );
                MTLComputeBuffer* buf_c = mtl_compute_buffer_create_with_data(
                    device, &contrast, sizeof(float), MTL_STORAGE_SHARED
                );
                static MTLComputeBuffer* temp_params[2];
                temp_params[0] = buf_b;
                temp_params[1] = buf_c;
                params = temp_params;
                param_count = 2;
                break;
            }
            case 5: {
                effect_name = "Blur";
                kernel_name = "blur";
                filename = "blur.png";
                int radius = 5;
                MTLComputeBuffer* buf_r = mtl_compute_buffer_create_with_data(
                    device, &radius, sizeof(int), MTL_STORAGE_SHARED
                );
                static MTLComputeBuffer* temp_params[1];
                temp_params[0] = buf_r;
                params = temp_params;
                param_count = 1;
                break;
            }
            case 6:
                effect_name = "Edge Detection";
                kernel_name = "edge_detect";
                filename = "edges.png";
                break;
            case 7: {
                effect_name = "Sharpen";
                kernel_name = "sharpen";
                filename = "sharpen.png";
                float amount = 2.0f;
                MTLComputeBuffer* buf_a = mtl_compute_buffer_create_with_data(
                    device, &amount, sizeof(float), MTL_STORAGE_SHARED
                );
                static MTLComputeBuffer* temp_params[1];
                temp_params[0] = buf_a;
                params = temp_params;
                param_count = 1;
                break;
            }
            case 8: {
                effect_name = "Vignette";
                kernel_name = "vignette";
                filename = "vignette.png";
                float strength = 2.0f;
                MTLComputeBuffer* buf_s = mtl_compute_buffer_create_with_data(
                    device, &strength, sizeof(float), MTL_STORAGE_SHARED
                );
                static MTLComputeBuffer* temp_params[1];
                temp_params[0] = buf_s;
                params = temp_params;
                param_count = 1;
                break;
            }
            case 9: {
                effect_name = "Pixelate";
                kernel_name = "pixelate";
                filename = "pixelate.png";
                int size = 10;
                MTLComputeBuffer* buf_s = mtl_compute_buffer_create_with_data(
                    device, &size, sizeof(int), MTL_STORAGE_SHARED
                );
                static MTLComputeBuffer* temp_params[1];
                temp_params[0] = buf_s;
                params = temp_params;
                param_count = 1;
                break;
            }
            default:
                fprintf(stderr, "Invalid choice\n");
                goto cleanup;
        }
        
        if (apply_effect(device, input_texture, output_texture, effect_name, 
                        kernel_name, params, param_count) == 0) {
            snprintf(output_path, sizeof(output_path), "%s/%s", output_dir, filename);
            mtl_compute_texture_save_to_file(output_texture, output_path);
            printf("\n✓ Effect applied successfully!\n");
            printf("Output saved to: %s\n", output_path);
        }
        
        // Cleanup parameters
        for (size_t i = 0; i < param_count; i++) {
            mtl_compute_buffer_destroy(params[i]);
        }
    }
    
cleanup:
    // Cleanup
    printf("\nCleaning up...\n");
    mtl_compute_texture_destroy(input_texture);
    mtl_compute_texture_destroy(output_texture);
    mtl_compute_device_destroy(device);
    
    printf("Done!\n");
    return 0;
}

