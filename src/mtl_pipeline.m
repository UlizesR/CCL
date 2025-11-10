/**
 * MTL Pipeline Management
 * Pipeline creation, compilation, library management, and reflection
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "mtl_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Pipeline creation from source
MTLComputePipeline* mtl_compute_pipeline_create(
    MTLComputeDevice* device,
    const char* source,
    const char* function_name,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
) {
    if (!device || !source || !function_name) {
        if (error) *error = MTL_ERROR_INVALID_PARAMETER;
        mtl_copy_error_log("Invalid parameter", error_log, error_log_size);
        return NULL;
    }
    
    @autoreleasepool {
        NSString* sourceString = [NSString stringWithUTF8String:source];
        NSError* compileError = nil;
        
        id<MTLLibrary> library = [device->device newLibraryWithSource:sourceString
                                                               options:nil
                                                                 error:&compileError];
        if (!library) {
            const char* errMsg = [[compileError localizedDescription] UTF8String];
            MTL_LOG("Shader compilation failed: %s", errMsg);
            mtl_copy_error_log(errMsg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_SHADER_COMPILATION;
            return NULL;
        }
        
        NSString* functionNameString = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> function = [library newFunctionWithName:functionNameString];
        if (!function) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Function '%s' not found in shader", function_name);
            MTL_LOG("%s", msg);
            mtl_copy_error_log(msg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_SHADER_COMPILATION;
            return NULL;
        }
        
        NSError* pipelineError = nil;
        id<MTLComputePipelineState> pipelineState = [device->device newComputePipelineStateWithFunction:function
                                                                                                  error:&pipelineError];
        if (!pipelineState) {
            const char* errMsg = [[pipelineError localizedDescription] UTF8String];
            MTL_LOG("Pipeline creation failed: %s", errMsg);
            mtl_copy_error_log(errMsg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_PIPELINE_CREATION;
            return NULL;
        }
        
        MTLComputePipeline* pipeline = (MTLComputePipeline*)malloc(sizeof(MTLComputePipeline));
        if (!pipeline) {
            if (error) *error = MTL_ERROR_PIPELINE_CREATION;
            mtl_copy_error_log("Memory allocation failed", error_log, error_log_size);
            return NULL;
        }
        
        pipeline->pipelineState = pipelineState;
        pipeline->function = function;
        
        if (error) *error = MTL_SUCCESS;
        return pipeline;
    }
}

// Pipeline creation from file
MTLComputePipeline* mtl_compute_pipeline_create_from_file(
    MTLComputeDevice* device,
    const char* filepath,
    const char* function_name,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
) {
    if (!filepath) {
        if (error) *error = MTL_ERROR_INVALID_PARAMETER;
        mtl_copy_error_log("Invalid filepath parameter", error_log, error_log_size);
        return NULL;
    }
    
    FILE* file = fopen(filepath, "r");
    if (!file) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Failed to open shader file: %s", filepath);
        MTL_LOG("%s", msg);
        mtl_copy_error_log(msg, error_log, error_log_size);
        if (error) *error = MTL_ERROR_IO;
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size < 0) {
        fclose(file);
        mtl_copy_error_log("Failed to determine file size", error_log, error_log_size);
        if (error) *error = MTL_ERROR_IO;
        return NULL;
    }
    
    char* source = (char*)malloc(file_size + 1);
    if (!source) {
        fclose(file);
        mtl_copy_error_log("Memory allocation failed", error_log, error_log_size);
        if (error) *error = MTL_ERROR_IO;
        return NULL;
    }
    
    size_t read_count = fread(source, 1, file_size, file);
    fclose(file);
    
    if ((long)read_count != file_size) {
        free(source);
        mtl_copy_error_log("Failed to read complete file", error_log, error_log_size);
        if (error) *error = MTL_ERROR_IO;
        return NULL;
    }
    
    source[file_size] = '\0';
    
    MTLComputePipeline* pipeline = mtl_compute_pipeline_create(device, source, function_name, error, error_log, error_log_size);
    free(source);
    
    return pipeline;
}

void mtl_compute_pipeline_destroy(MTLComputePipeline* pipeline) {
    if (pipeline) {
        free(pipeline);
    }
}

size_t mtl_compute_pipeline_max_threads_per_threadgroup(MTLComputePipeline* pipeline) {
    if (!pipeline || !pipeline->pipelineState) {
        return 0;
    }
    @autoreleasepool {
        return [pipeline->pipelineState maxTotalThreadsPerThreadgroup];
    }
}

// Pipeline creation with function constants
MTLComputePipeline* mtl_compute_pipeline_create_with_constants(
    MTLComputeDevice* device,
    const char* source,
    const char* function_name,
    const MTLComputeFunctionConstant* constants,
    size_t constant_count,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
) {
    if (!device || !source || !function_name) {
        if (error) *error = MTL_ERROR_INVALID_PARAMETER;
        mtl_copy_error_log("Invalid parameter", error_log, error_log_size);
        return NULL;
    }
    
    @autoreleasepool {
        NSString* sourceString = [NSString stringWithUTF8String:source];
        NSError* compileError = nil;
        
        id<MTLLibrary> library = [device->device newLibraryWithSource:sourceString
                                                               options:nil
                                                                 error:&compileError];
        if (!library) {
            const char* errMsg = [[compileError localizedDescription] UTF8String];
            MTL_LOG("Shader compilation failed: %s", errMsg);
            mtl_copy_error_log(errMsg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_SHADER_COMPILATION;
            return NULL;
        }
        
        NSString* functionNameString = [NSString stringWithUTF8String:function_name];
        
        // Build function constants
        MTLFunctionConstantValues* constantValues = nil;
        if (constants && constant_count > 0) {
            constantValues = [[MTLFunctionConstantValues alloc] init];
            
            for (size_t i = 0; i < constant_count; i++) {
                const MTLComputeFunctionConstant* c = &constants[i];
                MTLDataType dataType;
                const void* valuePtr = NULL;
                
                switch (c->type) {
                    case MTL_CONSTANT_BOOL:
                        dataType = MTLDataTypeBool;
                        valuePtr = &c->value.bool_value;
                        break;
                    case MTL_CONSTANT_INT:
                        dataType = MTLDataTypeInt;
                        valuePtr = &c->value.int_value;
                        break;
                    case MTL_CONSTANT_FLOAT:
                        dataType = MTLDataTypeFloat;
                        valuePtr = &c->value.float_value;
                        break;
                    default:
                        continue;
                }
                
                [constantValues setConstantValue:valuePtr type:dataType atIndex:c->index];
            }
        }
        
        NSError* functionError = nil;
        id<MTLFunction> function = [library newFunctionWithName:functionNameString
                                                 constantValues:constantValues
                                                          error:&functionError];
        if (!function) {
            const char* errMsg = functionError ? [[functionError localizedDescription] UTF8String] : "Function not found";
            MTL_LOG("Function creation failed: %s", errMsg);
            mtl_copy_error_log(errMsg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_SHADER_COMPILATION;
            return NULL;
        }
        
        NSError* pipelineError = nil;
        id<MTLComputePipelineState> pipelineState = 
            [device->device newComputePipelineStateWithFunction:function
                                                          error:&pipelineError];
        if (!pipelineState) {
            const char* errMsg = [[pipelineError localizedDescription] UTF8String];
            MTL_LOG("Pipeline creation failed: %s", errMsg);
            mtl_copy_error_log(errMsg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_PIPELINE_CREATION;
            return NULL;
        }
        
        MTLComputePipeline* pipeline = (MTLComputePipeline*)malloc(sizeof(MTLComputePipeline));
        if (!pipeline) {
            if (error) *error = MTL_ERROR_PIPELINE_CREATION;
            mtl_copy_error_log("Memory allocation failed", error_log, error_log_size);
            return NULL;
        }
        
        pipeline->pipelineState = pipelineState;
        pipeline->function = function;
        
        if (error) *error = MTL_SUCCESS;
        return pipeline;
    }
}

// Pipeline creation with compile options
MTLComputePipeline* mtl_compute_pipeline_create_ex(
    MTLComputeDevice* device,
    const char* source,
    const char* function_name,
    const MTLComputeShaderOptions* options,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
) {
    if (!device || !source || !function_name || !options) {
        if (error) *error = MTL_ERROR_INVALID_PARAMETER;
        mtl_copy_error_log("Invalid parameter", error_log, error_log_size);
        return NULL;
    }
    
    @autoreleasepool {
        NSString* sourceString = [NSString stringWithUTF8String:source];
        MTLCompileOptions* compileOptions = [[MTLCompileOptions alloc] init];
        
        // Set language version
        if (options->language_version) {
            NSString* version = [NSString stringWithUTF8String:options->language_version];
            if ([version isEqualToString:@"3.0"]) {
                compileOptions.languageVersion = MTLLanguageVersion3_0;
            } else if ([version isEqualToString:@"2.4"]) {
                compileOptions.languageVersion = MTLLanguageVersion2_4;
            }
        }
        
        // Set fast math (suppress deprecation warning - we handle both APIs)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        if (options->fast_math_enabled) {
            compileOptions.fastMathEnabled = YES;
        }
#pragma clang diagnostic pop
        
        // Set preprocessor macros
        if (options->preprocessor_macros) {
            NSMutableDictionary* macros = [[NSMutableDictionary alloc] init];
            for (size_t i = 0; options->preprocessor_macros[i] != NULL; i++) {
                NSString* macro = [NSString stringWithUTF8String:options->preprocessor_macros[i]];
                NSArray* parts = [macro componentsSeparatedByString:@"="];
                if ([parts count] == 2) {
                    macros[parts[0]] = parts[1];
                } else {
                    macros[parts[0]] = @"1";
                }
            }
            compileOptions.preprocessorMacros = macros;
        }
        
        NSError* compileError = nil;
        id<MTLLibrary> library = [device->device newLibraryWithSource:sourceString
                                                               options:compileOptions
                                                                 error:&compileError];
        if (!library) {
            const char* errMsg = [[compileError localizedDescription] UTF8String];
            mtl_copy_error_log(errMsg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_SHADER_COMPILATION;
            return NULL;
        }
        
        NSString* functionNameString = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> function = [library newFunctionWithName:functionNameString];
        if (!function) {
            if (error) *error = MTL_ERROR_SHADER_COMPILATION;
            return NULL;
        }
        
        NSError* pipelineError = nil;
        id<MTLComputePipelineState> pipelineState = 
            [device->device newComputePipelineStateWithFunction:function error:&pipelineError];
        if (!pipelineState) {
            if (error) *error = MTL_ERROR_PIPELINE_CREATION;
            return NULL;
        }
        
        MTLComputePipeline* pipeline = (MTLComputePipeline*)malloc(sizeof(MTLComputePipeline));
        if (!pipeline) {
            if (error) *error = MTL_ERROR_PIPELINE_CREATION;
            return NULL;
        }
        
        pipeline->pipelineState = pipelineState;
        pipeline->function = function;
        
        if (error) *error = MTL_SUCCESS;
        return pipeline;
    }
}

// Pipeline creation from metallib
MTLComputePipeline* mtl_compute_pipeline_create_from_metallib(
    MTLComputeDevice* device,
    const char* metallib_path,
    const char* function_name,
    MTLComputeError* error,
    char* error_log,
    size_t error_log_size
) {
    if (!device || !metallib_path || !function_name) {
        if (error) *error = MTL_ERROR_INVALID_PARAMETER;
        mtl_copy_error_log("Invalid parameter", error_log, error_log_size);
        return NULL;
    }
    
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSURL* url = [NSURL fileURLWithPath:path];
        
        NSError* libraryError = nil;
        id<MTLLibrary> library = [device->device newLibraryWithURL:url error:&libraryError];
        if (!library) {
            const char* errMsg = libraryError ? [[libraryError localizedDescription] UTF8String] : "Failed to load metallib";
            MTL_LOG("Failed to load metallib: %s", errMsg);
            mtl_copy_error_log(errMsg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_IO;
            return NULL;
        }
        
        NSString* functionNameString = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> function = [library newFunctionWithName:functionNameString];
        if (!function) {
            char msg[256];
            snprintf(msg, sizeof(msg), "Function '%s' not found in metallib", function_name);
            mtl_copy_error_log(msg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_SHADER_COMPILATION;
            return NULL;
        }
        
        NSError* pipelineError = nil;
        id<MTLComputePipelineState> pipelineState = 
            [device->device newComputePipelineStateWithFunction:function error:&pipelineError];
        if (!pipelineState) {
            const char* errMsg = [[pipelineError localizedDescription] UTF8String];
            mtl_copy_error_log(errMsg, error_log, error_log_size);
            if (error) *error = MTL_ERROR_PIPELINE_CREATION;
            return NULL;
        }
        
        MTLComputePipeline* pipeline = (MTLComputePipeline*)malloc(sizeof(MTLComputePipeline));
        if (!pipeline) {
            if (error) *error = MTL_ERROR_PIPELINE_CREATION;
            return NULL;
        }
        
        pipeline->pipelineState = pipelineState;
        pipeline->function = function;
        
        if (error) *error = MTL_SUCCESS;
        return pipeline;
    }
}

// Pipeline labels (no-op, kept for API consistency)
void mtl_compute_pipeline_set_label(MTLComputePipeline* pipeline, const char* label) {
    // Pipeline labels are set at creation time in Metal
    (void)pipeline;
    (void)label;
}

// Pipeline reflection (MSL 5.2.1)
MTLComputeError mtl_compute_pipeline_get_resource_info(
    MTLComputePipeline* pipeline,
    MTLComputeResourceInfo* resource_info
) {
    if (!pipeline || !resource_info) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        // Initialize
        memset(resource_info, 0, sizeof(MTLComputeResourceInfo));
        
        // Use heuristics from pipeline state
        // Metal's reflection API is complex and varies by version
        // Provide reasonable defaults based on common usage patterns
        
        // Most compute kernels use:
        resource_info->buffer_count = 8;    // 0-7 for general compute
        resource_info->texture_count = 4;   // 0-3 for image processing
        resource_info->sampler_count = 2;   // 0-1 typically
        
        // Threadgroup memory estimate
        resource_info->threadgroup_memory_length = (uint32_t)[pipeline->pipelineState threadExecutionWidth] * 4;
        
        // Note: For precise reflection, users should compile with specific binding indices
        // and document their kernel interfaces
        
        return MTL_SUCCESS;
    }
}

// ============================================================================
// PIPELINE LIBRARY
// ============================================================================

MTLComputePipelineLibrary* mtl_compute_pipeline_library_create(
    MTLComputeDevice* device,
    const char* descriptor_path
) {
    if (!device) {
        return NULL;
    }
    
    (void)descriptor_path; // Unused for now
    
    @autoreleasepool {
        MTLComputePipelineLibrary* library = (MTLComputePipelineLibrary*)malloc(sizeof(MTLComputePipelineLibrary));
        if (!library) {
            return NULL;
        }
        
        library->pipelines = [[NSMutableDictionary alloc] init];
        return library;
    }
}

MTLComputeError mtl_compute_pipeline_library_add(
    MTLComputePipelineLibrary* library,
    const char* name,
    MTLComputePipeline* pipeline
) {
    if (!library || !name || !pipeline) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        NSString* key = [NSString stringWithUTF8String:name];
        NSValue* value = [NSValue valueWithPointer:pipeline];
        library->pipelines[key] = value;
        return MTL_SUCCESS;
    }
}

MTLComputePipeline* mtl_compute_pipeline_library_get(
    MTLComputePipelineLibrary* library,
    const char* name
) {
    if (!library || !name) {
        return NULL;
    }
    
    @autoreleasepool {
        NSString* key = [NSString stringWithUTF8String:name];
        NSValue* value = library->pipelines[key];
        if (!value) {
            return NULL;
        }
        return (MTLComputePipeline*)[value pointerValue];
    }
}

void mtl_compute_pipeline_library_destroy(MTLComputePipelineLibrary* library) {
    if (library) {
        free(library);
    }
}

