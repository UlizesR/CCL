/**
 * MTL Compute Pass
 * High-level abstraction for batched compute execution
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "mtl_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct MTLComputePass {
    MTLComputeDevice* device;
    MTLComputeDispatchDesc* dispatches;
    size_t dispatch_count;
    size_t max_dispatches;
    char label[64];
};

MTLComputePass* mtl_compute_pass_create(MTLComputeDevice* device, size_t max_dispatches) {
    if (!device || max_dispatches == 0) {
        return NULL;
    }
    
    MTLComputePass* pass = (MTLComputePass*)malloc(sizeof(MTLComputePass));
    if (!pass) {
        return NULL;
    }
    
    pass->dispatches = (MTLComputeDispatchDesc*)calloc(max_dispatches, sizeof(MTLComputeDispatchDesc));
    if (!pass->dispatches) {
        free(pass);
        return NULL;
    }
    
    pass->device = device;
    pass->dispatch_count = 0;
    pass->max_dispatches = max_dispatches;
    pass->label[0] = '\0';
    
    return pass;
}

MTLComputeError mtl_compute_pass_add_dispatch(
    MTLComputePass* pass,
    const MTLComputeDispatchDesc* desc
) {
    if (!pass || !desc) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    if (pass->dispatch_count >= pass->max_dispatches) {
        MTL_LOG("Pass is full (max %zu dispatches)", pass->max_dispatches);
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    // Deep copy the descriptor
    pass->dispatches[pass->dispatch_count] = *desc;
    pass->dispatch_count++;
    
    return MTL_SUCCESS;
}

MTLComputeError mtl_compute_pass_execute(MTLComputePass* pass) {
    if (!pass || pass->dispatch_count == 0) {
        return MTL_ERROR_INVALID_PARAMETER;
    }
    
    @autoreleasepool {
        // Create command list for batch
        MTLComputeCommandList* cmd_list = NULL;
        MTLComputeError err = mtl_compute_begin(pass->device, &cmd_list);
        if (err != MTL_SUCCESS) {
            return err;
        }
        
        // Set label if provided
        if (pass->label[0] != '\0') {
            mtl_compute_command_list_set_label(cmd_list, pass->label);
        }
        
        // Encode all dispatches
        for (size_t i = 0; i < pass->dispatch_count; i++) {
            err = mtl_compute_encode_dispatch(cmd_list, &pass->dispatches[i]);
            if (err != MTL_SUCCESS) {
                MTL_LOG("Pass dispatch %zu failed: %s", i, mtl_compute_error_string(err));
                free(cmd_list);
                return err;
            }
        }
        
        // Submit all at once
        return mtl_compute_end_submit(cmd_list);
    }
}

void mtl_compute_pass_clear(MTLComputePass* pass) {
    if (pass) {
        pass->dispatch_count = 0;
    }
}

void mtl_compute_pass_destroy(MTLComputePass* pass) {
    if (pass) {
        if (pass->dispatches) {
            free(pass->dispatches);
        }
        free(pass);
    }
}

