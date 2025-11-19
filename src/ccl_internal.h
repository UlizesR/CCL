// src/ccl_internal.h
#pragma once

#include "../include/ccl.h"

typedef enum ccl_backend_kind {
    CCL_BACKEND_KIND_METAL,
    CCL_BACKEND_KIND_GL_COMPUTE,
    CCL_BACKEND_KIND_OPENCL
} ccl_backend_kind;

struct ccl_context {
    ccl_backend_kind kind;
    void *impl;   // backend-specific object
};

struct ccl_buffer {
    ccl_backend_kind kind;
    void *impl;
    size_t size;
};

struct ccl_kernel {
    ccl_backend_kind kind;
    void *impl;
};

typedef struct ccl_fence ccl_fence;

struct ccl_fence {
    ccl_backend_kind kind;
    void *impl;
};

struct ccl_function_table {
    ccl_backend_kind kind;
    void *impl;
};

struct ccl_binary_archive {
    ccl_backend_kind kind;
    void *impl;
};

struct ccl_acceleration_structure {
    ccl_backend_kind kind;
    void *impl;
};

struct ccl_raytracing_pipeline {
    ccl_backend_kind kind;
    void *impl;
};

struct ccl_indirect_command_buffer {
    ccl_backend_kind kind;
    void *impl;
};

