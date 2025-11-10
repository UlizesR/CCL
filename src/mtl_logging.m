/**
 * MTL Logging & Error Handling
 * Centralized logging system for all modules
 */

#include "mtl_internal.h"
#include <stdio.h>

// Default logger: print to stderr
static void default_logger(const char* message) {
    fprintf(stderr, "%s\n", message);
}

// Global logger
mtl_log_fn g_mtl_log = default_logger;

void mtl_set_logger(mtl_log_fn fn) {
    g_mtl_log = fn ? fn : default_logger;
}

