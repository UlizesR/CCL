cmake_minimum_required(VERSION 3.20)
project(MTLComp 
    VERSION 2.0.0
    LANGUAGES C OBJC
    DESCRIPTION "Metal Compute Library - C/Objective-C wrapper for Metal compute shaders"
)

# Require macOS
if(NOT APPLE)
    message(FATAL_ERROR "MTLComp requires macOS or iOS")
endif()

# C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_OBJC_STANDARD 11)

# Build options
option(MTLCOMP_BUILD_EXAMPLES "Build example programs" ON)
option(MTLCOMP_BUILD_SHARED "Build shared library" OFF)
option(MTLCOMP_ENABLE_ARC "Enable Automatic Reference Counting" ON)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")

# Objective-C flags
set(CMAKE_OBJC_FLAGS "${CMAKE_OBJC_FLAGS} -Wall -Wextra")
if(MTLCOMP_ENABLE_ARC)
    set(CMAKE_OBJC_FLAGS "${CMAKE_OBJC_FLAGS} -fobjc-arc")
endif()

# Find required frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(IMAGEIO_FRAMEWORK ImageIO REQUIRED)
find_library(UNIFORMTYPEIDENTIFIERS_FRAMEWORK UniformTypeIdentifiers REQUIRED)

set(REQUIRED_FRAMEWORKS
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${COREGRAPHICS_FRAMEWORK}
    ${IMAGEIO_FRAMEWORK}
    ${UNIFORMTYPEIDENTIFIERS_FRAMEWORK}
)

# Library sources - modularized by subsystem
set(MTLCOMP_SOURCES
    src/mtl_logging.m           # Centralized logging system
    src/mtl_device.m            # Device management & capabilities  
    src/mtl_pipeline.m          # Pipeline creation, compilation & library
    src/mtl_resource.m          # Buffers, samplers, heaps
    src/mtl_dispatch.m          # Unified dispatch & encoding (Tiers 1-3)
    src/mtl_dispatch_advanced.m # Advanced dispatch (async, profiled, indirect)
    src/mtl_argbuf.m            # Argument buffers & function tables
    src/mtl_pass.m              # Compute passes (high-level batching)
    src/mtl_tensor.m            # Tensor utilities & auto-tuning
    src/mtl_texture.m           # Texture operations
)

# Internal headers (not installed)
set(MTLCOMP_INTERNAL_HEADERS
    src/mtl_internal.h      # Shared internal definitions
)

set(MTLCOMP_HEADERS
    src/mtl_compute_core.h
    src/mtl_compute.h
    src/mtl_texture.h
    src/mtl_compute_advanced.h
)

# Build library
if(MTLCOMP_BUILD_SHARED)
    add_library(mtlcompute SHARED ${MTLCOMP_SOURCES})
else()
    add_library(mtlcompute STATIC ${MTLCOMP_SOURCES})
endif()

target_include_directories(mtlcompute
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(mtlcompute
    PUBLIC
        ${REQUIRED_FRAMEWORKS}
)

set_target_properties(mtlcompute PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${MTLCOMP_HEADERS}"
    FRAMEWORK FALSE
)

# Add alias for consistent naming
add_library(MTLComp::mtlcompute ALIAS mtlcompute)

# Examples
if(MTLCOMP_BUILD_EXAMPLES)
    # Image effects example
    add_executable(example_image_effects
        examples/example_image_effects.c
    )
    target_link_libraries(example_image_effects
        PRIVATE mtlcompute
    )
    
    # Feature test harness
    add_executable(example_feature_test
        examples/example_feature_test.c
    )
    target_link_libraries(example_feature_test
        PRIVATE mtlcompute
    )
    
    # Standard kernels demo
    add_executable(example_standard_kernels
        examples/example_standard_kernels.c
    )
    target_link_libraries(example_standard_kernels
        PRIVATE mtlcompute
    )
    
    # Set output directory for examples
    set_target_properties(example_image_effects example_feature_test example_standard_kernels
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

# Installation
include(GNUInstallDirs)

install(TARGETS mtlcompute
    EXPORT MTLCompTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mtlcomp
)

# Install shader samples
install(DIRECTORY samples/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/mtlcomp/shaders
    FILES_MATCHING PATTERN "*.metal"
)

# Export targets
install(EXPORT MTLCompTargets
    FILE MTLCompTargets.cmake
    NAMESPACE MTLComp::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MTLComp
)

# Create config file
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MTLCompConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/MTLCompConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MTLComp
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/MTLCompConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/MTLCompConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/MTLCompConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MTLComp
)

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )
    
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()

# Testing
enable_testing()
if(MTLCOMP_BUILD_EXAMPLES)
    add_test(NAME FeatureTest COMMAND example_feature_test)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "MTLComp Configuration Summary")
message(STATUS "=============================")
message(STATUS "Version:           ${PROJECT_VERSION}")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Library type:      ${MTLCOMP_BUILD_SHARED}")
message(STATUS "ARC enabled:       ${MTLCOMP_ENABLE_ARC}")
message(STATUS "Build examples:    ${MTLCOMP_BUILD_EXAMPLES}")
message(STATUS "Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")

