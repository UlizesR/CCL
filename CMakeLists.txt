cmake_minimum_required(VERSION 3.15)
project(CCL VERSION 1.0.0 LANGUAGES C OBJC)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required frameworks
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)

# Library source files
set(CCL_SOURCES
    src/ccl_metal.m
)

# Create static library
add_library(ccl STATIC ${CCL_SOURCES})

target_include_directories(ccl PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ccl
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
)

# Enable ARC for Objective-C files
set_source_files_properties(src/ccl_metal.m PROPERTIES
    COMPILE_FLAGS "-fobjc-arc"
)

# Example: vec_add_metal
add_executable(vec_add_metal examples/vec_add_metal.c)
target_link_libraries(vec_add_metal ccl
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
)

# Example: matrix_mult_metal
add_executable(matrix_mult_metal examples/matrix_mult_metal.c)
target_link_libraries(matrix_mult_metal ccl
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
)

# Example: device_info_metal
add_executable(device_info_metal examples/device_info_metal.c)
target_link_libraries(device_info_metal ccl
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
)

# Example: feature_test_metal (comprehensive feature test)
add_executable(feature_test_metal examples/feature_test_metal.c)
target_link_libraries(feature_test_metal ccl
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${COREFOUNDATION_FRAMEWORK}
)

# Installation
install(TARGETS ccl
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/ccl.h
    DESTINATION include
)

