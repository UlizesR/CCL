/**
 * Function Table Demo - MSL 2.15, 5.1.4, 5.1.5
 * Demonstrates visible functions and function pointers
 */

#include <metal_stdlib>
using namespace metal;

// Visible functions that can be put in function tables
[[visible]]
float operation_add(float a, float b) {
    return a + b;
}

[[visible]]
float operation_multiply(float a, float b) {
    return a * b;
}

[[visible]]
float operation_max(float a, float b) {
    return max(a, b);
}

// Main kernel that uses function table
kernel void compute_with_function_table(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float (*operations[3])(float, float) [[buffer(2)]],  // Function table
    constant uint& operation_index [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float a = input[gid * 2];
    float b = input[gid * 2 + 1];
    
    // Select operation from function table
    auto op = operations[operation_index];
    output[gid] = op(a, b);
}

