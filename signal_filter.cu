/*
 * Copyright 2025. GPU Signal Processing Capstone.
 * * Description: A CUDA-accelerated implementation of a Moving Average Filter
 * for noise reduction on massive 1D signal arrays.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// CUDA Kernel: Moving Average Filter
// Each thread calculates the average for one specific data point
__global__ void movingAverageKernel(float* input, float* output, int n, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = window_size / 2;

    if (idx < n) {
        float sum = 0.0f;
        int count = 0;

        // Simple convolution loop
        for (int i = -radius; i <= radius; i++) {
            int neighbor_idx = idx + i;
            // Boundary check to handle edges of the array
            if (neighbor_idx >= 0 && neighbor_idx < n) {
                sum += input[neighbor_idx];
                count++;
            }
        }
        output[idx] = sum / count;
    }
}

// Helper to parse CLI arguments
void parseArguments(int argc, char** argv, int& n, int& window) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) n = std::stoi(argv[++i]);
        if (arg == "-w" && i + 1 < argc) window = std::stoi(argv[++i]);
    }
}

int main(int argc, char** argv) {
    // Default: Process 50 Million data points (Large Data Requirement)
    int num_elements = 50000000; 
    int window_size = 7;

    parseArguments(argc, argv, num_elements, window_size);

    size_t bytes = num_elements * sizeof(float);
    std::cout << "Processing Signal Data..." << std::endl;
    std::cout << "Elements: " << num_elements << " | Window Size: " << window_size << std::endl;

    // Host Allocation
    std::vector<float> h_input(num_elements);
    std::vector<float> h_output(num_elements);

    // Initialize with synthetic noisy signal data
    for (int i = 0; i < num_elements; i++) {
        h_input[i] = sinf(i * 0.01f) + (rand() % 100 / 100.0f); // Sine wave + Noise
    }

    // Device Allocation
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));

    // Copy H2D
    auto start_copy = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // Kernel Launch Configuration
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    std::cout << "Launching Kernel with " << blocks << " blocks of " << threads << " threads..." << std::endl;
    
    // Launch
    movingAverageKernel<<<blocks, threads>>>(d_input, d_output, num_elements, window_size);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end_kernel = std::chrono::high_resolution_clock::now();

    // Copy D2H
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    std::chrono::duration<float, std::milli> duration = end_kernel - start_copy;
    std::cout << "Processing Complete." << std::endl;
    std::cout << "Total GPU Time (Transfer + Compute): " << duration.count() << " ms" << std::endl;
    std::cout << "Sample Output: " << h_output[0] << ", " << h_output[1] << ", " << h_output[2] << "..." << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
