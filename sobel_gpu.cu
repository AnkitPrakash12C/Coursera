#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <string>

// Error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// Sobel Kernel for GPU
// Uses global memory. For optimization, this could be upgraded to Shared Memory.
__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check (ignore 1-pixel border to avoid halo complexity for this demo)
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        float sumX = 0.0f;
        float sumY = 0.0f;

        // 3x3 Convolution
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixel = input[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int magnitude = (int)sqrtf(sumX * sumX + sumY * sumY);
        output[y * width + x] = (unsigned char)(magnitude > 255 ? 255 : magnitude);
    }
}

// CPU Reference Implementation
void sobelCPU(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height) {
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sumX = 0.0f;
            float sumY = 0.0f;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = input[(y + i) * width + (x + j)];
                    sumX += pixel * Gx[i + 1][j + 1];
                    sumY += pixel * Gy[i + 1][j + 1];
                }
            }
            int magnitude = (int)sqrt(sumX * sumX + sumY * sumY);
            output[y * width + x] = (unsigned char)(magnitude > 255 ? 255 : magnitude);
        }
    }
}

// Helper to parse CLI arguments
void parseArguments(int argc, char** argv, int& width, int& height) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-w" && i + 1 < argc) width = std::stoi(argv[++i]);
        if (arg == "-h" && i + 1 < argc) height = std::stoi(argv[++i]);
    }
}

int main(int argc, char** argv) {
    // Default size
    int width = 2048;
    int height = 2048;

    parseArguments(argc, argv, width, height);
    size_t imageSize = width * height * sizeof(unsigned char);

    std::cout << "Processing Image: " << width << " x " << height << std::endl;

    // Host allocation
    std::vector<unsigned char> h_input(width * height);
    std::vector<unsigned char> h_outputCPU(width * height, 0);
    std::vector<unsigned char> h_outputGPU(width * height, 0);

    // Initialize with random noise/gradients
    for (int i = 0; i < width * height; i++) {
        h_input[i] = rand() % 256;
    }

    // --- CPU Execution ---
    auto startCPU = std::chrono::high_resolution_clock::now();
    sobelCPU(h_input, h_outputCPU, width, height);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> durationCPU = endCPU - startCPU;
    std::cout << "CPU Time: " << durationCPU.count() << " ms" << std::endl;

    // --- GPU Execution ---
    unsigned char *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, imageSize));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), imageSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_output, 0, imageSize));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Warmup
    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float durationGPU = 0;
    cudaEventElapsedTime(&durationGPU, start, stop);
    std::cout << "GPU Time: " << durationGPU << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_outputGPU.data(), d_output, imageSize, cudaMemcpyDeviceToHost));

    // Speedup calculation
    std::cout << "Speedup: " << durationCPU.count() / durationGPU << "x" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}