#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

using namespace cv;
using namespace std;

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// --------------------------------------------------------
// CUDA KERNELS
// --------------------------------------------------------

__global__ void kernel_combine_images_avg(const unsigned char* robot, const unsigned char* house, unsigned char* output, int cols, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        // Average
        output[idx] = (unsigned char)((robot[idx] + house[idx]) / 2);
    }
}

__global__ void kernel_add_constant_robot(const unsigned char* robot, unsigned char* output, int cols, int rows, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        int val = robot[idx] + k;
        if (val > 255) val = 255;
        output[idx] = (unsigned char)val;
    }
}

__global__ void kernel_build_gradient(unsigned char* output, int cols, int rows, float dx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        float r = x * dx;
        if (r > 255.0f) r = 255.0f;
        output[idx] = (unsigned char)r;
    }
}

__global__ void kernel_add_gradient_avg(const unsigned char* robot, const unsigned char* grad, unsigned char* output, int cols, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        output[idx] = (unsigned char)((robot[idx] + grad[idx]) / 2);
    }
}

__global__ void kernel_add_gradient_saturated(const unsigned char* robot, const unsigned char* grad, unsigned char* output, int cols, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        int val = robot[idx] + grad[idx];
        if (val > 255) val = 255;
        output[idx] = (unsigned char)val;
    }
}

__global__ void kernel_blend_gradient(const unsigned char* robot, const unsigned char* house, unsigned char* output, int cols, int rows, float inc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        // Calculate weights based on ROW index (y)
        float a = y * inc;
        if (a > 1.0f) a = 1.0f;
        float b = 1.0f - a;

        int idx = y * cols + x;
        float val = a * robot[idx] + b * house[idx];
        
        if (val > 255.0f) val = 255.0f;
        if (val < 0.0f) val = 0.0f;
        output[idx] = (unsigned char)val;
    }
}

// --------------------------------------------------------
// MAIN
// --------------------------------------------------------

int main() {
    // 1. Load Images (Host)
    Mat h_robot_mat = imread("robot.jpg", IMREAD_GRAYSCALE);
    Mat h_house_mat = imread("house.jpg", IMREAD_GRAYSCALE);

    if (h_robot_mat.empty() || h_house_mat.empty()) {
        cerr << "Error loading images." << endl;
        return -1;
    }

    // Determine dimensions (Common ROI)
    int rows = min(h_robot_mat.rows, h_house_mat.rows);
    int cols = min(h_robot_mat.cols, h_house_mat.cols);
    size_t dataSize = rows * cols * sizeof(unsigned char);

    // Create Host buffers for results
    Mat h_result(rows, cols, CV_8UC1);
    Mat h_cte(rows, cols, CV_8UC1);
    Mat h_grad(rows, cols, CV_8UC1);
    Mat h_result2(rows, cols, CV_8UC1);
    Mat h_result3(rows, cols, CV_8UC1);
    Mat h_result4(rows, cols, CV_8UC1);

    // 2. Allocate Device Memory
    unsigned char *d_robot, *d_house;
    unsigned char *d_result, *d_cte, *d_grad, *d_result2, *d_result3, *d_result4;

    CHECK_CUDA(cudaMalloc(&d_robot, dataSize));
    CHECK_CUDA(cudaMalloc(&d_house, dataSize));
    CHECK_CUDA(cudaMalloc(&d_result, dataSize));
    CHECK_CUDA(cudaMalloc(&d_cte, dataSize));
    CHECK_CUDA(cudaMalloc(&d_grad, dataSize));
    CHECK_CUDA(cudaMalloc(&d_result2, dataSize));
    CHECK_CUDA(cudaMalloc(&d_result3, dataSize));
    CHECK_CUDA(cudaMalloc(&d_result4, dataSize));

    // 3. Copy Inputs Host -> Device (Once)
    // We use cudaMemcpy2D to ensure that if the original image is wider than 'cols',
    // we strictly copy the relevant ROI without grabbing the wrong pixels from the next line.
    // Destination pitch = cols (packed), Source pitch = step (potentially padded).
    CHECK_CUDA(cudaMemcpy2D(d_robot, cols, h_robot_mat.data, h_robot_mat.step, cols, rows, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy2D(d_house, cols, h_house_mat.data, h_house_mat.step, cols, rows, cudaMemcpyHostToDevice));

    // Setup Grid/Block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    int k = 30;
    float dx = 255.0f / (float)cols;
    float inc = 1.0f / (float)rows;

    cout << "Starting CUDA processing (" << cols << "x" << rows << ")..." << endl;

    // 4. Processing Loop
    for (int iter = 0; iter < 15; ++iter) {
        string iter_name = "iteration_" + to_string(iter);
        nvtxRangePushA(iter_name.c_str());

        // --- Kernel 1: Combine Avg ---
        nvtxRangePushA("combine_images_avg");
        kernel_combine_images_avg<<<gridDim, blockDim>>>(d_robot, d_house, d_result, cols, rows);
        cudaDeviceSynchronize(); // Synchronize for accurate NVTX profiling
        nvtxRangePop();

        // --- Kernel 2: Add Constant ---
        nvtxRangePushA("add_constant_robot");
        kernel_add_constant_robot<<<gridDim, blockDim>>>(d_robot, d_cte, cols, rows, k);
        cudaDeviceSynchronize();
        nvtxRangePop();

        // --- Kernel 3: Build Gradient ---
        nvtxRangePushA("build_gradient");
        kernel_build_gradient<<<gridDim, blockDim>>>(d_grad, cols, rows, dx);
        cudaDeviceSynchronize();
        nvtxRangePop();

        // --- Kernel 4: Gradient Avg ---
        nvtxRangePushA("add_gradient_avg");
        kernel_add_gradient_avg<<<gridDim, blockDim>>>(d_robot, d_grad, d_result2, cols, rows);
        cudaDeviceSynchronize();
        nvtxRangePop();

        // --- Kernel 5: Gradient Saturated ---
        nvtxRangePushA("add_gradient_saturated");
        kernel_add_gradient_saturated<<<gridDim, blockDim>>>(d_robot, d_grad, d_result3, cols, rows);
        cudaDeviceSynchronize();
        nvtxRangePop();

        // --- Kernel 6: Blend ---
        nvtxRangePushA("blend_gradient_robot_house");
        kernel_blend_gradient<<<gridDim, blockDim>>>(d_robot, d_house, d_result4, cols, rows, inc);
        cudaDeviceSynchronize();
        nvtxRangePop();

        nvtxRangePop(); // End Iteration
    }

    // 5. Copy Results Device -> Host
    cout << "Copying results back to CPU..." << endl;
    
    // We can use standard linear copy here because our device buffers are packed exactly to cols * rows
    CHECK_CUDA(cudaMemcpy(h_result.data, d_result, dataSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cte.data, d_cte, dataSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_grad.data, d_grad, dataSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_result2.data, d_result2, dataSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_result3.data, d_result3, dataSize, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_result4.data, d_result4, dataSize, cudaMemcpyDeviceToHost));

    // 6. Save Images
    cout << "Saving images..." << endl;
    imwrite("cuda_out_1_avg.jpg", h_result);
    imwrite("cuda_out_2_const.jpg", h_cte);
    imwrite("cuda_out_3_grad.jpg", h_grad);
    imwrite("cuda_out_4_grad_avg.jpg", h_result2);
    imwrite("cuda_out_5_grad_sat.jpg", h_result3);
    imwrite("cuda_out_6_blend.jpg", h_result4);

    // 7. Cleanup
    cudaFree(d_robot); cudaFree(d_house);
    cudaFree(d_result); cudaFree(d_cte); cudaFree(d_grad);
    cudaFree(d_result2); cudaFree(d_result3); cudaFree(d_result4);

    cout << "Done." << endl;
    return 0;
}