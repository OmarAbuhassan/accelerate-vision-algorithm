#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include "nvToolsExt.h"
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// ------------------------ CUDA Kernels ------------------------

// Histogram kernel using shared memory (CUDA-specific optimization)
__global__ void histogram_kernel(unsigned char* img_data, int* arr, int rows, int cols) {
    // Shared memory for block-local histogram (CUDA-specific)
    __shared__ int local_hist[256];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory (CUDA-specific)
    if (tid < 256) {
        local_hist[tid] = 0;
    }
    __syncthreads();  // CUDA-specific barrier
    
    // Compute histogram in shared memory (faster atomics)
    if (i < rows && j < cols) {
        int index = static_cast<int>(img_data[i * cols + j]);
        atomicAdd(&local_hist[index], 1);  // Shared memory atomic (fast)
    }
    __syncthreads();  // CUDA-specific barrier
    
    // Reduce shared histogram to global memory (CUDA-specific optimization)
    if (tid < 256) {
        atomicAdd(&arr[tid], local_hist[tid]);  // Global memory atomic (only 256 per block)
    }
}

// Apply equalization kernel
__global__ void apply_equalization_kernel(unsigned char* img_data, unsigned char* out_data, float* arr3, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < rows && j < cols) {
        int idx = i * cols + j;
        out_data[idx] = static_cast<unsigned char>(floor(255.0f * arr3[img_data[idx]]));
    }
}

// Histogram equalized kernel using shared memory
__global__ void histogram_equalized_kernel(unsigned char* out_data, int* h2, int rows, int cols) {
    __shared__ int local_hist[256];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 256) {
        local_hist[tid] = 0;
    }
    __syncthreads();
    
    if (i < rows && j < cols) {
        int idx = i * cols + j;
        int index = static_cast<int>(out_data[idx]);
        atomicAdd(&local_hist[index], 1);
    }
    __syncthreads();
    
    if (tid < 256) {
        atomicAdd(&h2[tid], local_hist[tid]);
    }
}

// CUDA error checking macro (CUDA-specific)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while(0)

int main() {

    Mat image = imread("./input/trees_big.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Failed to load ./input/trees_big.jpg" << endl;
        return -1;
    }
    Mat myMat1;

    unsigned char* img_data = image.ptr<unsigned char>(0);
    int rows = image.rows;
    int cols = image.cols;
    int size = rows * cols;

    // CUDA-specific: Define block and grid dimensions
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, 
                 (rows + blockDim.y - 1) / blockDim.y);

    for (int iter = 0; iter < 15; iter++) {

        string iter_name = "iteration_" + to_string(iter);
        nvtxRangePushA(iter_name.c_str());

        int arr[256] = {0};
        float arr2[256] = {0};
        float arr3[256] = {0};
        int h2[256] = {0};

        // ------------------------ histogram_original ------------------------
        nvtxRangePushA("histogram_original");

        // CUDA-specific: Explicit device pointers
        unsigned char* d_img_data;
        int* d_arr;

        // CUDA-specific: cudaMalloc instead of enter data create
        CUDA_CHECK(cudaMalloc(&d_img_data, size * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_arr, 256 * sizeof(int)));

        // CUDA-specific: cudaMemcpy instead of enter data copyin
        CUDA_CHECK(cudaMemcpy(d_img_data, img_data, size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_arr, arr, 256 * sizeof(int), cudaMemcpyHostToDevice));

        // CUDA-specific: Kernel launch syntax <<<grid, block>>>
        histogram_kernel<<<gridDim, blockDim>>>(d_img_data, d_arr, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());  // CUDA-specific: Explicit sync

        // CUDA-specific: cudaMemcpy instead of exit data copyout
        CUDA_CHECK(cudaMemcpy(arr, d_arr, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_arr));  // CUDA-specific: Explicit free

        nvtxRangePop();

        // ------------------------ find_max_original_hist --------------------
        nvtxRangePushA("find_max_original_hist");

        int maxVal = 0;
        for (int i = 0; i < 256; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
            }
        }

        nvtxRangePop();

        // ------------------------ pmf_compute -------------------------------
        nvtxRangePushA("pmf_compute");

        float total = static_cast<float>(size);
        for (int i = 0; i < 256; i++) {
            arr2[i] = static_cast<float>(arr[i]) / total;
        }

        nvtxRangePop();

        // ------------------------ cdf_compute -------------------------------
        nvtxRangePushA("cdf_compute");

        arr3[0] = arr2[0];
        for (int i = 1; i < 256; i++) {
            arr3[i] = arr2[i] + arr3[i - 1];
        }

        nvtxRangePop();

        // ------------------------ apply_equalization -----------------------
        nvtxRangePushA("apply_equalization");

        myMat1 = Mat(rows, cols, CV_8U, Scalar(0));
        unsigned char* out_data = myMat1.ptr<unsigned char>(0);

        float* d_arr3;
        unsigned char* d_out_data;

        CUDA_CHECK(cudaMalloc(&d_arr3, 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_data, size * sizeof(unsigned char)));

        CUDA_CHECK(cudaMemcpy(d_arr3, arr3, 256 * sizeof(float), cudaMemcpyHostToDevice));

        apply_equalization_kernel<<<gridDim, blockDim>>>(d_img_data, d_out_data, d_arr3, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(out_data, d_out_data, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_arr3));

        nvtxRangePop();

        // ------------------------ histogram_equalized -----------------------
        nvtxRangePushA("histogram_equalized");

        int* d_h2;

        CUDA_CHECK(cudaMalloc(&d_h2, 256 * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_out_data, out_data, size * sizeof(unsigned char), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_h2, h2, 256 * sizeof(int), cudaMemcpyHostToDevice));

        histogram_equalized_kernel<<<gridDim, blockDim>>>(d_out_data, d_h2, rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h2, d_h2, 256 * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_h2));
        CUDA_CHECK(cudaFree(d_out_data));
        CUDA_CHECK(cudaFree(d_img_data));

        nvtxRangePop();

        // ------------------------ find_max_equalized_hist -------------------
        nvtxRangePushA("find_max_equalized_hist");

        int maxH2 = 0;
        for (int i = 0; i < 256; i++) {
            if (h2[i] > maxH2) {
                maxH2 = h2[i];
            }
        }

        nvtxRangePop();

        nvtxRangePop();
    }

    imwrite("./output/equalized_cuda.jpg", myMat1);
    cout << "Output saved to ./output/equalized_cuda.jpg" << endl;

    return 0;
}