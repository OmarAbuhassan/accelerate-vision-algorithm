#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

// Error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// =================================================================================================
// DEVICE HELPER FUNCTIONS
// =================================================================================================

__device__ void swap(unsigned char &a, unsigned char &b) {
    unsigned char temp = a;
    a = b;
    b = temp;
}

// Bubble sort for median filter (fixed size 9)
__device__ void bubble_sort(unsigned char* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// =================================================================================================
// KERNELS
// =================================================================================================

__global__ void average_blur_kernel(const unsigned char* src, unsigned char* dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        float sum = 0.0f;
        for (int a = -1; a <= 1; ++a) {
            for (int b = -1; b <= 1; ++b) {
                int r_idx = y + a;
                int c_idx = x + b;
                if (r_idx >= 0 && r_idx < rows && c_idx >= 0 && c_idx < cols) {
                    sum += src[r_idx * cols + c_idx];
                }
            }
        }
        dst[y * cols + x] = static_cast<unsigned char>(sum / 9.0f);
    }
}

__global__ void difference_kernel(const unsigned char* img1, const unsigned char* img2, unsigned char* dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int idx = y * cols + x;
        int val = abs(static_cast<int>(img1[idx]) - static_cast<int>(img2[idx]));
        dst[idx] = static_cast<unsigned char>(val);
    }
}

__global__ void threshold_kernel(unsigned char* img, int rows, int cols, int limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = rows * cols;

    if (idx < total_pixels) {
        if (img[idx] <= limit) {
            img[idx] = 0;
        } else {
            img[idx] = 255;
        }
    }
}

__global__ void median_filter_kernel(const unsigned char* src, unsigned char* dst, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        unsigned char window[9];
        int count = 0;

        for (int a = -1; a <= 1; ++a) {
            for (int b = -1; b <= 1; ++b) {
                int r_idx = y + a;
                int c_idx = x + b;
                unsigned char val = 0;
                if (r_idx >= 0 && r_idx < rows && c_idx >= 0 && c_idx < cols) {
                    val = src[r_idx * cols + c_idx];
                }
                window[count++] = val;
            }
        }
        bubble_sort(window, 9);
        dst[y * cols + x] = window[4];
    }
}

__global__ void gaussian_3x3_kernel(const unsigned char* src, unsigned char* dst, int rows, int cols, const int* kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int acc = 0;
        int k_idx = 0;
        for (int a = -1; a <= 1; ++a) {
            for (int b = -1; b <= 1; ++b) {
                int r_idx = y + a;
                int c_idx = x + b;
                if (r_idx >= 0 && r_idx < rows && c_idx >= 0 && c_idx < cols) {
                    acc += static_cast<int>(src[r_idx * cols + c_idx]) * kernel[k_idx];
                }
                k_idx++;
            }
        }
        dst[y * cols + x] = static_cast<unsigned char>(acc / 16);
    }
}

__global__ void gaussian_5x5_kernel(const unsigned char* src, unsigned char* dst, int rows, int cols, const int* kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int acc = 0;
        int k_idx = 0;
        for (int a = -2; a <= 2; ++a) {
            for (int b = -2; b <= 2; ++b) {
                int r_idx = y + a;
                int c_idx = x + b;
                if (r_idx >= 0 && r_idx < rows && c_idx >= 0 && c_idx < cols) {
                    acc += static_cast<int>(src[r_idx * cols + c_idx]) * kernel[k_idx];
                }
                k_idx++;
            }
        }
        dst[y * cols + x] = static_cast<unsigned char>(acc / 273);
    }
}

__global__ void gaussian_25x25_kernel(const unsigned char* src, unsigned char* dst, int rows, int cols, const float* kernel, float kernel_sum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        float acc = 0.0f;
        int half = 12; // 25 / 2
        int bigSize = 25;

        for (int a = -half; a <= half; ++a) {
            for (int b = -half; b <= half; ++b) {
                int r_idx = y + a;
                int c_idx = x + b;
                if (r_idx >= 0 && r_idx < rows && c_idx >= 0 && c_idx < cols) {
                    int k_idx = (a + half) * bigSize + (b + half);
                    acc += static_cast<float>(src[r_idx * cols + c_idx]) * kernel[k_idx];
                }
            }
        }
        dst[y * cols + x] = static_cast<unsigned char>(acc / kernel_sum);
    }
}

// =================================================================================================
// MAIN
// =================================================================================================

int main() {
    // 1. Load Images
    std::cout << "Loading images..." << std::endl;
    cv::Mat img_host = cv::imread("../data/mounten.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat pepper_host = cv::imread("../data/sun.jpg", cv::IMREAD_GRAYSCALE);

    if (img_host.empty() || pepper_host.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        return -1;
    }

    // Resize pepper if needed
    if (img_host.size() != pepper_host.size()) {
        std::cout << "Resizing pepper image..." << std::endl;
        cv::resize(pepper_host, pepper_host, img_host.size());
    }

    int rows = img_host.rows;
    int cols = img_host.cols;
    size_t img_size = rows * cols * sizeof(unsigned char);

    // 2. Prepare Kernels on Host
    int h_gauss3x3[] = {1,2,1, 2,4,2, 1,2,1};
    int h_gauss5x5[] = {
        1,4,7,4,1, 4,16,26,16,4, 7,26,41,26,7, 4,16,26,16,4, 1,4,7,4,1
    };

    // Prepare 25x25 Kernel
    int bigSize = 25;
    float sigma = 3.0f;
    std::vector<float> h_gauss25x25(bigSize * bigSize);
    float k_sum = 0.0f;
    int half = bigSize / 2;
    for (int y = 0; y < bigSize; ++y) {
        for (int x = 0; x < bigSize; ++x) {
            int cX = y - half;
            int cY = half - x;
            float val = expf(-(cX*cX + cY*cY) / (2.0f * sigma * sigma));
            h_gauss25x25[y * bigSize + x] = val;
            k_sum += val;
        }
    }

    // 3. Allocate Device Memory
    std::cout << "Allocating GPU memory..." << std::endl;
    unsigned char *d_img, *d_pepper;
    unsigned char *d_blur, *d_diff, *d_med, *d_g1, *d_g2, *d_g3;
    int *d_k3, *d_k5;
    float *d_k25;

    CHECK_CUDA(cudaMalloc(&d_img, img_size));
    CHECK_CUDA(cudaMalloc(&d_pepper, img_size));
    CHECK_CUDA(cudaMalloc(&d_blur, img_size));
    CHECK_CUDA(cudaMalloc(&d_diff, img_size));
    CHECK_CUDA(cudaMalloc(&d_med, img_size));
    CHECK_CUDA(cudaMalloc(&d_g1, img_size));
    CHECK_CUDA(cudaMalloc(&d_g2, img_size));
    CHECK_CUDA(cudaMalloc(&d_g3, img_size));

    CHECK_CUDA(cudaMalloc(&d_k3, sizeof(h_gauss3x3)));
    CHECK_CUDA(cudaMalloc(&d_k5, sizeof(h_gauss5x5)));
    CHECK_CUDA(cudaMalloc(&d_k25, h_gauss25x25.size() * sizeof(float)));

    // 4. Copy Data Host -> Device
    CHECK_CUDA(cudaMemcpy(d_img, img_host.data, img_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_pepper, pepper_host.data, img_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k3, h_gauss3x3, sizeof(h_gauss3x3), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k5, h_gauss5x5, sizeof(h_gauss5x5), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k25, h_gauss25x25.data(), h_gauss25x25.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 5. Configure Grid
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    int threads_1d = 256;
    int blocks_1d = (rows * cols + threads_1d - 1) / threads_1d;

    std::cout << "Starting Pipeline (15 Iterations)..." << std::endl;
    std::cout << "Grid: " << grid.x << "x" << grid.y << ", Block: " << block.x << "x" << block.y << std::endl;

    for (int i = 0; i < 15; ++i) {
        std::string iter_label = "iteration_" + std::to_string(i);
        nvtxRangePushA(iter_label.c_str());

        nvtxRangePushA("average_blur");
        average_blur_kernel<<<grid, block>>>(d_img, d_blur, rows, cols);
        nvtxRangePop();

        nvtxRangePushA("difference_compute");
        difference_kernel<<<grid, block>>>(d_img, d_blur, d_diff, rows, cols);
        nvtxRangePop();

        nvtxRangePushA("thresholding");
        threshold_kernel<<<blocks_1d, threads_1d>>>(d_diff, rows, cols, 40);
        nvtxRangePop();

        nvtxRangePushA("median_filter");
        median_filter_kernel<<<grid, block>>>(d_pepper, d_med, rows, cols);
        nvtxRangePop();

        nvtxRangePushA("gaussian_3x3");
        gaussian_3x3_kernel<<<grid, block>>>(d_img, d_g1, rows, cols, d_k3);
        nvtxRangePop();

        nvtxRangePushA("gaussian_5x5");
        gaussian_5x5_kernel<<<grid, block>>>(d_img, d_g2, rows, cols, d_k5);
        nvtxRangePop();

        nvtxRangePushA("gaussian_25x25");
        gaussian_25x25_kernel<<<grid, block>>>(d_img, d_g3, rows, cols, d_k25, k_sum);
        nvtxRangePop();

        // Sync for profiling accuracy within iteration
        cudaDeviceSynchronize();
        nvtxRangePop(); // End Iteration
    }

    // 6. Cleanup
    CHECK_CUDA(cudaFree(d_img));
    CHECK_CUDA(cudaFree(d_pepper));
    CHECK_CUDA(cudaFree(d_blur));
    CHECK_CUDA(cudaFree(d_diff));
    CHECK_CUDA(cudaFree(d_med));
    CHECK_CUDA(cudaFree(d_g1));
    CHECK_CUDA(cudaFree(d_g2));
    CHECK_CUDA(cudaFree(d_g3));
    CHECK_CUDA(cudaFree(d_k3));
    CHECK_CUDA(cudaFree(d_k5));
    CHECK_CUDA(cudaFree(d_k25));

    std::cout << "Done." << std::endl;
    return 0;
}