#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "nvToolsExt.h"

// OpenACC header (optional, but good for explicit runtime calls)
#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;
using namespace cv;

// Helper function to swap elements (for Median Filter on GPU)
#pragma acc routine seq
void swap_uchar(unsigned char &a, unsigned char &b) {
    unsigned char t = a;
    a = b;
    b = t;
}

// Simple Bubble Sort for small fixed arrays on GPU
#pragma acc routine seq
void bubble_sort_9(unsigned char* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap_uchar(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    // 1. Load Data on Host
    Mat image = imread("../data/mounten.jpg", IMREAD_GRAYSCALE);
    Mat pepperNoise = imread("../data/sun.jpg", IMREAD_GRAYSCALE);

    if (image.empty() || pepperNoise.empty()) {
        cerr << "Failed to load images." << endl;
        return -1;
    }

    // ---------------------------------------------------------
    // CRITICAL FIX: Ensure both images are exactly the same size
    // ---------------------------------------------------------
    if (image.size() != pepperNoise.size()) {
        cout << "Resizing pepperNoise to match image dimensions..." << endl;
        cv::resize(pepperNoise, pepperNoise, image.size());
    }

    // Ensure images are continuous in memory for raw pointer access
    if (!image.isContinuous()) image = image.clone();
    if (!pepperNoise.isContinuous()) pepperNoise = pepperNoise.clone();

    int rows = image.rows;
    int cols = image.cols;
    int imgSize = rows * cols;

    // 2. Allocate Output Memory on Host (Containers only)
    Mat blur(rows, cols, CV_8UC1);
    Mat diferencia(rows, cols, CV_8UC1);
    Mat filtroMediana(rows, cols, CV_8UC1);
    Mat gaussiano(rows, cols, CV_8UC1);
    Mat gaussiano2(rows, cols, CV_8UC1);
    Mat gaussiano3(rows, cols, CV_8UC1);

    // Get Raw Pointers
    unsigned char* ptr_src = image.ptr<uchar>();
    unsigned char* ptr_pepper = pepperNoise.ptr<uchar>();
    unsigned char* ptr_blur = blur.ptr<uchar>();
    unsigned char* ptr_diff = diferencia.ptr<uchar>();
    unsigned char* ptr_med = filtroMediana.ptr<uchar>();
    unsigned char* ptr_gauss1 = gaussiano.ptr<uchar>();
    unsigned char* ptr_gauss2 = gaussiano2.ptr<uchar>();
    unsigned char* ptr_gauss3 = gaussiano3.ptr<uchar>();

    // Constants
    const int thresholdLimit = 40;
    
    // Gaussian Kernels (Flattened for easier GPU mapping)
    const int gaussMatrixFlat[9] = { 1,2,1, 2,4,2, 1,2,1 };
    const int gaussMatrix2Flat[25] = { 
        1,4,7,4,1, 
        4,16,26,16,4, 
        7,26,41,26,7, 
        4,16,26,16,4, 
        1,4,7,4,1 
    };

    // Pre-calculate the large 25x25 kernel on Host to save GPU cycles
    // (Calculating constants inside a kernel loop is inefficient)
    const int bigSize = 25;
    const int bigKernLen = bigSize * bigSize;
    vector<float> h_bigKernel(bigKernLen);
    float bigKernelSum = 0.0f;
    {
        int x0 = bigSize / 2;
        int y0 = bigSize / 2;
        const int sigma = 3;
        const float pi = 3.1415926535f;
        float denom = 2.0f * sigma * sigma;
        float normConst = 1.0f / (sigma * sigma * 2.0f * pi);
        
        for (int y = 0; y < bigSize; ++y) {
            for (int x = 0; x < bigSize; ++x) {
                int cX = y - y0;
                int cY = x0 - x;
                float r2 = float(cX * cX + cY * cY);
                float val = normConst * expf(-r2 / denom);
                h_bigKernel[y * bigSize + x] = val;
                bigKernelSum += val;
            }
        }
    }
    float* ptr_bigKernel = h_bigKernel.data();

    // =========================================================================
    // DATA STRATEGY: Enter Data Region
    // Copy Inputs (src, pepper, kernels) -> GPU
    // Create Outputs (buffers) -> GPU (Don't copy junk from CPU)
    // =========================================================================
    
    #pragma acc enter data copyin(ptr_src[0:imgSize], ptr_pepper[0:imgSize], \
                                  gaussMatrixFlat[0:9], gaussMatrix2Flat[0:25], \
                                  ptr_bigKernel[0:bigKernLen]) \
                           create(ptr_blur[0:imgSize], ptr_diff[0:imgSize], \
                                  ptr_med[0:imgSize], ptr_gauss1[0:imgSize], \
                                  ptr_gauss2[0:imgSize], ptr_gauss3[0:imgSize])

    for (int iter = 0; iter < 15; ++iter) {
        string iter_name = string("iteration_") + to_string(iter);
        nvtxRangePushA(iter_name.c_str());

        // ---------------- average_blur ----------------
        nvtxRangePushA("average_blur");
        #pragma acc parallel loop collapse(2) present(ptr_src, ptr_blur)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float sum = 0.0f;
                for (int a = -1; a <= 1; ++a) {
                    for (int b = -1; b <= 1; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < rows && xx >= 0 && xx < cols) {
                            sum += ptr_src[yy * cols + xx];
                        }
                    }
                }
                // Original code divided by 9 regardless of boundary, keeping original logic:
                ptr_blur[i * cols + j] = static_cast<uchar>(sum / 9.0f);
            }
        }
        nvtxRangePop();

        // ---------------- difference_compute ----------------
        nvtxRangePushA("difference_compute");
        #pragma acc parallel loop collapse(2) present(ptr_src, ptr_blur, ptr_diff)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int idx = i * cols + j;
                int val = abs(int(ptr_src[idx]) - int(ptr_blur[idx]));
                ptr_diff[idx] = static_cast<uchar>(val);
            }
        }
        nvtxRangePop();

        // ---------------- thresholding ----------------
        nvtxRangePushA("thresholding");
        #pragma acc parallel loop present(ptr_diff)
        for (int k = 0; k < imgSize; ++k) {
            ptr_diff[k] = (ptr_diff[k] <= thresholdLimit) ? 0 : 255;
        }
        nvtxRangePop();

        // ---------------- median_filter ----------------
        nvtxRangePushA("median_filter");
        #pragma acc parallel loop collapse(2) present(ptr_pepper, ptr_med)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                unsigned char window[9]; // Private array per thread
                int count = 0;
                
                for (int a = -1; a <= 1; ++a) {
                    for (int b = -1; b <= 1; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < rows && xx >= 0 && xx < cols) {
                            window[count++] = ptr_pepper[yy * cols + xx];
                        } else {
                            // Pad with 0 for edge cases or handle logically
                            // For simplicity, we fill with 0 to keep sort valid
                            window[count++] = 0; 
                        }
                    }
                }
                bubble_sort_9(window, 9);
                ptr_med[i * cols + j] = window[4]; // Middle element
            }
        }
        nvtxRangePop();

        // ---------------- gaussian_blur_3x3 ----------------
        nvtxRangePushA("gaussian_blur_3x3");
        #pragma acc parallel loop collapse(2) present(ptr_src, ptr_gauss1, gaussMatrixFlat)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int acc = 0;
                int k_idx = 0;
                for (int a = -1; a <= 1; ++a) {
                    for (int b = -1; b <= 1; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < rows && xx >= 0 && xx < cols) {
                            acc += int(ptr_src[yy * cols + xx]) * gaussMatrixFlat[k_idx];
                        }
                        k_idx++;
                    }
                }
                ptr_gauss1[i * cols + j] = static_cast<uchar>(acc / 16);
            }
        }
        nvtxRangePop();

        // ---------------- gaussian_blur_5x5 ----------------
        nvtxRangePushA("gaussian_blur_5x5");
        #pragma acc parallel loop collapse(2) present(ptr_src, ptr_gauss2, gaussMatrix2Flat)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int acc = 0;
                int k_idx = 0;
                for (int a = -2; a <= 2; ++a) {
                    for (int b = -2; b <= 2; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < rows && xx >= 0 && xx < cols) {
                            acc += int(ptr_src[yy * cols + xx]) * gaussMatrix2Flat[k_idx];
                        }
                        k_idx++;
                    }
                }
                ptr_gauss2[i * cols + j] = static_cast<uchar>(acc / 273);
            }
        }
        nvtxRangePop();

        // ---------------- gaussian_blur_25x25_apply ----------------
        // Note: Kernel generation moved outside loop (Pre-calculated)
        nvtxRangePushA("gaussian_blur_25x25_apply");
        #pragma acc parallel loop collapse(2) present(ptr_src, ptr_gauss3, ptr_bigKernel)
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float acc = 0.0f;
                int half = bigSize / 2;
                
                for (int a = -half; a <= half; ++a) {
                    for (int b = -half; b <= half; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < rows && xx >= 0 && xx < cols) {
                            int k_idx = (a + half) * bigSize + (b + half);
                            acc += float(ptr_src[yy * cols + xx]) * ptr_bigKernel[k_idx];
                        }
                    }
                }
                // Use pre-calculated sum
                ptr_gauss3[i * cols + j] = static_cast<uchar>(acc / bigKernelSum);
            }
        }
        nvtxRangePop();

        nvtxRangePop(); // Iteration end
    }

    // =========================================================================
    // Clean up GPU Memory
    // =========================================================================
    #pragma acc exit data delete(ptr_src, ptr_pepper, ptr_blur, ptr_diff, \
                                 ptr_med, ptr_gauss1, ptr_gauss2, ptr_gauss3, \
                                 gaussMatrixFlat, gaussMatrix2Flat, ptr_bigKernel)

    return 0;
}