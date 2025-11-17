// Modernized Gaussian blur and related filters with NVTX instrumentation
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "nvToolsExt.h"

using namespace std;
using namespace cv;

int main() {
    // Load source images once (mimic histogram_equalization.cpp style)
    Mat image = imread("mounten.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Failed to load mounten.jpg" << endl;
        return -1;
    }
    Mat pepperNoise = imread("sun.jpg", IMREAD_GRAYSCALE);
    if (pepperNoise.empty()) {
        cerr << "Failed to load sun.jpg" << endl;
        return -1;
    }

    // Reusable output mats allocated once (will be overwritten each iteration)
    Mat blur(image.rows, image.cols, CV_8UC1);
    Mat diferencia(image.rows, image.cols, CV_8UC1);
    Mat filtroMediana(pepperNoise.rows, pepperNoise.cols, CV_8UC1);
    Mat gaussiano(image.rows, image.cols, CV_8UC1);
    Mat gaussiano2(image.rows, image.cols, CV_8UC1);
    Mat gaussiano3(image.rows, image.cols, CV_8UC1);

    const int innerMatrixIndex = 3;
    const int tamCampana = 5;
    const int gaussMatrix[3][3] = { {1,2,1},{2,4,2},{1,2,1} };
    const int gaussMatrix2[5][5] = { {1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1} };
    const int thresholdLimit = 40;

    // Run full pipeline multiple times for profiling consistency
    for (int iter = 0; iter < 15; ++iter) {
        string iter_name = string("iteration_") + to_string(iter);
        nvtxRangePushA(iter_name.c_str());

        // ---------------- average_blur ----------------
        nvtxRangePushA("average_blur");
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                float sum = 0.0f;
                for (int a = -(innerMatrixIndex/2); a <= innerMatrixIndex/2; ++a) {
                    for (int b = -(innerMatrixIndex/2); b <= innerMatrixIndex/2; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < image.rows && xx >= 0 && xx < image.cols) {
                            sum += image.at<uchar>(yy, xx);
                        }
                    }
                }
                blur.at<uchar>(i, j) = static_cast<uchar>(sum / float(innerMatrixIndex * innerMatrixIndex));
            }
        }
        nvtxRangePop(); // average_blur

        // ---------------- difference_compute ----------------
        nvtxRangePushA("difference_compute");
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                diferencia.at<uchar>(i, j) = static_cast<uchar>(abs(int(image.at<uchar>(i, j)) - int(blur.at<uchar>(i, j))));
            }
        }
        nvtxRangePop(); // difference_compute

        // ---------------- thresholding ----------------
        nvtxRangePushA("thresholding");
        for (int i = 0; i < diferencia.rows; ++i) {
            for (int j = 0; j < diferencia.cols; ++j) {
                uchar &pix = diferencia.at<uchar>(i, j);
                pix = (pix <= thresholdLimit) ? 0 : 255;
            }
        }
        nvtxRangePop(); // thresholding

        // ---------------- median_filter ----------------
        nvtxRangePushA("median_filter");
        vector<uchar> window; // reused
        window.reserve(innerMatrixIndex * innerMatrixIndex);
        for (int i = 0; i < pepperNoise.rows; ++i) {
            for (int j = 0; j < pepperNoise.cols; ++j) {
                window.clear();
                for (int a = -(innerMatrixIndex/2); a <= innerMatrixIndex/2; ++a) {
                    for (int b = -(innerMatrixIndex/2); b <= innerMatrixIndex/2; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < pepperNoise.rows && xx >= 0 && xx < pepperNoise.cols) {
                            window.push_back(pepperNoise.at<uchar>(yy, xx));
                        }
                    }
                }
                sort(window.begin(), window.end());
                filtroMediana.at<uchar>(i, j) = window[window.size()/2];
            }
        }
        nvtxRangePop(); // median_filter

        // ---------------- gaussian_blur_3x3 ----------------
        nvtxRangePushA("gaussian_blur_3x3");
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                int acc = 0;
                int y = 0;
                for (int a = -(innerMatrixIndex/2); a <= innerMatrixIndex/2; ++a) {
                    int x = 0;
                    for (int b = -(innerMatrixIndex/2); b <= innerMatrixIndex/2; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < image.rows && xx >= 0 && xx < image.cols) {
                            acc += int(image.at<uchar>(yy, xx)) * gaussMatrix[y][x];
                        }
                        ++x;
                    }
                    ++y;
                }
                gaussiano.at<uchar>(i, j) = static_cast<uchar>(acc / 16); // sum of kernel = 16
            }
        }
        nvtxRangePop(); // gaussian_blur_3x3

        // ---------------- gaussian_blur_5x5 ----------------
        nvtxRangePushA("gaussian_blur_5x5");
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                int acc = 0;
                int y = 0;
                for (int a = -(tamCampana/2); a <= tamCampana/2; ++a) {
                    int x = 0;
                    for (int b = -(tamCampana/2); b <= tamCampana/2; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < image.rows && xx >= 0 && xx < image.cols) {
                            acc += int(image.at<uchar>(yy, xx)) * gaussMatrix2[y][x];
                        }
                        ++x;
                    }
                    ++y;
                }
                gaussiano2.at<uchar>(i, j) = static_cast<uchar>(acc / 273); // sum of 5x5 kernel = 273
            }
        }
        nvtxRangePop(); // gaussian_blur_5x5

        // ---------------- gaussian_blur_25x25_kernel ----------------
        nvtxRangePushA("gaussian_blur_25x25_kernel");
        const int bigSize = 25;
        static vector<float> gaussKernel(bigSize * bigSize);
        int x0 = bigSize / 2;
        int y0 = bigSize / 2;
        const int sigma = 3;
        const float pi = 3.1415926535f;
        float denom = 2.0f * sigma * sigma;
        float normConst = 1.0f / (sigma * sigma * 2.0f * pi);
        float kernelSum = 0.0f;
        for (int y = 0; y < bigSize; ++y) {
            for (int x = 0; x < bigSize; ++x) {
                int cX = y - y0; // y index maps to row
                int cY = x0 - x; // x index maps to col
                float r2 = float(cX * cX + cY * cY);
                float val = normConst * expf(-r2 / denom);
                gaussKernel[y * bigSize + x] = val;
                kernelSum += val;
            }
        }
        nvtxRangePop(); // gaussian_blur_25x25_kernel

        // ---------------- gaussian_blur_25x25_apply ----------------
        nvtxRangePushA("gaussian_blur_25x25_apply");
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                float acc = 0.0f;
                for (int a = -(bigSize/2); a <= bigSize/2; ++a) {
                    for (int b = -(bigSize/2); b <= bigSize/2; ++b) {
                        int yy = i + a;
                        int xx = j + b;
                        if (yy >= 0 && yy < image.rows && xx >= 0 && xx < image.cols) {
                            int ky = a + bigSize/2;
                            int kx = b + bigSize/2;
                            acc += float(image.at<uchar>(yy, xx)) * gaussKernel[ky * bigSize + kx];
                        }
                    }
                }
                gaussiano3.at<uchar>(i, j) = static_cast<uchar>(acc / kernelSum);
            }
        }
        nvtxRangePop(); // gaussian_blur_25x25_apply

        nvtxRangePop(); // iteration_X
    }

    // Pure compute version (no GUI) for profiling like equalization.cpp
    return 0;
}