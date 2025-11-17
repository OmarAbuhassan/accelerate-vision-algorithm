#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include "nvToolsExt.h"

using namespace std;
using namespace cv;

int main() {

    // Load once
    Mat image = imread("./input/trees_big.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Failed to load ./input/trees_big.jpg" << endl;
        return -1;
    }

    // Run full pipeline 100 iterations (sequential)
    for (int iter = 0; iter < 15; iter++) {

        // ---- Iteration-level range ----
        string iter_name = "iteration_" + to_string(iter);
        nvtxRangePushA(iter_name.c_str());

        int arr[256]  = { 0 };
        float arr2[256] = { 0 };
        float arr3[256] = { 0 };
        int h2[256]   = { 0 };

        // ------------------------ histogram_original ------------------------
        nvtxRangePushA("histogram_original");
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int index = static_cast<int>(image.at<uchar>(i, j));
                arr[index]++;
            }
        }
        nvtxRangePop(); // histogram_original

        // ------------------------ find_max_original_hist --------------------
        nvtxRangePushA("find_max_original_hist");
        int maxVal = 0;
        for (int i = 0; i < 255; i++) {
            if (arr[i] > maxVal) {
                maxVal = arr[i];
            }
        }
        nvtxRangePop(); // find_max_original_hist

        // ------------------------ pmf_compute -------------------------------
        nvtxRangePushA("pmf_compute");
        float total = static_cast<float>(image.cols * image.rows);
        for (int i = 0; i < 255; i++) {
            arr2[i] = static_cast<float>(arr[i]) / total;
        }
        nvtxRangePop(); // pmf_compute

        // ------------------------ cdf_compute -------------------------------
        nvtxRangePushA("cdf_compute");
        arr3[0] = arr2[0];
        for (int i = 1; i < 255; i++) {
            arr3[i] = arr2[i] + arr3[i - 1];
        }
        nvtxRangePop(); // cdf_compute

        // ------------------------ apply_equalization -----------------------
        nvtxRangePushA("apply_equalization");
        Mat myMat1(image.rows, image.cols, CV_8U, Scalar(0));
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                myMat1.at<uchar>(i, j) =
                    static_cast<uchar>(floor((256 - 1) * arr3[image.at<uchar>(i, j)]));
            }
        }
        nvtxRangePop(); // apply_equalization

        // ------------------------ histogram_equalized -----------------------
        nvtxRangePushA("histogram_equalized");
        for (int i = 0; i < myMat1.rows; i++) {
            for (int j = 0; j < myMat1.cols; j++) {
                int index = static_cast<int>(myMat1.at<uchar>(i, j));
                h2[index]++;
            }
        }
        nvtxRangePop(); // histogram_equalized

        // ------------------------ find_max_equalized_hist -------------------
        nvtxRangePushA("find_max_equalized_hist");
        int maxH2 = 0;
        for (int i = 0; i < 255; i++) {
            if (h2[i] > maxH2) {
                maxH2 = h2[i];
            }
        }
        nvtxRangePop(); // find_max_equalized_hist

        // (No drawing or display here – pure computation / profiling)

        nvtxRangePop(); // iteration_X
    }

    return 0;
}
