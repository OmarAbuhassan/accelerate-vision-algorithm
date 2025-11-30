// 






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

    Mat image = imread("./input/trees_big.jpg", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Failed to load ./input/trees_big.jpg" << endl;
        return -1;
    }
    Mat myMat1;

    uchar* img_data = image.ptr<uchar>(0);
    int rows = image.rows;
    int cols = image.cols;
    int size = rows * cols;


    for (int iter = 0; iter < 15; iter++) {

        string iter_name = "iteration_" + to_string(iter);
        nvtxRangePushA(iter_name.c_str());

        int arr[256] = {0};
        float arr2[256] = {0};
        float arr3[256] = {0};
        int h2[256] = {0};

        // ------------------------ histogram_original ------------------------
        nvtxRangePushA("histogram_original");

        #pragma acc enter data copyin(arr[0:256], img_data[0:size])

        #pragma acc parallel loop collapse(2) present(img_data[0:size], arr[0:256])
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int index = static_cast<int>(img_data[i * cols + j]);
                #pragma acc atomic
                arr[index]++;
            }
        }

        #pragma acc exit data copyout(arr[0:256])

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
        uchar* out_data = myMat1.ptr<uchar>(0);

        #pragma acc enter data copyin(arr3[0:256]) create(out_data[0:size])

        #pragma acc parallel loop collapse(2) present(img_data[0:size], out_data[0:size], arr3[0:256])
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                out_data[idx] = static_cast<uchar>(floor(255.0f * arr3[img_data[idx]]));
            }
        }

        #pragma acc exit data copyout(out_data[0:size]) delete(arr3[0:256])

        nvtxRangePop();

        // ------------------------ histogram_equalized -----------------------
        nvtxRangePushA("histogram_equalized");

        #pragma acc enter data copyin(out_data[0:size], h2[0:256])

        #pragma acc parallel loop collapse(2) present(out_data[0:size], h2[0:256])
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                int index = static_cast<int>(out_data[idx]);
                #pragma acc atomic
                h2[index]++;
            }
        }

        #pragma acc exit data copyout(h2[0:256]) delete(out_data[0:size], img_data[0:size])

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

    imwrite("./output/equalized_acc.jpg", myMat1);
    cout << "Output saved to ./output/equalized_acc.jpg" << endl;

    return 0;
}