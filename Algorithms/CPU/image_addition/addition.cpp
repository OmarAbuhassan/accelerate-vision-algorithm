#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <string>

// NVTX for profiling (Keep this included)
#include "nvToolsExt.h"

using namespace std;
using namespace cv;

int main() {
    // Load images (once)
    Mat robot = imread("robot.jpg", IMREAD_GRAYSCALE);
    Mat house = imread("house.jpg", IMREAD_GRAYSCALE);

    if (robot.empty() || house.empty()) {
        cerr << "Error: could not load robot.jpg or house.jpg" << endl;
        return -1;
    }

    // Determine common size
    int rows = (robot.rows > house.rows) ? house.rows : robot.rows;
    int cols = (robot.cols > house.cols) ? house.cols : robot.cols;

    // Allocate output images once, reused every iteration
    Mat result(rows, cols, CV_8UC1, Scalar(0));
    Mat cte(rows, cols, CV_8UC1, Scalar(0));
    Mat grad(rows, cols, CV_8UC1, Scalar(0));
    Mat result2(rows, cols, CV_8UC1, Scalar(0));
    Mat result3(rows, cols, CV_8UC1, Scalar(0));
    Mat result4(rows, cols, CV_8UC1, Scalar(0));

    int k = 30;
    float dx = 255.0f / static_cast<float>(cols);
    int total_iterations = 15;

    // Run 15 sequential iterations
    for (int iter = 0; iter < total_iterations; ++iter) {
        // Iteration-level NVTX range
        string iter_name = "iteration_" + to_string(iter);
        nvtxRangePushA(iter_name.c_str());

        // ---------------- combine images (average) ----------------
        nvtxRangePushA("combine_images_avg");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.at<uchar>(i, j) =
                    static_cast<uchar>((robot.at<uchar>(i, j) + house.at<uchar>(i, j)) / 2);
            }
        }
        nvtxRangePop(); // combine_images_avg

        // ---------------- add constant to robot -------------------
        nvtxRangePushA("add_constant_robot");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int val = robot.at<uchar>(i, j) + k;
                if (val > 255) val = 255;
                cte.at<uchar>(i, j) = static_cast<uchar>(val);
            }
        }
        nvtxRangePop(); // add_constant_robot

        // ---------------- build horizontal gradient --------------
        nvtxRangePushA("build_gradient");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float r = j * dx;
                if (r > 255.0f) r = 255.0f;
                grad.at<uchar>(i, j) = static_cast<uchar>(r);
            }
        }
        nvtxRangePop(); // build_gradient

        // ---------------- add gradient (average) ------------------
        nvtxRangePushA("add_gradient_avg");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result2.at<uchar>(i, j) =
                    static_cast<uchar>((robot.at<uchar>(i, j) + grad.at<uchar>(i, j)) / 2);
            }
        }
        nvtxRangePop(); // add_gradient_avg

        // ---------------- add gradient (saturated sum) -----------
        nvtxRangePushA("add_gradient_saturated");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int val = robot.at<uchar>(i, j) + grad.at<uchar>(i, j);
                if (val > 255) val = 255;
                result3.at<uchar>(i, j) = static_cast<uchar>(val);
            }
        }
        nvtxRangePop(); // add_gradient_saturated

        // ---------------- blended gradient between robot/house ---
        nvtxRangePushA("blend_gradient_robot_house");
        float a = 0.0f;
        float b = 1.0f;
        float inc = 1.0f / static_cast<float>(rows);

        for (int i = 0; i < rows; i++) {
            // update blending weights per row
            a += inc;
            if (a > 1.0f) a = 1.0f;
            b -= inc;
            if (b < 0.0f) b = 0.0f;

            for (int j = 0; j < cols; j++) {
                float val = a * robot.at<uchar>(i, j) + b * house.at<uchar>(i, j);
                if (val > 255.0f) val = 255.0f;
                if (val < 0.0f) val = 0.0f;
                result4.at<uchar>(i, j) = static_cast<uchar>(val);
            }
        }
        nvtxRangePop(); // blend_gradient_robot_house

        // === NEW: SAVE ONLY ON LAST ITERATION ===
        if (iter == total_iterations - 1) {
            cout << "Last iteration reached (" << iter << "). Saving images..." << endl;
            imwrite("out_average.jpg", result);
            imwrite("out_constant.jpg", cte);
            imwrite("out_gradient.jpg", grad);
            imwrite("out_grad_avg.jpg", result2);
            imwrite("out_grad_sat.jpg", result3);
            imwrite("out_blend.jpg", result4);
        }

        nvtxRangePop(); // iteration_X
    }

    return 0;
}