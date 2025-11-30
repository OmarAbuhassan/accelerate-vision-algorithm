#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <algorithm> // for std::min
#include "nvToolsExt.h"

#ifdef _OPENACC
#include <openacc.h>
#endif

using namespace std;
using namespace cv;

int main() {
    // ---------------------------------------------------------
    // 1. Setup and Loading
    // ---------------------------------------------------------
    Mat robot = imread("robot.jpg", IMREAD_GRAYSCALE);
    Mat house = imread("house.jpg", IMREAD_GRAYSCALE);

    if (robot.empty() || house.empty()) {
        cerr << "Error: could not load robot.jpg or house.jpg" << endl;
        return -1;
    }

    // Determine processing dimensions (intersection of the two images)
    int rows = (robot.rows > house.rows) ? house.rows : robot.rows;
    int cols = (robot.cols > house.cols) ? house.cols : robot.cols;

    // Allocate output images (reused every iteration)
    Mat result(rows, cols, CV_8UC1, Scalar(0));
    Mat cte(rows, cols, CV_8UC1, Scalar(0));
    Mat grad(rows, cols, CV_8UC1, Scalar(0));
    Mat result2(rows, cols, CV_8UC1, Scalar(0));
    Mat result3(rows, cols, CV_8UC1, Scalar(0));
    Mat result4(rows, cols, CV_8UC1, Scalar(0));

    // ---------------------------------------------------------
    // 2. Prepare Pointers and Strides for OpenACC
    // ---------------------------------------------------------
    
    // We need raw pointers
    unsigned char* p_robot = robot.data;
    unsigned char* p_house = house.data;
    unsigned char* p_res1  = result.data;
    unsigned char* p_cte   = cte.data;
    unsigned char* p_grad  = grad.data;
    unsigned char* p_res2  = result2.data;
    unsigned char* p_res3  = result3.data;
    unsigned char* p_res4  = result4.data;

    // CRITICAL: Get the "Step" (Stride) for each image.
    // Since images are different sizes, row 1 in robot starts at a different 
    // memory offset than row 1 in house.
    int s_robot = (int)robot.step;
    int s_house = (int)house.step;
    // Outputs are all same size, but good practice to get their steps too
    int s_res1  = (int)result.step;
    int s_cte   = (int)cte.step;
    int s_grad  = (int)grad.step;
    int s_res2  = (int)result2.step;
    int s_res3  = (int)result3.step;
    int s_res4  = (int)result4.step;

    // Calculate total bytes to copy to GPU (not just rows*cols, but full memory footprint)
    size_t bytes_robot = robot.rows * robot.step;
    size_t bytes_house = house.rows * house.step;
    size_t bytes_out   = rows * result.step; // Outputs are exactly rows*cols usually

    int k = 30;
    float dx = 255.0f / static_cast<float>(cols);
    float inc = 1.0f / static_cast<float>(rows); // Pre-calc for blend
    int total_iterations = 15;

    cout << "Starting OpenACC processing (" << total_iterations << " iterations)..." << endl;

    // ---------------------------------------------------------
    // 3. OpenACC Region
    // ---------------------------------------------------------
    
    // Note: We copyin the FULL buffer size of robot/house, even if we only use a corner of it.
    #pragma acc data copyin(p_robot[0:bytes_robot], p_house[0:bytes_house]) \
                     copyout(p_res1[0:bytes_out], p_cte[0:bytes_out], p_grad[0:bytes_out], \
                             p_res2[0:bytes_out], p_res3[0:bytes_out], p_res4[0:bytes_out])
    {
        for (int iter = 0; iter < total_iterations; ++iter) {
            string iter_name = "iteration_" + to_string(iter);
            nvtxRangePushA(iter_name.c_str());

            // --- 1. Average (Robot + House) ---
            nvtxRangePushA("combine_images_avg");
            #pragma acc parallel loop collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    // Manual stride calculation: ptr + (row * step) + col
                    unsigned char val_r = p_robot[i * s_robot + j];
                    unsigned char val_h = p_house[i * s_house + j];
                    
                    p_res1[i * s_res1 + j] = (unsigned char)((val_r + val_h) / 2);
                }
            }
            nvtxRangePop();

            // --- 2. Add Constant to Robot ---
            nvtxRangePushA("add_constant_robot");
            #pragma acc parallel loop collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    unsigned char val_r = p_robot[i * s_robot + j];
                    
                    int val = val_r + k;
                    if (val > 255) val = 255;
                    
                    p_cte[i * s_cte + j] = (unsigned char)val;
                }
            }
            nvtxRangePop();

            // --- 3. Build Horizontal Gradient ---
            nvtxRangePushA("build_gradient");
            #pragma acc parallel loop collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    float r = j * dx;
                    if (r > 255.0f) r = 255.0f;
                    
                    p_grad[i * s_grad + j] = (unsigned char)r;
                }
            }
            nvtxRangePop();

            // --- 4. Gradient Average (Robot + Gradient) ---
            nvtxRangePushA("add_gradient_avg");
            #pragma acc parallel loop collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    unsigned char val_r = p_robot[i * s_robot + j];
                    unsigned char val_g = p_grad[i * s_grad + j];
                    
                    p_res2[i * s_res2 + j] = (unsigned char)((val_r + val_g) / 2);
                }
            }
            nvtxRangePop();

            // --- 5. Gradient Saturated (Robot + Gradient) ---
            nvtxRangePushA("add_gradient_saturated");
            #pragma acc parallel loop collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    unsigned char val_r = p_robot[i * s_robot + j];
                    unsigned char val_g = p_grad[i * s_grad + j];
                    
                    int val = val_r + val_g;
                    if (val > 255) val = 255;
                    
                    p_res3[i * s_res3 + j] = (unsigned char)val;
                }
            }
            nvtxRangePop();

            // --- 6. Blended Gradient ---
            nvtxRangePushA("blend_gradient_robot_house");
            #pragma acc parallel loop collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    // Replicate the iterative 'a += inc' logic using array index
                    // In the CPU code, 'a' starts at 0, adds inc immediately, so row 0 uses (0+1)*inc
                    float local_a = (i + 1) * inc;
                    if (local_a > 1.0f) local_a = 1.0f;
                    
                    float local_b = 1.0f - local_a;
                    if (local_b < 0.0f) local_b = 0.0f;

                    unsigned char val_r = p_robot[i * s_robot + j];
                    unsigned char val_h = p_house[i * s_house + j];

                    float val = local_a * val_r + local_b * val_h;
                    
                    if (val > 255.0f) val = 255.0f;
                    if (val < 0.0f)   val = 0.0f;
                    
                    p_res4[i * s_res4 + j] = (unsigned char)val;
                }
            }
            nvtxRangePop();
        }
    } 
    // End of Data Region - data is automatically copied back to CPU here due to 'copyout'

    // ---------------------------------------------------------
    // 4. Save Outputs 
    // ---------------------------------------------------------
    cout << "Processing done. Saving images..." << endl;

    imwrite("out_average.jpg", result);
    imwrite("out_constant.jpg", cte);
    imwrite("out_gradient.jpg", grad);
    imwrite("out_grad_avg.jpg", result2);
    imwrite("out_grad_sat.jpg", result3);
    imwrite("out_blend.jpg", result4);

    return 0;
}