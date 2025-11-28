import sys
import os
import math
import numpy as np
import cv2
from numba import cuda, uint8, float32

# Prevent TBB/OpenMP conflicts with OpenCV
# os.environ["NUMBA_DISABLE_TBB"] = "1"

# this script works in colab environment

# ==============================================================================
# HELPERS
# ==============================================================================

# Minimal NVTX wrapper for profiling
try:
    import nvtx
except ImportError:
    class nvtx:
        @staticmethod
        def annotate(*args, **kwargs):
            class Dummy:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            return Dummy()

# ==============================================================================
# GPU KERNELS
# ==============================================================================

@cuda.jit
def average_blur_kernel(src, dst, rows, cols):
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        acc = 0.0
        for a in range(-1, 2):
            for b in range(-1, 2):
                r_idx = y + a
                c_idx = x + b
                if 0 <= r_idx < rows and 0 <= c_idx < cols:
                    acc += src[r_idx, c_idx]
        dst[y, x] = uint8(acc / 9.0)

@cuda.jit
def difference_kernel(img1, img2, dst, rows, cols):
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        val = abs(int(img1[y, x]) - int(img2[y, x]))
        dst[y, x] = uint8(val)

@cuda.jit
def threshold_kernel(src, rows, cols, limit):
    idx = cuda.grid(1)
    if idx < rows * cols:
        row = idx // cols
        col = idx % cols
        if src[row, col] <= limit:
            src[row, col] = 0
        else:
            src[row, col] = 255

@cuda.jit
def median_filter_kernel(src, dst, rows, cols):
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        window = cuda.local.array(9, dtype=uint8)
        count = 0
        # Collect
        for a in range(-1, 2):
            for b in range(-1, 2):
                r_idx = y + a
                c_idx = x + b
                val = 0
                if 0 <= r_idx < rows and 0 <= c_idx < cols:
                    val = src[r_idx, c_idx]
                window[count] = val
                count += 1
        # Sort
        for i in range(8):
            for j in range(0, 8 - i):
                if window[j] > window[j + 1]:
                    temp = window[j]
                    window[j] = window[j + 1]
                    window[j + 1] = temp
        dst[y, x] = window[4]

@cuda.jit
def gaussian_3x3_kernel(src, dst, rows, cols, kernel):
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        acc = 0
        k_idx = 0
        for a in range(-1, 2):
            for b in range(-1, 2):
                r_idx = y + a
                c_idx = x + b
                if 0 <= r_idx < rows and 0 <= c_idx < cols:
                    acc += int(src[r_idx, c_idx]) * kernel[k_idx]
                k_idx += 1
        dst[y, x] = uint8(acc / 16)

@cuda.jit
def gaussian_5x5_kernel(src, dst, rows, cols, kernel):
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        acc = 0
        k_idx = 0
        for a in range(-2, 3):
            for b in range(-2, 3):
                r_idx = y + a
                c_idx = x + b
                if 0 <= r_idx < rows and 0 <= c_idx < cols:
                    acc += int(src[r_idx, c_idx]) * kernel[k_idx]
                k_idx += 1
        dst[y, x] = uint8(acc / 273)

@cuda.jit
def gaussian_25x25_kernel(src, dst, rows, cols, kernel, kernel_sum):
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        acc = 0.0
        # 25x25 Kernel application
        for a in range(-12, 13):
            for b in range(-12, 13):
                r_idx = y + a
                c_idx = x + b
                if 0 <= r_idx < rows and 0 <= c_idx < cols:
                    k_idx = (a + 12) * 25 + (b + 12)
                    acc += float(src[r_idx, c_idx]) * kernel[k_idx]
        dst[y, x] = uint8(acc / kernel_sum)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if not cuda.is_available():
        print("Error: CUDA not available. Check your runtime/drivers.")
        sys.exit(1)

    print("Loading data...")

    img_host = cv2.imread("../data/mounten.jpg", cv2.IMREAD_GRAYSCALE)
    pepper_host = cv2.imread("../data/sun.jpg", cv2.IMREAD_GRAYSCALE)

    if img_host is None or pepper_host is None:
        print("Error: Failed to load images.")
        sys.exit(1)

    # Resize to match dimensions
    if img_host.shape != pepper_host.shape:
        pepper_host = cv2.resize(pepper_host, (img_host.shape[1], img_host.shape[0]))

    # Critical: Ensure contiguous memory layout for GPU transfer
    img_host = np.ascontiguousarray(img_host)
    pepper_host = np.ascontiguousarray(pepper_host)
    rows, cols = img_host.shape

    # Prepare Constants
    gauss3x3 = np.array([1,2,1, 2,4,2, 1,2,1], dtype=np.int32)
    gauss5x5 = np.array([1,4,7,4,1, 4,16,26,16,4, 7,26,41,26,7, 4,16,26,16,4, 1,4,7,4,1], dtype=np.int32)

    # Prepare 25x25 Kernel
    bigSize = 25
    sigma = 3.0
    gauss25x25 = np.zeros((bigSize, bigSize), dtype=np.float32)
    k_sum = 0.0
    half = bigSize // 2
    for y in range(bigSize):
        for x in range(bigSize):
            cX = y - half
            cY = half - x
            val = math.exp(-(cX*cX + cY*cY) / (2 * sigma * sigma))
            gauss25x25[y, x] = val
            k_sum += val
    gauss25x25_flat = np.ascontiguousarray(gauss25x25.flatten())

    print(f"Allocating GPU memory ({cols}x{rows})...")
    d_img = cuda.to_device(img_host)
    d_pepper = cuda.to_device(pepper_host)
    
    # Output buffers
    d_blur = cuda.device_array_like(img_host)
    d_diff = cuda.device_array_like(img_host)
    d_med = cuda.device_array_like(img_host)
    d_g1 = cuda.device_array_like(img_host)
    d_g2 = cuda.device_array_like(img_host)
    d_g3 = cuda.device_array_like(img_host)
    
    # Kernel buffers
    d_k3 = cuda.to_device(gauss3x3)
    d_k5 = cuda.to_device(gauss5x5)
    d_k25 = cuda.to_device(gauss25x25_flat)

    # Grid Configuration
    threads_per_block = (16, 16)
    blocks_x = int(math.ceil(cols / threads_per_block[0]))
    blocks_y = int(math.ceil(rows / threads_per_block[1]))
    blocks_per_grid = (blocks_x, blocks_y)
    
    threads_1d = 256
    blocks_1d = int(math.ceil((rows * cols) / threads_1d))

    print("Running Pipeline (15 Iterations)...")
    
    for i in range(15):
        with nvtx.annotate(f"iteration_{i}", color="blue"):
            
            with nvtx.annotate("average_blur", color="green"):
                average_blur_kernel[blocks_per_grid, threads_per_block](d_img, d_blur, rows, cols)
            
            with nvtx.annotate("difference_compute", color="red"):
                difference_kernel[blocks_per_grid, threads_per_block](d_img, d_blur, d_diff, rows, cols)
            
            with nvtx.annotate("thresholding", color="yellow"):
                threshold_kernel[blocks_1d, threads_1d](d_diff, rows, cols, 40)
            
            with nvtx.annotate("median_filter", color="purple"):
                median_filter_kernel[blocks_per_grid, threads_per_block](d_pepper, d_med, rows, cols)
            
            with nvtx.annotate("gaussian_3x3", color="orange"):
                gaussian_3x3_kernel[blocks_per_grid, threads_per_block](d_img, d_g1, rows, cols, d_k3)
            
            with nvtx.annotate("gaussian_5x5", color="orange"):
                gaussian_5x5_kernel[blocks_per_grid, threads_per_block](d_img, d_g2, rows, cols, d_k5)
            
            with nvtx.annotate("gaussian_25x25", color="orange"):
                gaussian_25x25_kernel[blocks_per_grid, threads_per_block](d_img, d_g3, rows, cols, d_k25, k_sum)
        
        # Synchronize once per iteration to ensure the profiler captures the full iteration time
        cuda.synchronize()

    print("Done.")

if __name__ == "__main__":
    main()