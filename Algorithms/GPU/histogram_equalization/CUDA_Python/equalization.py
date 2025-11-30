import numpy as np
import cv2
from numba import cuda
import math
import nvtx

# CUDA-specific: Shared memory size constant
HIST_BINS = 256

# ------------------------ CUDA Kernels ------------------------

# Histogram kernel using shared memory (CUDA-specific optimization)
@cuda.jit
def histogram_kernel(img_data, arr, rows, cols):
    # CUDA-specific: Shared memory allocation
    local_hist = cuda.shared.array(256, dtype=np.int32)
    
    # CUDA-specific: Thread indexing
    tid = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # Initialize shared memory
    if tid < 256:
        local_hist[tid] = 0
    
    # CUDA-specific: Block-level synchronization
    cuda.syncthreads()
    
    # Compute histogram in shared memory (faster atomics)
    if i < rows and j < cols:
        index = int(img_data[i, j])
        # CUDA-specific: Shared memory atomic (fast)
        cuda.atomic.add(local_hist, index, 1)
    
    cuda.syncthreads()
    
    # Reduce shared histogram to global memory
    if tid < 256:
        # CUDA-specific: Global memory atomic (only 256 per block)
        cuda.atomic.add(arr, tid, local_hist[tid])


# Apply equalization kernel
@cuda.jit
def apply_equalization_kernel(img_data, out_data, arr3, rows, cols):
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if i < rows and j < cols:
        index = int(img_data[i, j])
        out_data[i, j] = int(math.floor(255.0 * arr3[index]))


# Histogram equalized kernel using shared memory
@cuda.jit
def histogram_equalized_kernel(out_data, h2, rows, cols):
    local_hist = cuda.shared.array(256, dtype=np.int32)
    
    tid = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    i = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    j = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    if tid < 256:
        local_hist[tid] = 0
    
    cuda.syncthreads()
    
    if i < rows and j < cols:
        index = int(out_data[i, j])
        cuda.atomic.add(local_hist, index, 1)
    
    cuda.syncthreads()
    
    if tid < 256:
        cuda.atomic.add(h2, tid, local_hist[tid])


def main():
    # Load image
    image = cv2.imread("./input/trees_big.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Failed to load ./input/trees_big.jpg")
        return -1

    rows, cols = image.shape
    img_data = image.astype(np.uint8)

    # CUDA-specific: Define block and grid dimensions
    threads_per_block = (16, 16)  # 256 threads per block
    blocks_per_grid_x = (cols + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (rows + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    myMat1 = None

    for iter in range(15):

        with nvtx.annotate(f"iteration_{iter}"):

            arr = np.zeros(256, dtype=np.int32)
            arr2 = np.zeros(256, dtype=np.float32)
            arr3 = np.zeros(256, dtype=np.float32)
            h2 = np.zeros(256, dtype=np.int32)

            # ------------------------ histogram_original ------------------------
            with nvtx.annotate("histogram_original"):

                # CUDA-specific: Explicit device allocation with cuda.to_device
                d_img_data = cuda.to_device(img_data)
                d_arr = cuda.to_device(arr)

                # CUDA-specific: Kernel launch with [blocks, threads]
                histogram_kernel[blocks_per_grid, threads_per_block](d_img_data, d_arr, rows, cols)
                
                # CUDA-specific: Explicit synchronization
                cuda.synchronize()

                # CUDA-specific: Explicit copy back with copy_to_host()
                arr = d_arr.copy_to_host()

            # ------------------------ find_max_original_hist --------------------
            with nvtx.annotate("find_max_original_hist"):

                maxVal = 0
                for i in range(256):
                    if arr[i] > maxVal:
                        maxVal = arr[i]

            # ------------------------ pmf_compute -------------------------------
            with nvtx.annotate("pmf_compute"):

                total = float(rows * cols)
                for i in range(256):
                    arr2[i] = float(arr[i]) / total

            # ------------------------ cdf_compute -------------------------------
            with nvtx.annotate("cdf_compute"):

                arr3[0] = arr2[0]
                for i in range(1, 256):
                    arr3[i] = arr2[i] + arr3[i - 1]

            # ------------------------ apply_equalization -----------------------
            with nvtx.annotate("apply_equalization"):

                # CUDA-specific: Device array creation without host copy
                d_arr3 = cuda.to_device(arr3)
                d_out_data = cuda.device_array((rows, cols), dtype=np.uint8)

                apply_equalization_kernel[blocks_per_grid, threads_per_block](d_img_data, d_out_data, d_arr3, rows, cols)
                cuda.synchronize()

                out_data = d_out_data.copy_to_host()
                
                # CUDA-specific: Explicit memory cleanup
                del d_arr3

            # ------------------------ histogram_equalized -----------------------
            with nvtx.annotate("histogram_equalized"):

                d_out_data = cuda.to_device(out_data)
                d_h2 = cuda.to_device(h2)

                histogram_equalized_kernel[blocks_per_grid, threads_per_block](d_out_data, d_h2, rows, cols)
                cuda.synchronize()

                h2 = d_h2.copy_to_host()
                
                # CUDA-specific: Explicit memory cleanup
                del d_out_data
                del d_img_data
                del d_h2

            # ------------------------ find_max_equalized_hist -------------------
            with nvtx.annotate("find_max_equalized_hist"):

                maxH2 = 0
                for i in range(256):
                    if h2[i] > maxH2:
                        maxH2 = h2[i]

            myMat1 = out_data

    cv2.imwrite("./output/equalized_cuda_python.jpg", myMat1)
    print("Output saved to ./output/equalized_cuda_python.jpg")

    return 0


if __name__ == "__main__":
    main()