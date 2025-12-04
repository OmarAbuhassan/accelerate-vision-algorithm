import cv2
import numpy as np
from numba import cuda, uint8, float32, void
import math
import os
import nvtx  # <- use nvtx directly

# ------------------------------------------------------------------------------
# CUDA KERNELS
# ------------------------------------------------------------------------------

@cuda.jit
def kernel_combine_avg(img1, img2, out):
    """
    Equivalent to: result = (robot + house) / 2
    """
    x, y = cuda.grid(2)
    rows, cols = out.shape

    if x < rows and y < cols:
        # Cast to uint16 to prevent overflow during addition, then back to uint8
        val = (np.uint16(img1[x, y]) + np.uint16(img2[x, y])) // 2
        out[x, y] = np.uint8(val)

@cuda.jit
def kernel_add_constant(img, out, k):
    """
    Equivalent to: cte = robot + k (saturated)
    """
    x, y = cuda.grid(2)
    rows, cols = out.shape

    if x < rows and y < cols:
        val = np.uint16(img[x, y]) + k
        if val > 255:
            val = 255
        out[x, y] = np.uint8(val)

@cuda.jit
def kernel_build_gradient(out, dx):
    """
    Equivalent to building the 'grad' image based on column index.
    """
    x, y = cuda.grid(2)
    rows, cols = out.shape

    if x < rows and y < cols:
        val = float32(y) * dx
        if val > 255.0:
            val = 255.0
        out[x, y] = np.uint8(val)

@cuda.jit
def kernel_add_grad_avg(img, grad, out):
    """
    Equivalent to: result2 = (robot + grad) / 2
    """
    x, y = cuda.grid(2)
    rows, cols = out.shape

    if x < rows and y < cols:
        val = (np.uint16(img[x, y]) + np.uint16(grad[x, y])) // 2
        out[x, y] = np.uint8(val)

@cuda.jit
def kernel_add_grad_saturated(img, grad, out):
    """
    Equivalent to: result3 = robot + grad (saturated)
    """
    x, y = cuda.grid(2)
    rows, cols = out.shape

    if x < rows and y < cols:
        val = np.uint16(img[x, y]) + np.uint16(grad[x, y])
        if val > 255:
            val = 255
        out[x, y] = np.uint8(val)

@cuda.jit
def kernel_blend_gradient(img1, img2, out):
    """
    Equivalent to blending robot and house based on row index.
    Matches C++ logic:
    a starts at 0, adds inc per row (inside loop).
    b starts at 1, subtracts inc per row (inside loop).
    """
    x, y = cuda.grid(2)
    rows, cols = out.shape

    if x < rows and y < cols:
        inc = 1.0 / float32(rows)
        
        # C++: a += inc (starts at 0.0)
        # For row x, this is (x + 1) * inc
        a = (float32(x) + 1.0) * inc
        if a > 1.0:
            a = 1.0
        
        # C++: b -= inc (starts at 1.0)
        # For row x, this is 1.0 - (x + 1) * inc
        b = 1.0 - ((float32(x) + 1.0) * inc)
        if b < 0.0:
            b = 0.0

        p1 = float32(img1[x, y])
        p2 = float32(img2[x, y])
        
        val = (a * p1) + (b * p2)
        
        if val > 255.0:
            val = 255.0
        if val < 0.0:
            val = 0.0
        
        out[x, y] = np.uint8(val)

# ------------------------------------------------------------------------------
# HELPER: Data Generator
# ------------------------------------------------------------------------------
def get_images():
    """Loads images or generates synthetic ones if files are missing."""
    img1 = cv2.imread("robot.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("house.jpg", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Warning: Images not found. Generating synthetic data for demonstration.")
        # Generate 1024x1024 noise images
        img1 = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
    
    return img1, img2

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    # 1. Load Data
    robot_host, house_host = get_images()

    # Determine common size (C++ logic)
    rows = min(robot_host.shape[0], house_host.shape[0])
    cols = min(robot_host.shape[1], house_host.shape[1])

    # Crop to match sizes
    robot_host = np.ascontiguousarray(robot_host[:rows, :cols])
    house_host = np.ascontiguousarray(house_host[:rows, :cols])

    print(f"Processing Image Size: {rows}x{cols}")

    # 2. Configure CUDA Grid
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(rows / threads_per_block[0])
    blocks_per_grid_y = math.ceil(cols / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    print(f"Grid Config: {blocks_per_grid} blocks, {threads_per_block} threads/block")

    # 3. GPU Memory Allocation & Transfer (Done ONCE)
    d_robot = cuda.to_device(robot_host)
    d_house = cuda.to_device(house_host)

    # Allocate output arrays on GPU
    d_result = cuda.device_array((rows, cols), dtype=np.uint8)
    d_cte = cuda.device_array((rows, cols), dtype=np.uint8)
    d_grad = cuda.device_array((rows, cols), dtype=np.uint8)
    d_result2 = cuda.device_array((rows, cols), dtype=np.uint8)
    d_result3 = cuda.device_array((rows, cols), dtype=np.uint8)
    d_result4 = cuda.device_array((rows, cols), dtype=np.uint8)

    # Constants
    k = 30
    dx = float32(255.0 / cols)

    # 4. Processing Loop (15 Iterations)
    for i in range(15):
        with nvtx.annotate(f"iteration_{i}"):

            # --- Kernel 1: Average ---
            with nvtx.annotate("combine_images_avg"):
                kernel_combine_avg[blocks_per_grid, threads_per_block](d_robot, d_house, d_result)
                cuda.synchronize()

            # --- Kernel 2: Add Constant ---
            with nvtx.annotate("add_constant_robot"):
                kernel_add_constant[blocks_per_grid, threads_per_block](d_robot, d_cte, k)
                cuda.synchronize()

            # --- Kernel 3: Build Gradient ---
            with nvtx.annotate("build_gradient"):
                kernel_build_gradient[blocks_per_grid, threads_per_block](d_grad, dx)
                cuda.synchronize()

            # --- Kernel 4: Add Gradient (Avg) ---
            with nvtx.annotate("add_gradient_avg"):
                kernel_add_grad_avg[blocks_per_grid, threads_per_block](d_robot, d_grad, d_result2)
                cuda.synchronize()

            # --- Kernel 5: Add Gradient (Sat) ---
            with nvtx.annotate("add_gradient_saturated"):
                kernel_add_grad_saturated[blocks_per_grid, threads_per_block](d_robot, d_grad, d_result3)
                cuda.synchronize()

            # --- Kernel 6: Blend ---
            with nvtx.annotate("blend_gradient_robot_house"):
                kernel_blend_gradient[blocks_per_grid, threads_per_block](d_robot, d_house, d_result4)
                cuda.synchronize()

    print("Processing complete.")

    # 5. Retrieve Results (Copy back to Host)
    result = d_result.copy_to_host()
    cte = d_cte.copy_to_host()
    grad = d_grad.copy_to_host()
    result2 = d_result2.copy_to_host()
    result3 = d_result3.copy_to_host()
    result4 = d_result4.copy_to_host()

    # 6. Save/Visualize Results
    cv2.imwrite("out_average.jpg", result)
    cv2.imwrite("out_constant.jpg", cte)
    cv2.imwrite("out_gradient.jpg", grad)
    cv2.imwrite("out_grad_avg.jpg", result2)
    cv2.imwrite("out_grad_sat.jpg", result3)
    cv2.imwrite("out_blend.jpg", result4)
    
    print("Saved: out_average.jpg, out_constant.jpg, out_gradient.jpg, out_grad_avg.jpg, out_grad_sat.jpg, out_blend.jpg")

if __name__ == "__main__":
    if cuda.is_available():
        main()
    else:
        print("Error: CUDA not detected on this system.")
