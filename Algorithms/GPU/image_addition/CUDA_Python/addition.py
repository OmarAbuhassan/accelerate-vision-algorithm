import cv2
import numpy as np
from numba import cuda, uint8, float32, void
import math
import os

# ------------------------------------------------------------------------------
# NVTX HANDLING (Graceful Fallback)
# ------------------------------------------------------------------------------
try:
    from numba.cuda import nvtx
except ImportError:
    print("Warning: Numba NVTX not found. Profiling ranges will be skipped.")
    # Define a dummy class to prevent crashes if nvtx is missing
    class nvtx:
        @staticmethod
        def range_push(name):
            pass
            
        @staticmethod
        def range_pop():
            pass

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
    """
    x, y = cuda.grid(2)
    rows, cols = out.shape

    if x < rows and y < cols:
        # Calculate weights based on row index (x)
        # inc = 1.0 / rows
        # a = x * inc
        
        inc = 1.0 / float32(rows)
        a = float32(x) * inc
        
        # Clamp a
        if a > 1.0: a = 1.0
        
        b = 1.0 - a
        if b < 0.0: b = 0.0

        p1 = float32(img1[x, y])
        p2 = float32(img2[x, y])
        
        val = (a * p1) + (b * p2)
        
        if val > 255.0: val = 255.0
        if val < 0.0: val = 0.0
        
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
    # We maintain data on the device to avoid PCI-e bottleneck inside the loop
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
        range_name = f"iteration_{i}"
        
        # NVTX Range Push (Iteration)
        nvtx.range_push(range_name)
        
        # --- Kernel 1: Average ---
        nvtx.range_push("combine_images_avg")
        kernel_combine_avg[blocks_per_grid, threads_per_block](d_robot, d_house, d_result)
        cuda.synchronize() # Optional: Ensures accurate NVTX timing for this block
        nvtx.range_pop()

        # --- Kernel 2: Add Constant ---
        nvtx.range_push("add_constant_robot")
        kernel_add_constant[blocks_per_grid, threads_per_block](d_robot, d_cte, k)
        cuda.synchronize()
        nvtx.range_pop()

        # --- Kernel 3: Build Gradient ---
        nvtx.range_push("build_gradient")
        kernel_build_gradient[blocks_per_grid, threads_per_block](d_grad, dx)
        cuda.synchronize()
        nvtx.range_pop()

        # --- Kernel 4: Add Gradient (Avg) ---
        nvtx.range_push("add_gradient_avg")
        kernel_add_grad_avg[blocks_per_grid, threads_per_block](d_robot, d_grad, d_result2)
        cuda.synchronize()
        nvtx.range_pop()

        # --- Kernel 5: Add Gradient (Sat) ---
        nvtx.range_push("add_gradient_saturated")
        kernel_add_grad_saturated[blocks_per_grid, threads_per_block](d_robot, d_grad, d_result3)
        cuda.synchronize()
        nvtx.range_pop()

        # --- Kernel 6: Blend ---
        nvtx.range_push("blend_gradient_robot_house")
        kernel_blend_gradient[blocks_per_grid, threads_per_block](d_robot, d_house, d_result4)
        cuda.synchronize()
        nvtx.range_pop()

        # NVTX Range Pop (Iteration)
        nvtx.range_pop()

    print("Processing complete.")

    # 5. Retrieve Results (Copy back to Host)
    # We only bring back the final results to save time
    result_host = d_result.copy_to_host()
    result4_host = d_result4.copy_to_host()

    # 6. Save/Visualize Results
    # Saving just two examples to prove it works
    cv2.imwrite("output_avg_cuda.jpg", result_host)
    cv2.imwrite("output_blend_cuda.jpg", result4_host)
    print("Saved output_avg_cuda.jpg and output_blend_cuda.jpg")

if __name__ == "__main__":
    if cuda.is_available():
        main()
    else:
        print("Error: CUDA not detected on this system.")