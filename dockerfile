# Use the NVIDIA HPC SDK base image (includes nvc++, CUDA, Nsight Systems, NVTX)
FROM nvcr.io/nvidia/nvhpc:24.1-devel-cuda12.3-ubuntu22.04

# Add metadata
LABEL description="OpenACC Dev Environment with OpenCV"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install OpenCV and build tools
# We assume NVTX and nvc++ are provided by the base image.
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. (Optional) Default command
# Since we will use this interactively, /bin/bash is a good default
CMD ["/bin/bash"]