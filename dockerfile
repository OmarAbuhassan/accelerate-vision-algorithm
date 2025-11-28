# Use the NVIDIA HPC SDK base image (includes nvc++, CUDA, Nsight Systems, NVTX)
FROM nvcr.io/nvidia/nvhpc:24.1-devel-cuda12.3-ubuntu22.04

# Add metadata
LABEL description="OpenACC & CUDA Python Dev Environment"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install OpenCV (C++), build tools, AND Python 3 + Pip
# We assume NVTX and nvc++ are provided by the base image.
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory
WORKDIR /app

# 3. Default command
CMD ["/bin/bash"]