# Compiler
NVCC = nvcc

# OpenCV Flags (using pkg-config)
OPENCV_FLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS  := $(shell pkg-config --libs opencv4)

# CUDA Architecture (optimized for common GPUs, adjust if needed)
# -O3 for optimization
NVCC_FLAGS = -O3 $(OPENCV_FLAGS)

# Libraries to link
# -lnvToolsExt is required for NVTX profiling
LIBS = $(OPENCV_LIBS) -lnvToolsExt

# Target name
TARGET = hist_eq_gpu

# Source files
SRC = equalization.cu

# ----------------------------------------------------
# Rules
# ----------------------------------------------------

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC) $(LIBS)

run: $(TARGET)
	./$(TARGET)

profile: $(TARGET)
	nsys profile --trace=nvtx,cuda --stats=true --output=cuda_cpp_report --force-overwrite true ./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.nsys-rep *.sqlite *.qdstrm