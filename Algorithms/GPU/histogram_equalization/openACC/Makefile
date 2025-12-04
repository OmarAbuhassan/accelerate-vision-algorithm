# --------------------------------------------------------
# Compiler & Flags
# --------------------------------------------------------
CXX = nvc++

# Use pkg-config to automatically get OpenCV flags and libs
# This fixes missing libraries like imgproc or highgui
OPENCV_FLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS  = $(shell pkg-config --libs opencv4)

# Compiler Flags
# -acc          : Enable OpenACC
# -gpu=managed  : Use Unified Memory
# -Minfo=accel  : Print optimization details
CXXFLAGS = -acc -gpu=managed -Minfo=accel $(OPENCV_FLAGS)

# Linker Flags
# 1. NVTX path (from NVIDIA SDK)
# 2. System library path
# 3. Libraries: nvToolsExt + OpenCV (via pkg-config)
LDFLAGS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda/lib64 \
          -L/usr/local/cuda/lib64 \
          -lnvToolsExt \
          $(OPENCV_LIBS)

# --------------------------------------------------------
# Targets
# --------------------------------------------------------

TARGET = hist_eq_gpu
SRC    = equalization.cpp

# Default target
all: $(TARGET)

# Build Rule
# IMPORTANT: $(LDFLAGS) must be at the END of the line
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Run Rule
run: $(TARGET)
	./$(TARGET)

# Profile Rule
profile: $(TARGET)
	nsys profile --trace=nvtx,openacc,cuda --stats=true --output=blur_report --force-overwrite true ./$(TARGET)

# Clean Rule
clean:
	rm -f $(TARGET) *.o *.nsys-rep *.qdrep