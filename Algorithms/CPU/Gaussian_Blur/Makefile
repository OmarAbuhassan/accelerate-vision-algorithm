CXX      := g++
# Using same optimization level/style as histogram_equalization. Bump std if needed.
CXXFLAGS := -std=c++11 -O2 `pkg-config --cflags opencv4` -I/usr/local/cuda/include
LDFLAGS  := `pkg-config --libs opencv4` -L/usr/local/cuda/lib64 -lnvToolsExt

TARGET   := gaussian_profile
SRC      := gauss.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean run

# Notes:
# 1. If pkg-config opencv4 fails, manually set include/lib paths, e.g.:
#    CXXFLAGS := -std=c++11 -O2 -I/usr/local/include/opencv4 -I/usr/local/cuda/include
#    LDFLAGS  := -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -L/usr/local/cuda/lib64 -lnvToolsExt
# 2. NVTX headers come from CUDA toolkit installation (/usr/local/cuda/include). Ensure CUDA is installed.
# 3. Profiling: use NVIDIA Nsight Systems or nvprof (where available) to capture NVTX ranges.
