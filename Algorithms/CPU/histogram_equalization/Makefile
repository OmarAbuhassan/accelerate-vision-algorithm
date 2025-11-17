CXX      := g++
CXXFLAGS := -std=c++11 -O2 `pkg-config --cflags opencv4` -I/usr/local/cuda/include
LDFLAGS  := `pkg-config --libs opencv4` -L/usr/local/cuda/lib64 -lnvToolsExt

TARGET   := hist_eq
SRC      := equalization.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
