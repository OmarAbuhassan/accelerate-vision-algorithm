CXX := g++
CXXFLAGS := -std=c++11 -O2 `pkg-config --cflags opencv4` -I/usr/local/cuda/include
LDFLAGS := `pkg-config --libs opencv4` -L/usr/local/cuda/lib64 -lnvToolsExt

TARGET := blend_ops
SRC := addition.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
