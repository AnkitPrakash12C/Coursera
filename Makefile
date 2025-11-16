```makefile
# Compiler and Flags
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_35 --std=c++11

# Target
TARGET = sobel_gpu.exe

# Build Rules
all: build

build: $(TARGET)

$(TARGET): sobel_gpu.cu
	$(NVCC) $(NVCC_FLAGS) sobel_gpu.cu -o $(TARGET)

run: build
	./$(TARGET) -w 2048 -h 2048

clean:
	rm -f $(TARGET) output_*.pgm