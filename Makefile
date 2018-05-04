PROJECT_NAME = spgemm_sc

NVCC = nvcc
CC = g++


CUDA_INSTALL_PATH = /opt/cuda-9.1
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc -I.
CUDA_GEN = -O3 -m64 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
CXX_GEN = -O3 -m64
LINKARG = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart


CXXFLAGS = -c $(CXX_GEN) -D_FORCE_INLINES $(INCLUDES)
NVCCFLAGS = -c $(CUDA_GEN) -D_FORCE_INLINES $(INCLUDES) 

BUILD_DIR = linux_x64

all: build clean

build: build_dir gpu cpu
	$(NVCC) $(LINKARG) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o

build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

cpu:			
	$(CC) $(CXXFLAGS) *.cpp

clean:
	rm *.o

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)
