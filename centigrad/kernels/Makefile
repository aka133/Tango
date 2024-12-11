CUDA_PATH = /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc

CFLAGS = -O3 -shared -Xcompiler -fPIC
INCLUDES = -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas -lcudnn

all: libcuda_kernels.so libcublas_utils.so libcudnn_utils.so

libcuda_kernels.so: cuda_kernels.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

libcublas_utils.so: cublas_utils.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

libcudnn_utils.so: cudnn_utils.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.so