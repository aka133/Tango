CUDA_PATH = /usr/local/cuda
NVCC = /usr/bin/nvcc
TORCH_CUDA_DSA = 0

CFLAGS = -O3 -shared -Xcompiler -fPIC
INCLUDES = -I$(CUDA_PATH)/include -I/root/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -L/usr/local/lib/python3.10/dist-packages/torch/lib -lcudart -lcublas

all: libcuda_kernels.so libcublas_utils.so

libcuda_kernels.so: cuda_kernels.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

libcublas_utils.so: cublas_utils.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

# libcudnn_utils.so: cudnn_utils.cu
#	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.so