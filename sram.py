import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

# CUDA Kernel: Dùng Global Memory
global_memory_kernel = """
__global__ void global_memory_add(int *a, int *b, int *c, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}
"""

# CUDA Kernel: Dùng Shared Memory
shared_memory_kernel = """
__global__ void shared_memory_add(int *a, int *b, int *c, int N) {
    extern __shared__ int shared_a[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;  

    if (tid < N) {
        shared_a[local_tid] = a[tid];
        __syncthreads();
        c[tid] = shared_a[local_tid] + b[tid];
    }
}
"""

# Biên dịch Kernel
mod_global = SourceModule(global_memory_kernel)
mod_shared = SourceModule(shared_memory_kernel)

# Lấy kernel function
global_add = mod_global.get_function("global_memory_add")
shared_add = mod_shared.get_function("shared_memory_add")

# Kích thước dữ liệu
N = 1024 * 1024  
a = np.random.randint(0, 100, N).astype(np.int32)
b = np.random.randint(0, 100, N).astype(np.int32)
c = np.zeros_like(a)

# Cấp phát bộ nhớ GPU
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

# Copy dữ liệu vào GPU
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Cấu hình CUDA
block_size = 256
grid_size = (N + block_size - 1) // block_size
shared_mem_size = block_size * a.itemsize  # Fix lỗi

# Tạo event đo thời gian
start = cuda.Event()
end = cuda.Event()

# Chạy kernel Global Memory
start.record()
global_add(a_gpu, b_gpu, c_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))
cuda.Context.synchronize()  # Fix lỗi timing
end.record()
end.synchronize()
time_hbm = start.time_till(end)

# Chạy kernel Shared Memory
start.record()
shared_add(a_gpu, b_gpu, c_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1), shared=shared_mem_size)
cuda.Context.synchronize()
end.record()
end.synchronize()
time_sram = start.time_till(end)

# Kết quả
print(f"Thời gian với Global Memory (HBM): {time_hbm:.2f} ms")
print(f"Thời gian với Shared Memory (SRAM): {time_sram:.2f} ms")
print(f"SRAM nhanh hơn HBM khoảng: {time_hbm / time_sram:.2f}x lần")
