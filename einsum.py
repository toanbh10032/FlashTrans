import torch
import time

# Thiết lập GPU nếu có
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Định nghĩa thông số
B, H, N, M, D = 2, 4, 8, 8, 64  # Batch size, Heads, Seq len, Seq len, Dim
Q = torch.randn(B, H, N, D, device=device)
K = torch.randn(B, H, M, D, device=device)
V = torch.randn(B, H, M, D, device=device)
W_out = torch.randn(D, D, device=device)
d = D ** 0.5

# 🔹 Hàm gộp
def attention_fused(Q, K, V, W_out, d):
    return torch.einsum(
        'bhnm,bhmd,dd->bhnd',
        torch.softmax(torch.einsum('bhnd,bhmd->bhnm', Q, K) / d, dim=-1),
        V,
        W_out
    )
    

# 🔹 Hàm tách lẻ
def attention_separate(Q, K, V, W_out, d):
    A = torch.einsum('bhnd,bhmd->bhnm', Q, K) / d  # Attention scores
    A = torch.softmax(A, dim=-1)  # Apply softmax
    O = torch.einsum('bhnm,bhmd->bhnd', A, V)  # Weighted sum over V
    O = torch.einsum('bhnd,dd->bhnd', O, W_out)  # Apply output weights
    return O

# 🔹 Đo thời gian thực thi trên GPU
def benchmark(func, *args, runs=100):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(runs):
        _ = func(*args)
    end.record()
    
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs  # Thời gian trung bình mỗi lần chạy (ms)

# Chạy benchmark
runs = 1000  # Số lần chạy để đo tốc độ trung bình
# time_fused = benchmark(attention_fused, Q, K, V, W_out, d, runs=runs)
# # 🔹 In kết quả
# print(f"Hàm gộp: {time_fused:.4f} ms")

time_separate = benchmark(attention_separate, Q, K, V, W_out, d, runs=runs)
print(f"Hàm tách lẻ: {time_separate:.4f} ms")
# print(f"Tăng tốc: {100 * (time_separate - time_fused) / time_separate:.2f}%")
