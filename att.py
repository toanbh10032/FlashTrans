
import torch
import time

# Cấu hình GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print ('device =================================', device)
# Kích thước tensor
seq_len, d_k, d_v, d_model = 128, 64, 64, 128

# Tạo dữ liệu random trên GPU
Q = torch.randn(seq_len, d_k, device=device)
K = torch.randn(seq_len, d_k, device=device)
V = torch.randn(seq_len, d_v, device=device)
W = torch.randn(d_v, d_model, device=device)

# Hàm attention sử dụng torch.matmul
def attention_matmul(Q, K, V, W):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(d_k, device=device))
    attention_weights = torch.softmax(scores, dim=-1)
    A = torch.matmul(attention_weights, V)
    O = torch.matmul(A, W)
    return O

def attention_mul(Q, K, V, W):
    d_k = Q.shape[-1]
    scores = (Q@ K.T) / d_k**0.5
    attention_weights = torch.softmax(scores, dim=-1)
    A = (attention_weights@ V)
    O = (A@ W)
    return O

# Hàm attention sử dụng einsum
def attention_einsum(Q, K, V, W):
    d_k = Q.shape[-1]
    scores = torch.einsum('ij,kj->ik', Q, K) / torch.sqrt(torch.tensor(d_k, device=device))
    attention_weights = torch.softmax(scores, dim=-1)
    A = torch.einsum('ik,kj->ij', attention_weights, V)
    O = torch.einsum('ij,jk->ik', A, W)
    return O

# Đo tốc độ
def benchmark(func, Q, K, V, W, num_runs=100):
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        func(Q, K, V, W)
    torch.cuda.synchronize()
    return (time.time() - start_time) / num_runs

# Chạy benchmark
time_matmul = benchmark(attention_matmul, Q, K, V, W)
time_mul = benchmark(attention_mul, Q, K, V, W)
time_einsum = benchmark(attention_einsum, Q, K, V, W)

print(f"MatMul GPU Time: {time_matmul:.6f} sec")
print(f"Mul GPU Time: {time_mul:.6f} sec")
print(f"Einsum GPU Time: {time_einsum:.6f} sec")
print(f"Speedup: {time_matmul / time_einsum:.2f}x")
# print(f"Speedup: {time_einsum /time_matmul:.2f}x")
