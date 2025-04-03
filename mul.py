import torch
import time

def attention_einsum(Q, K, V):
    A = torch.einsum('bhnd,bhmd->bhnm', Q, K) / (Q.shape[-1] ** 0.5)  # (B, H, N, M)
    A = torch.softmax(A, dim=-1)
    O = torch.einsum('bhnm,bhmd->bhnd', A, V)  # (B, H, N, D_head)
    return O

def attention_matmul(Q, K, V):
    A = (Q @ K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # (B, H, N, M)
    A = torch.softmax(A, dim=-1)
    O = A @ V  # (B, H, N, D_head)
    return O

# Dữ liệu đầu vào
B, H, N, D = 8, 8, 64, 128
Q = torch.randn(B, H, N, D // H, device='cuda')
K = torch.randn(B, H, N, D // H, device='cuda')
V = torch.randn(B, H, N, D // H, device='cuda')

# # Đo thời gian chạy
# start = time.time()
# O_einsum = attention_einsum(Q, K, V)
# torch.cuda.synchronize()
# einsum_time = time.time() - start
# print(f"Einsum time: {einsum_time:.6f} s")



start = time.time()
O_matmul = attention_matmul(Q, K, V)
torch.cuda.synchronize()
matmul_time = time.time() - start
print(f"Matmul time: {matmul_time:.6f} s")
