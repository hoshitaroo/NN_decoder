import torch

# 5×20のテンソルを作成
original_tensor = torch.randn(20, 5)
print(original_tensor)

# 5×20のテンソルを2つの5×10のテンソルに分割
chunks = torch.chunk(original_tensor, 2, dim=0)

# 分割されたテンソルを表示
tensor1, tensor2 = chunks
print("Tensor 1:")
print(tensor1)
print("Tensor 2:")
print(tensor2)
