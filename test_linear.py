import torch
import torch.nn as nn

# 定义线性层：输入维度2048，输出维度1536
linear = nn.Linear(in_features=2048, out_features=1536)

# 输入形状：[batch_size=1, seq_len=1041, feature_dim=2048]
x = torch.randn(1, 1041, 2048)

# 通过线性层
output = linear(x)

print(output.shape)  # 输出：torch.Size([1, 1041, 1536])