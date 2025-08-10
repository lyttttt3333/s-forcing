import torch
import torch.nn as nn

class DimensionReductionAdapter(nn.Module):
    def __init__(self, input_dim=2048, output_dim=1536, use_activation=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.main_proj = nn.Linear(input_dim, output_dim)
        
        self.use_activation = use_activation
        if use_activation:
            self.activation = nn.GELU()  
            self.norm = nn.LayerNorm(output_dim) 
        
    def forward(self, x):
        x = self.main_proj(x)
        if self.use_activation:
            x = self.activation(x)
            x = self.norm(x)
        return x


if __name__ == "__main__":
    # 1. 初始化适配器
    adapter = DimensionReductionAdapter(
        input_dim=2048,
        output_dim=1536,
        use_activation=True  # 测试带激活函数的情况
    )
    print("适配器初始化完成")
    print(f"输入维度: {adapter.input_dim}, 输出维度: {adapter.output_dim}\n")

    # 2. 检查权重维度（验证是否为2D矩阵，避免之前的"1-D tensor"错误）
    print("主投影层权重信息:")
    print(f"权重形状: {adapter.main_proj.weight.shape}")  # 应输出 [1536, 2048]
    print(f"权重维度: {adapter.main_proj.weight.dim()} (应为2)\n")  # 确认是2D矩阵

    # 3. 生成测试输入（模拟 [batch_size=1, seq_len=1041, feature_dim=2048] 的 tensor）
    test_input = torch.randn(1, 1041, 2048)  # 随机生成符合形状的输入
    print("测试输入信息:")
    print(f"输入形状: {test_input.shape}")  # 应输出 [1, 1041, 2048]
    print(f"输入数据类型: {test_input.dtype}\n")  # 默认为 float32

    # 4. 执行前向传播
    adapter.eval()  # 切换到评估模式（避免 BatchNorm 等层的随机行为，这里无影响但规范）
    with torch.no_grad():  # 关闭梯度计算，加快推理
        test_output = adapter(test_input)

    # 5. 验证输出结果
    print("测试输出信息:")
    print(f"输出形状: {test_output.shape}")  # 应输出 [1, 1041, 1536]
    print(f"输出数据类型: {test_output.dtype}")  # 应与输入一致（float32）
    print(f"输出最小值: {test_output.min().item():.4f}, 最大值: {test_output.max().item():.4f}")  # 确认数值合