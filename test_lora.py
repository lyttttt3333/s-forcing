import torch
from peft import LoraConfig, get_peft_model

# 定义基础模型
class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
    
    def forward(self, x):
        return self.linear(x)

# 初始化并应用LoRA
base_model = BaseModel()
lora_config = LoraConfig(r=1, lora_alpha=2, target_modules=["linear"], task_type="SEQ_CLASSIFICATION")
peft_model = get_peft_model(base_model, lora_config)

# 随机输入
x = torch.tensor([[1.0, 2.0]])

# 1. 使用PEFT模型forward（包含LoRA效果）
peft_output = peft_model(x)

# 2. 直接调用基础模型forward（无LoRA效果）
base_output = peft_model.base_model.forward(x)

print("PEFT输出（含LoRA）:", peft_output)
print("基础模型输出（无LoRA）:", base_output)