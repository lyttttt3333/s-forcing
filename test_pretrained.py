from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from wan.modules.causal_model import CausalWanModel
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock

# if is_causal:
#     print(f"########### Loading from wan_models/{model_name}/")
#     self.model = CausalWanModel.from_pretrained(
#         f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
# else:
model_name = "Wan2.2-TI2V-5B"
model = CausalWanModel.from_pretrained(f"wan_models/{model_name}/")
model.requires_grad_(False)
input()

# real_model_name = "Wan2.2-TI2V-5B"
# self.real_score = WanDiffusionWrapper(model_name=real_model_name, is_causal=False)