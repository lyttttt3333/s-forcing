from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

real_model_name = "wan_models/Wan2.2-TI2V-5B"
self.real_score = WanDiffusionWrapper(model_name=real_model_name, is_causal=False)