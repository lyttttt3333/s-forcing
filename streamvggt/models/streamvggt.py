import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from streamvggt.models.aggregator import Aggregator
from typing import Optional, Tuple, List, Any


class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        
    def inference(self, frame, idx, query_points: torch.Tensor = None, past_key_values=None):        
        images = frame.unsqueeze(0) 
        aggregator_output = self.aggregator(
            images, 
            past_key_values=past_key_values,
            use_cache=True, 
            past_frame_idx=idx
        )
        
        if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
            aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
        else:
            aggregated_tokens, patch_start_idx = aggregator_output

        state = aggregated_tokens[-1]
        batch_size = state.shape[0]
        last_dim = state.shape[-1]
        state = state.view(batch_size, -1, last_dim)
        
        return state, patch_start_idx, past_key_values