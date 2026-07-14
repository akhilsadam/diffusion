import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# embeddings
class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        half = dim // 2
        self.register_buffer('freqs', torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        )[None,:,None,None])
        
    def forward(self,t):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        args = t * self.freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1).to(t.dtype)
        return embedding
