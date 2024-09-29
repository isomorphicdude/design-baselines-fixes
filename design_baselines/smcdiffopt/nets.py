"""Implements the neural networks used in the diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectedWithTime(nn.Module):
    """
    A simple model with multiple fully connected layers and some Fourier features for the time variable.
    Adapted from the jax code in smcdiffopt.
    
    Attributes:
        in_size: The size of the input tensor.
        time_embed_size: The size of the time embedding.
        max_t: The maximum time value.
    """
    
    def __init__(self, in_size: int, time_embed_size: int = 4, max_t: int = 999):
        super(FullyConnectedWithTime, self).__init__()
        out_size = in_size
        self.time_embed_size = time_embed_size
        self.max_t = max_t
        
        self.layers = nn.ModuleList([
            nn.Linear(in_size + self.time_embed_size, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, out_size),
        ])
        
    def _get_time_embedding(self, t):
        t = t / self.max_t
        device = t.device
        half_dim = self.time_embed_size // 2
        emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        time_emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return time_emb
        
    def forward(self, x, t):
        t_fourier = self._get_time_embedding(t)
        # rershape t_fourier to match the batch size
        t_fourier = t_fourier.expand(x.shape[0], -1).to(x.device)
        x = torch.cat([x, t_fourier], dim=1)
        
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        
        x = self.layers[-1](x)
        
        return x