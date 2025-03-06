#!/usr/bin/env python3
import torch
import torch.nn as nn
from collections import defaultdict

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model,))
    
    def forward(self, in_features) -> torch.FloatTensor:
        tem = torch.sum(in_features * in_features, dim = -1)
        rms = torch.sqrt(tem / self.d_model + self.eps)
        rmsnorm = (in_features / torch.unsqueeze(rms, dim = -1)) * self.weight
        return rmsnorm
