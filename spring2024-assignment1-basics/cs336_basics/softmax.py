#!usr/bin/env python3
import torch
import torch.nn as nn

class softmax(nn.Module):
    def __init__(self, dim_: int):
        super(softmax, self).__init__()
        self.dim = dim_

    def forward(self, in_features):
        max_item, _ = torch.max(in_features, dim = self.dim, keepdim = True)
        tem = in_features - max_item
        numerator = torch.exp(tem)
        denominator = torch.unsqueeze(torch.sum(numerator, dim = self.dim), dim = self.dim)
        return numerator / denominator