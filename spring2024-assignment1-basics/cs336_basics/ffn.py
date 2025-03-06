#!usr/env/bin python3
import torch
import torch.nn as nn

class ffn(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(ffn, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias = False)
        self.w2 = nn.Linear(d_ff, d_model, bias = False)
        # self.w1 = torch.load_state_dict(weights["w1.weight"])
        # self.w2 = torch.load_state_dict(weights["w2.weight"])
        # self.w1.weight.data = weights[name1]
        # self.w2.weight.data = weights[name2]
    
    def forward(self, in_features: torch.FloatTensor):
        return self.w2(self.GELU(self.w1(in_features)))

    def GELU(self, in_features: torch.FloatTensor) -> torch.FloatTensor:
        return (torch.erf(in_features / torch.sqrt(torch.tensor(2))) + 1) * in_features / 2