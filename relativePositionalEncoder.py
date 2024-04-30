import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=10):
        super(RelativePositionalEncoder, self).__init__()
        self.d_model = d_model
        self.max_len = max_seq_len

    def forward(self, x):
        device = x.device
        unit_tensors = F.normalize(x, p=2, dim=-1)
        halved_units = F.normalize(x, p=2, dim=-1) / 2

        if len(x.shape) == 3:
            B, L, D = x.shape
            zero_tensor = torch.zeros((B, 1, D)).to(device)
            one_offset = torch.cat((zero_tensor, unit_tensors), dim=-2)
            one_offset = one_offset[:, :L, :]
            zero_tensor = torch.zeros((B, 2, D)).to(device)
            two_offset = torch.cat((zero_tensor, halved_units), dim=-2)
            two_offset = two_offset[:, :L, :]
        elif len(x.shape) == 2:
            L, D = x.shape
            zero_tensor = torch.zeros((1, D)).to(device)
            one_offset = torch.cat((zero_tensor, unit_tensors), dim=-2)
            one_offset = one_offset[:L, :]
            zero_tensor = torch.zeros((2, D)).to(device)
            two_offset = torch.cat((zero_tensor, halved_units), dim=-2)
            two_offset = two_offset[:L, :]
        
        return x + one_offset + two_offset
