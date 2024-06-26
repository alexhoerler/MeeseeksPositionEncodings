import torch
import torch.nn as nn

class IncrementalPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_length=11):
        super(IncrementalPositionalEncoder, self).__init__()

        #max pe = 1
        #1/0.05 = 20
        pe = torch.linspace(0, 1, steps=max_seq_length).unsqueeze(1)
        
        pe = pe.repeat(1, d_model)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]