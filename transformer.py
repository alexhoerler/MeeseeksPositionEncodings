import torch
import torch.nn as nn
import math
from positionalEncoder import PositionalEncoder

class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        input_position_encoder=PositionalEncoder,
        output_position_encoder=PositionalEncoder
    ):
        super(Transformer, self).__init__()
        self.d_model = d_model
        
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.input_pos_encoder = input_position_encoder(d_model)
        self.output_pos_encoder = output_position_encoder(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.out_linear = nn.Linear(d_model, output_vocab_size)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        src = self.input_pos_encoder(src)

        tgt = self.output_embedding(tgt)
        tgt = self.output_pos_encoder(tgt)

        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out_linear(transformer_out)
        
        return out
    
    def get_tgt_mask(self, size) -> torch.tensor:
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        return (matrix == pad_token)