import torch
import torch.nn as nn
import math
from positionalEncoder import PositionalEncoder

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoder(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)

    def forward(self, src, tgt):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output
    

# import math
# import os
# from tempfile import TemporaryDirectory
# from typing import Tuple

# import torch
# from torch import nn, Tensor
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from torch.utils.data import dataset

# class TransformerModel(nn.Module):

#     def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
#                  nlayers: int, dropout: float = 0.5):
#         super().__init__()
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
#         encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
#         self.embedding = nn.Embedding(ntoken, d_model)
#         self.d_model = d_model
#         self.linear = nn.Linear(d_model, ntoken)

#         self.init_weights()

#     def init_weights(self) -> None:
#         initrange = 0.1
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.linear.bias.data.zero_()
#         self.linear.weight.data.uniform_(-initrange, initrange)

#     def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
#         """
#         Arguments:
#             src: Tensor, shape ``[seq_len, batch_size]``
#             src_mask: Tensor, shape ``[seq_len, seq_len]``

#         Returns:
#             output Tensor of shape ``[seq_len, batch_size, ntoken]``
#         """
#         src = self.embedding(src) * math.sqrt(self.d_model)
#         src = self.pos_encoder(src)
#         if src_mask is None:
#             """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
#             Unmasked positions are filled with float(0.0).
#             """
#             src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
#         output = self.transformer_encoder(src, src_mask)
#         output = self.linear(output)
#         return output