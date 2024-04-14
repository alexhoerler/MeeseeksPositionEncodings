import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(RelativePositionalEncoder, self).__init__()
        self.rel_pos_embeddings = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, q, k):
        seq_len = q.size(1)

        range_matrix = torch.arange(seq_len).unsqueeze(0).repeat(seq_len, 1)
        distance_matrix = range_matrix - range_matrix.t()

        clipped_distances = distance_matrix.clamp(min=0, max=self.rel_pos_embeddings.size(0) - 1)
        rel_embeddings = self.rel_pos_embeddings[clipped_distances]

        attn_scores = torch.einsum('bij,bjk->bik', q, k.transpose(-2, -1))
        attn_scores += torch.einsum('bij,jik->bik', q, rel_embeddings)

        return attn_scores