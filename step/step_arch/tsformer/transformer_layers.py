import math

import numpy as np
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.transformer_encoders = []
        # for head_cnt in range(3, -1, -1):
        #     encoder_layers = TransformerEncoderLayer(hidden_dim, head_cnt+1, hidden_dim * mlp_ratio, dropout)
        #     transformer_encoder = TransformerEncoder(encoder_layers, 1)
        #     self.transformer_encoders.append(transformer_encoder)
        # self.transformer_encoders = nn.Sequential(*self.transformer_encoders)


    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        # output = self.transformer_encoders(src)
        output = output.transpose(0, 1).view(B, N, L, D)
        return output
