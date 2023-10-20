import math

import torch.nn as nn
from einops.layers.torch import Rearrange


BN = True


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n p d -> b n d p'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b n d p -> b n p d'),
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixerEncoder(nn.Module):
    def __init__(self,
                 nlayer,
                 nhid,
                 n_patches,
                 with_final_norm=True,
                 dropout=0, mask=False):
        super().__init__()
        self.n_patches = n_patches
        self.nhid = nhid
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid*4, nhid//2, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)
        if mask:
            self.out_proj = nn.Linear(42, 168)

    def forward(self, x, mask=False):
        # B, N, L, D = x.shape
        x = x * math.sqrt(self.nhid)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        if mask:
            x = self.out_proj(x.transpose(-1, -2)).transpose(-1, -2)
        return x
