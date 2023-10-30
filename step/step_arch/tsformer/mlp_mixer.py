import math
from typing import Optional

import torch
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


class GConv(nn.Module):
    def __init__(self, A, dropout):
        super().__init__()
        self.A = A
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        self.A = self.A.to(x.device)
        x = nn.GELU()(self.dropout(torch.einsum('nclv,vw->nclw', (x, self.A))))
        return x


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, node_number, token_dim, channel_dim, dropout=0., adj=Optional[torch.Tensor]):
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
        self.node_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n p d -> b p d n'),
            GConv(adj, dropout),
            FeedForward(node_number, node_number, dropout),
            Rearrange('b p d n -> b n p d'),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        x = x + self.node_mixer(x)
        return x


class MLPMixerEncoder(nn.Module):
    def __init__(self,
                 nlayer,
                 nhid,
                 n_patches,
                 node_number,
                 with_final_norm=True,
                 dropout=0, mask=False, adj=None):
        super().__init__()
        self.n_patches = n_patches
        self.nhid = nhid
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, node_number, nhid*4, nhid//2, dropout=dropout, adj=adj) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, mask=False):
        # B, N, L, D = x.shape
        x = x * math.sqrt(self.nhid)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x, x
