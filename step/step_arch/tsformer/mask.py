import random
import networkx as nx
import torch
from torch import nn


class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        mask = list(range(int(self.num_tokens)))
        random.shuffle(mask)
        mask_len = int(self.num_tokens * self.mask_ratio)
        self.masked_tokens = mask[:mask_len]
        self.unmasked_tokens = mask[mask_len:]
        if self.sort:
            self.masked_tokens = sorted(self.masked_tokens)
            self.unmasked_tokens = sorted(self.unmasked_tokens)
        return self.unmasked_tokens, self.masked_tokens

    def forward(self):
        self.unmasked_tokens, self.masked_tokens = self.uniform_rand()
        return self.unmasked_tokens, self.masked_tokens


class SpatialMaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, city, mask_ratio):
        super().__init__()
        self.city = city
        self.mask_ratio = mask_ratio
        self.sort = True

    def spatial_masking(self):
        mask = list(range(int(self.city)))
        random.shuffle(mask)
        mask_len = int(self.city * self.mask_ratio)
        self.masked_tokens = sorted(mask[:mask_len])
        self.unmasked_tokens = sorted(mask[mask_len:])
        return self.unmasked_tokens, self.masked_tokens

    def _build_nx_object(self, graph_adj) -> nx.Graph:
        return nx.Graph(graph_adj)

    def forward(self,graph_adj=None):
        self.unmasked_tokens, self.masked_tokens = self.spatial_masking()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        return self.unmasked_tokens, self.masked_tokens

class KnnUtil:

    def _build_nx_object(graph_adj) -> nx.Graph:
        return nx.Graph(graph_adj)
    @staticmethod
    def knn_masking(graph_adj, city, mask_ratio):
        marked_nodes = [False] * city
        graph = KnnUtil._build_nx_object(graph_adj)
        init_node = random.choice(nx.center(graph))
        from collections import deque
        pot_nodes = deque()
        mask = list()
        pot_nodes.append(init_node)
        while len(list(pot_nodes)) != 0:
            cur_node = pot_nodes.popleft()
            marked_nodes[cur_node] = True
            neighbors = list(graph.neighbors(cur_node))
            mask_count = int(mask_ratio * len(neighbors))
            masked, un_masked = neighbors[: mask_count] , neighbors[mask_count: ]
            mask.extend(masked)
            for node in masked:
                marked_nodes[node] = True
            for node_unmask in un_masked:
                if not marked_nodes[node_unmask]:
                    pot_nodes.append(node_unmask)
        r1, r2 = list(sorted(set(mask))), list(sorted(set(range(city)) - set(mask)))
        torch.cuda.empty_cache()
        mask_ratio = (len(r1)/city)
        del graph
        return r1, r2, mask_ratio


class KnnMask(SpatialMaskGenerator):

    def forward(self, graph_adj=None):
        self.masked_tokens, self.unmasked_tokens, mask_ratio = KnnUtil.knn_masking(graph_adj, self.city, self.mask_ratio)
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        self.mask_ratio = mask_ratio
        return self.unmasked_tokens, self.masked_tokens