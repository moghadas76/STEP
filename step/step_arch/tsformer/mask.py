import random

from torch import nn
import random
import networkx as nx
# import community  # This is python-louvain library


class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, num_tokens, mask_ratio):
        super().__init__()
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.sort = True

    def uniform_rand(self):
        # Maks: 126
        # Unmasked: 42
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
    

class KNNMaskGenerator(nn.Module):
    """K-Nearest Neighbors (KNN) mask generator based on the input graph."""
    ADJ_PATH = "/home/seyed/PycharmProjects/dashboard/dashboard/src/dashboard/results/adj_6_nn.pkl" 

    def __init__(self, num_tokens, mask_ratio, spatial_ratio, node_count):
        super().__init__()
        import pickle
        with open(self.ADJ_PATH, "rb") as f:
            graph = pickle.load(f)
        self.graph = nx.from_numpy_array(graph[:node_count, :node_count])
        self.mask_ratio = mask_ratio
        self.spatial_ratio = spatial_ratio
        self.k = 2
        self.temporal_masks = MaskGenerator(num_tokens, mask_ratio)
    
    def find_k_hop_neighbors(self, node, k):
        if node not in self.graph:
            return []  # Node not in the graph

        if k == 0:
            return [node]

        k_hop_neighbors = list(nx.single_source_shortest_path_length(self.graph, node, cutoff=k).keys())
        # k_hop_neighbors.remove(node)  # Remove the node itself if it's in the list

        return k_hop_neighbors

    def generate_mask(self):
        num_nodes = len(self.graph)
        mask_len = int(num_nodes * self.spatial_ratio)
        masked_nodes = []
        for node in self.graph:
            neighbors = self.find_k_hop_neighbors(node, self.k)
            if node not in masked_nodes:
                masked_nodes.append(node)
            for neighbor in neighbors[:int(len(neighbors) * self.spatial_ratio)]:
                if len(masked_nodes) >= mask_len:
                    break
                if neighbor not in masked_nodes:
                    masked_nodes.append(neighbor)
            if len(masked_nodes) >= mask_len:
                break
        return sorted(masked_nodes), sorted(list(set(range(num_nodes)) - set(masked_nodes)))

    def forward(self):
        temporal_masks, temporal_unmasks = self.temporal_masks()
        masked_nodes_spatial, unmasked_nodes_spatial = self.generate_mask()
        return masked_nodes_spatial, unmasked_nodes_spatial, temporal_masks, temporal_unmasks
