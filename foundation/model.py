import torch.nn as nn


class ClipModel(nn.Module):
    """
    Clip model wrapper.
    """

    def __init__(self, graph_model, transformer):
        super().__init__()
        self.graph_model = graph_model
        self.transformer = transformer

    def forward(self, input_ids, adj):
        graph = self.graph_model(adj)
        ts = self.transformer(input_ids)
        return ts, graph
    

