import torch
from torch import nn
from basicts.utils.serialization import load_adj


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.view(batch_size*num_nodes, num_patches, num_feat)
        # positional encoding
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data


class GraphEncoding(nn.Module):
    """Positional encoding."""

    def load_random_walk(self, path):
        _, adj = load_adj(path, "scalap")
        return self._calculate_random_walk_matrix(torch.Tensor(adj).to(self.device))

    def _calculate_random_walk_matrix(self, adj_mx):

        # tf.Print(adj_mx, [adj_mx], message="This is adj: ")

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0])).to(adj_mx.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(
            d_inv.shape).to(d_inv.device), d_inv)
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def __init__(self, hidden_dim, dropout=0.1, num_nodes: int = 1000, adj_path="/home/seyed/PycharmProjects/step/STEP/datasets/METR-LA/adj_mx.pkl"):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(num_nodes, hidden_dim), requires_grad=True)
        self.device = self.position_embedding.device
        self.random_walk = self.load_random_walk(path=adj_path)

    def forward(self, input_data, index=None, abs_idx=None, **kwargs):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """
        device = kwargs.get("device")
        assert str(device) in ["cpu", "cuda:0", "cuda:1"]
        batch_size, num_nodes, num_patches, num_feat = input_data.shape
        input_data = input_data.contiguous().view(batch_size*num_patches, num_nodes, num_feat)
        # positional encoding
        if device is not None:
            pe = torch.matmul(self.random_walk.to(device), self.position_embedding).unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        input_data = pe + input_data
        input_data = self.dropout(input_data)
        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data


class PositionalEncoding_Decoder(nn.Module):
    """Positional encoding."""

    def __init__(self, hidden_dim, dropout=0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Parameter(torch.empty(max_len, hidden_dim), requires_grad=True)

    def forward(self, input_data, index=None, abs_idx=None):
        """Positional encoding

        Args:
            input_data (torch.tensor): input sequence with shape [B, N, P, d].
            index (list or None): add positional embedding by index.

        Returns:
            torch.tensor: output sequence
        """

        batch_size, num_patches, num_nodes, num_feat = input_data.shape

        input_data = input_data.view(batch_size*num_nodes, num_patches, num_feat)
        # positional encoding
        if index is None:
            pe = self.position_embedding[:input_data.size(1), :].unsqueeze(0)
        else:
            pe = self.position_embedding[index].unsqueeze(0)
        input_data = input_data + pe
        input_data = self.dropout(input_data)
        # reshape
        input_data = input_data.view(batch_size, num_nodes, num_patches, num_feat)
        return input_data

