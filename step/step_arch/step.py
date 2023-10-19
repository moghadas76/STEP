import torch
from torch import nn
from functools import lru_cache
from .tsformer import TSFormer
from .graphwavenet import GraphWaveNet
from .tsformer.positional_encoding import PositionalEncoding_Decoder
from .discrete_graph_learning import DiscreteGraphLearning
from basicts.utils import load_pkl


class PatchEmbedding_2(nn.Module):
    """Patchify time series."""

    def __init__(self, patch_size, in_channel, embed_dim, norm_layer):
        super().__init__()
        self.output_channel = embed_dim
        self.len_patch = 1  # the L
        self.input_channel = in_channel
        self.output_channel = embed_dim
        self.input_embedding = nn.Conv2d(
            in_channel,
            embed_dim,
            kernel_size=(self.len_patch, 1),
            stride=(self.len_patch, 1))
        self.norm_layer = norm_layer if norm_layer is not None else nn.Identity()

    def forward(self, long_term_history):
        """
        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).

        Returns:
            torch.Tensor: patchified time series with shape [B, N, d, P]
        """

        batch_size, num_nodes, num_feat, len_time_series = long_term_history.shape
        long_term_history = long_term_history.unsqueeze(-1)  # B, N, C, L, 1
        # B*N,  C, L, 1
        long_term_history = long_term_history.reshape(batch_size * num_nodes, num_feat, len_time_series, 1)
        # B*N,  d, L/P, 1
        output = self.input_embedding(long_term_history)
        # norm
        output = self.norm_layer(output)
        # reshape
        output = output.squeeze(-1).view(batch_size, num_nodes, self.output_channel, -1)  # B, N, d, P
        assert output.shape[-1] == len_time_series / self.len_patch
        return output


class STEP(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, dataset_name, pre_trained_tsformer_path, tsformer_args, backend_args, dgl_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tsformer_path = pre_trained_tsformer_path

        # iniitalize the tsformer and backend models
        self.tsformer = TSFormer(**tsformer_args)
        self.backend = GraphWaveNet(**backend_args)
        # self.p_decoding = PositionalEncoding_Decoder(96, dropout=0.5)
        # self.patch_embd = PatchEmbedding_2(1,1,96,norm_layer=None)
        # decoder = nn.TransformerDecoderLayer(96 , 1)
        # self.decoder = nn.TransformerDecoder(decoder, 1)
        # # # load pre-trained tsformer
        # self.gcn = nn.Conv2d(207,1, kernel_size=(1,1))
        # self.gcn_2 = nn.Conv2d(207,1, kernel_size=(156 + 1, 1))
        # self.trans_gcn = nn.ConvTranspose1d(96, 207, kernel_size=1)
        self.load_pre_trained_model()

        # discrete graph learning
        self.discrete_graph_learning = DiscreteGraphLearning(**dgl_args)

    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tsformer_path)
        self.tsformer.load_state_dict(checkpoint_dict["model_state_dict"])
        # freeze parameters
        for param in self.tsformer.parameters():
            param.requires_grad = False

    @lru_cache(maxsize=255)
    def get_knn(self, src, k):
        import networkx as nx
        from operator import itemgetter
        def knn(graph: nx.Graph, node, n):
            return list(map(
                itemgetter(1),
                sorted(
                    [(e[2]['weight'], e[1]) for e in graph.edges(node, data=True)])[:n]
            )
            )
        adj = load_pkl("/home/seyed/PycharmProjects/step/STEP/datasets/METR-LA/adj_mx.pkl")
        graph_obj: nx.Graph = nx.from_numpy_array(adj[2])
        graph_obj = graph_obj.subgraph(nodes=next(nx.connected_components(graph_obj)))
        cands = knn(graph_obj, src, k)
        return cands

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """Feed forward of STEP.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]
            future_data (torch.Tensor): future data
            batch_seen (int): number of batches that have been seen
            epoch (int): number of epochs

        Returns:
            torch.Tensor: prediction with shape [B, N, L].
            torch.Tensor: the Bernoulli distribution parameters with shape [B, N, N].
            torch.Tensor: the kNN graph with shape [B, N, N], which is used to guide the training of the dependency graph.
        """
        # reshape
        short_term_history = history_data  # [B, L, N, 1]
        long_term_history = long_history_data

        # STEP
        batch_size, _, num_nodes, _ = short_term_history.shape

        # discrete graph learning & feed forward of TSFormer
        k, src = kwargs.get("k"), kwargs.get("src")
        # enhancing downstream STGNNs

        if k and src >= 0:
            cands = self.get_knn(src, k)
            long_term_history[:, :, cands, :] = 1e-8

        # hidden_states -> torch.Size([B, N, #PATCHes, d])
        # sampled_adj -> torch.Size([B, N, N])
        # adj_knn -> torch.Size([B, N, N])
        # hidden_states -> torch.Size([B, N, #PATCHes, d])
        bernoulli_unnorm, hidden_states, adj_knn, sampled_adj = self.discrete_graph_learning(long_term_history,
                                                                                             self.tsformer)

        # import remote_pdb;
        # remote_pdb.set_trace()
        hidden_states = hidden_states[:, :, -1, :]  # Original

        # y_hat => ([16, 12, 207])
        y_hat = self.backend(short_term_history, hidden_states=hidden_states, sampled_adj=sampled_adj).transpose(1, 2)

        # f_data = future_data.permute(0, 2, 3, 1)
        # f_data = f_data[:,:, [0], :]
        # patches = self.patch_embd(f_data)  # B, N, d, P
        # patches = patches.transpose(-1, -2)  # B, N, P, d
        # # positional embedding
        # B8,N207, T12, D96 = patches.size()
        # patches = patches.reshape(B8, T12, N207, D96)
        # patches = self.p_decoding(patches)
        # # patches = patches.reshape(8, 12, 207 * 96)
        # # hidden_states_re = hidden_states.reshape(8, 207*96)
        #
        # patches = self.gcn(patches)
        # hidden_states_re = self.gcn_2(hidden_states)
        # y_hat = self.decoder(patches.squeeze(1), hidden_states_re.squeeze(1))
        # y_hat = self.trans_gcn(y_hat.transpose(1,2))
        # y_hat = y_hat.transpose(1,2)
        # graph structure loss coefficient
        if epoch is not None:
            gsl_coefficient = 1 / (int(epoch / 6) + 1)
        else:
            gsl_coefficient = 0
        # print(y_hat.shape, "77777")
        return y_hat.unsqueeze(-1), bernoulli_unnorm.softmax(-1)[..., 0].clone().reshape(batch_size, num_nodes,
                                                                                         num_nodes), adj_knn, gsl_coefficient
