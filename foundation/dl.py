import os
import urllib
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import torch
import numpy as np
from typing import Sequence, Union
from torch_geometric.data import Data
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

Edge_Index = Union[np.ndarray, None]
Edge_Weight = Union[np.ndarray, None]
Node_Features = Sequence[Union[np.ndarray, None]]
Targets = Sequence[Union[np.ndarray, None]]
Additional_Features = Sequence[np.ndarray]


class StaticGraphTemporalSignal(object):
    r"""A data iterator object to contain a static graph with a dynamically
    changing constant time difference temporal feature set (multiple signals).
    The node labels (target) are also temporal. The iterator returns a single
    constant time difference temporal snapshot for a time period (e.g. day or week).
    This single temporal snapshot is a Pytorch Geometric Data object. Between two
    temporal snapshots the features and optionally passed attributes might change.
    However, the underlying graph is the same.

    Args:
        edge_index (Numpy array): Index tensor of edges.
        edge_weight (Numpy array): Edge weight tensor.
        features (Sequence of Numpy arrays): Sequence of node feature tensors.
        targets (Sequence of Numpy arrays): Sequence of node label (target) tensors.
        **kwargs (optional Sequence of Numpy arrays): Sequence of additional attributes.
    """

    def __init__(
        self,
        edge_index: Edge_Index,
        edge_weight: Edge_Weight,
        features: Node_Features,
        targets: Targets,
        **kwargs: Additional_Features
    ):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.features = features
        self.targets = targets
        self.additional_feature_keys = []
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.additional_feature_keys.append(key)
        self._check_temporal_consistency()
        self._set_snapshot_count()

    def _check_temporal_consistency(self):
        assert len(self.features) == len(
            self.targets
        ), "Temporal dimension inconsistency."
        for key in self.additional_feature_keys:
            assert len(self.targets) == len(
                getattr(self, key)
            ), "Temporal dimension inconsistency."

    def _set_snapshot_count(self):
        self.snapshot_count = len(self.features)

    def _get_edge_index(self):
        if self.edge_index is None:
            return self.edge_index
        else:
            return torch.LongTensor(self.edge_index)

    def _get_edge_weight(self):
        if self.edge_weight is None:
            return self.edge_weight
        else:
            return torch.FloatTensor(self.edge_weight)

    def _get_features(self, time_index: int):
        if self.features[time_index] is None:
            return self.features[time_index]
        else:
            return torch.FloatTensor(self.features[time_index])

    def _get_target(self, time_index: int):
        if self.targets[time_index] is None:
            return self.targets[time_index]
        else:
            if self.targets[time_index].dtype.kind == "i":
                return torch.LongTensor(self.targets[time_index])
            elif self.targets[time_index].dtype.kind == "f":
                return torch.FloatTensor(self.targets[time_index])

    def _get_additional_feature(self, time_index: int, feature_key: str):
        feature = getattr(self, feature_key)[time_index]
        if feature.dtype.kind == "i":
            return torch.LongTensor(feature)
        elif feature.dtype.kind == "f":
            return torch.FloatTensor(feature)

    def _get_additional_features(self, time_index: int):
        additional_features = {
            key: self._get_additional_feature(time_index, key)
            for key in self.additional_feature_keys
        }
        return additional_features

    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = StaticGraphTemporalSignal(
                self.edge_index,
                self.edge_weight,
                self.features[time_index],
                self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index()
            edge_weight = self._get_edge_weight()
            y = self._get_target(time_index)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, **additional_features)
        return snapshot

    def __next__(self):
        if self.t < len(self.features):
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration

    def __iter__(self):
        self.t = 0
        return self


class METRLADatasetLoader(object):
    """A traffic forecasting dataset based on Los Angeles
    Metropolitan traffic conditions. The dataset contains traffic
    readings collected from 207 loop detectors on highways in Los Angeles
    County in aggregated 5 minute intervals for 4 months between March 2012
    to June 2012.

    For further details on the version of the sensor network and
    discretization see: `"Diffusion Convolutional Recurrent Neural Network:
    Data-Driven Traffic Forecasting" <https://arxiv.org/abs/1707.01926>`_
    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(METRLADatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self._read_web_data()

    def _download_url(self, url, save_path):  # pragma: no cover
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _read_web_data(self):
        url = "https://graphmining.ai/temporal_datasets/METR-LA.zip"

        # Check if zip file is in data folder from working directory, otherwise download
        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "METR-LA.zip")
        ):  # pragma: no cover
            if not os.path.exists(self.raw_data_dir):
                os.makedirs(self.raw_data_dir)
            # self._download_url(url, os.path.join(self.raw_data_dir, "METR-LA.zip"))

        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "adj_mat.npy")
        ) or not os.path.isfile(
            os.path.join(self.raw_data_dir, "node_values.npy")
        ):  # pragma: no cover
            with zipfile.ZipFile(
                os.path.join(self.raw_data_dir, "METR-LA.zip"), "r"
            ) as zip_fh:
                zip_fh.extractall(self.raw_data_dir)

        A = np.load(os.path.join(self.raw_data_dir, "adj_mat.npy"))
        X = np.load(os.path.join(self.raw_data_dir, "node_values.npy")).transpose(
            (1, 2, 0)
        )
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for METR-LA dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
    

###########################
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import ChebConv
# class TemporalConv(nn.Module):
#     r"""Temporal convolution block applied to nodes in the STGCN Layer
#     For details see: `"Spatio-Temporal Graph Convolutional Networks:
#     A Deep Learning Framework for Traffic Forecasting."
#     <https://arxiv.org/abs/1709.04875>`_ Based off the temporal convolution
#     introduced in "Convolutional Sequence to Sequence Learning"  <https://arxiv.org/abs/1709.04875>`_

#     Args:
#         in_channels (int): Number of input features.
#         out_channels (int): Number of output features.
#         kernel_size (int): Convolutional kernel size.
#     """

#     def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
#         super(TemporalConv, self).__init__()
#         self.conv_1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
#         self.conv_2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
#         self.conv_3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

#     def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
#         """Forward pass through temporal convolution block.

#         Arg types:
#             * **X** (torch.FloatTensor) -  Input data of shape
#                 (batch_size, input_time_steps, num_nodes, in_channels).

#         Return types:
#             * **H** (torch.FloatTensor) - Output data of shape
#                 (batch_size, in_channels, num_nodes, input_time_steps).
#         """
#         X = X.permute(0, 3, 2, 1)
#         P = self.conv_1(X)
#         Q = torch.sigmoid(self.conv_2(X))
#         PQ = P * Q
#         H = F.relu(PQ + self.conv_3(X))
#         H = H.permute(0, 3, 2, 1)
#         return H

# class STConv(nn.Module):
#         r"""Spatio-temporal convolution block using ChebConv Graph Convolutions.
#             bias (bool, optional): If set to :obj:`False`, the layer will not learn
#                 an additive bias. (default: :obj:`True`)

#         """

#         def __init__(
#             self,
#             num_nodes: int,
#             in_channels: int,
#             hidden_channels: int,
#             out_channels: int,
#             kernel_size: int,
#             K: int,
#             normalization: str = "sym",
#             bias: bool = True,
#         ):
#             super(STConv, self).__init__()
#             self.num_nodes = num_nodes
#             self.in_channels = in_channels
#             self.hidden_channels = hidden_channels
#             self.out_channels = out_channels
#             self.kernel_size = kernel_size
#             self.K = K
#             self.normalization = normalization
#             self.bias = bias

#             self._temporal_conv1 = TemporalConv(
#                 in_channels=in_channels,
#                 out_channels=hidden_channels,
#                 kernel_size=kernel_size,
#             )

#             self._graph_conv = ChebConv(
#                 in_channels=hidden_channels,
#                 out_channels=hidden_channels,
#                 K=K,
#                 normalization=normalization,
#                 bias=bias,
#             )

#             self._temporal_conv2 = TemporalConv(
#                 in_channels=hidden_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#             )

#             self._batch_norm = nn.BatchNorm2d(num_nodes)

#         def forward(
#             self,
#             X: torch.FloatTensor,
#             edge_index: torch.LongTensor,
#             edge_weight: torch.FloatTensor = None,
#         ) -> torch.FloatTensor:

#             r"""Forward pass. If edge weights are not present the forward pass
#             defaults to an unweighted graph.

#             Arg types:
#                 * **X** (PyTorch FloatTensor) - Sequence of node features of shape (Batch size X Input time steps X Num nodes X In channels).
#                 * **edge_index** (PyTorch LongTensor) - Graph edge indices.
#                 * **edge_weight** (PyTorch LongTensor, optional)- Edge weight vector.

#             Return types:
#                 * **T** (PyTorch FloatTensor) - Sequence of node features.
#             """
#             T_0 = self._temporal_conv1(X)
#             T = torch.zeros_like(T_0).to(T_0.device)
#             for b in range(T_0.size(0)):
#                 for t in range(T_0.size(1)):
#                     T[b][t] = self._graph_conv(T_0[b][t], edge_index, edge_weight)

#             T = F.relu(T)
#             T = self._temporal_conv2(T)
#             T = T.permute(0, 2, 1, 3)
#             T = self._batch_norm(T)
#             T = T.permute(0, 2, 1, 3)
#             return T


# graph_model = STConv(207,1,32,1,3,10,normalization="sym",bias=True)

# dataset = METRLADatasetLoader().get_dataset(num_timesteps_in=12, num_timesteps_out=12)
# train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# device = torch.device('cuda:0')
# # model = TemporalGNN(node_features=2, periods=12).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# break_step = 2000

# model.train()
# print("Running training...")
# for epoch in range(100): 
#     model.train()
#     loss = 0
#     val_loss = 0
#     step = 0
#     val_step = 0
#     for data in train_dataset:
#       	#----------------------------------#
#         data = data.to(device)
#         X = data.x
#         E = data.edge_index
#         y = data.y
#         #----------------------------------#
#         y_hat = model(X,E) # (207, 12)
#         #----------------------------------#
#         loss += torch.mean((y_hat - y)**2) 
#         step += 1
#         if step > break_step:
#           break

#     loss = loss / (step + 1)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
#     with torch.no_grad():
#         model.eval()
#         for data in test_dataset:
#           	#----------------------------------#
#             data = data.to(device)
#             X = data.x
#             E = data.edge_index
#             y = data.y
#             #----------------------------------#
#             model.eval()
#             y_hat = model(X,E)
#             val_loss += torch.mean((y_hat - y)**2) 
#             val_step += 1
#             if val_step > break_step:
#                 print("Epoch {} val MSE: {:.4f}".format(epoch, val_loss.item()))
#                 torch.save(model.state_dict(), "model.pt")
#                 break
