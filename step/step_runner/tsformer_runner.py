from easytorch.config import Config
import torch
import torch.nn as nn
from easytorch.utils.dist import master_only
from torch.utils.data import DataLoader
from basicts.data.registry import SCALER_REGISTRY
from basicts.runners import BaseTimeSeriesForecastingRunner
from foundation.model import ClipModel
from foundation.dl import METRLADatasetLoader
from easytorch.core.data_loader import build_data_loader, build_data_loader_ddp
# from torch_geometric.nn import Graph

class TSFormerRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data
    
    def temporal_signal_split(
    self, data_iterator, train_ratio: float = 0.8
    ):
        r"""Function to split a data iterator according to a fixed ratio.

        Arg types:
            * **data_iterator** *(Signal Iterator)* - Node features.
            * **train_ratio** *(float)* - Graph edge indices.

        Return types:
            * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
        """

        # train_snapshots = int(train_ratio * data_iterator.snapshot_count)
        test_num_short = 6850
        valid_num_short = 3425
        train_num_short = len(data_iterator.features) - valid_num_short - test_num_short
        train_iterator = data_iterator[0:test_num_short]
        valid_iterator = data_iterator[train_num_short: train_num_short + valid_num_short]
        test_iterator = data_iterator[train_num_short + valid_num_short:]
        return train_iterator, valid_iterator , test_iterator

    def build_train_dataset(self, cfg: dict):
        graph = METRLADatasetLoader().get_dataset()
        train, _, _ = self.temporal_signal_split(graph)
        return super().build_train_dataset(cfg), train
    
    def build_val_dataset(self, cfg: dict):
        graph = METRLADatasetLoader().get_dataset()
        _, valid, _ = self.temporal_signal_split(graph)
        return super().build_val_dataset(cfg), valid
    
    def build_test_dataset(self, cfg: dict):
        graph = METRLADatasetLoader().get_dataset()
        _, _, test = self.temporal_signal_split(graph)
        return super().build_test_dataset(cfg), test
    
    def build_test_data_loader(self, cfg: dict) -> DataLoader:
        """Support "setup_graph" for the models acting like tensorflow.

        Args:
            cfg (dict): all in one configurations

        Returns:
            DataLoader: train dataloader
        """

        self.logger.info('Building training data loader.')
        dataset = self.build_test_dataset(cfg)
        if torch.distributed.is_initialized():
            return build_data_loader_ddp(dataset, cfg['TRAIN.DATA'])
        else:
            return [build_data_loader(d, cfg['TRAIN.DATA']) for d in dataset]
    # def build_test_data_loader(self, cfg: Config) -> DataLoader:
    #     graph = METRLADatasetLoader().get_dataset()
    #     _, _, test = self.temporal_signal_split(graph)
    #     return super().build_test_data_loader(cfg), test
    
    def define_model(self, cfg: dict) -> nn.Module:
        self.transformer = super().define_model(cfg)
        from torch_geometric_temporal.nn.recurrent import A3TGCN
        import torch.nn.functional as F

        class TemporalGNN(torch.nn.Module):
            def __init__(self, node_features, input_periods, output_periods):
                super(TemporalGNN, self).__init__()
                # node_features = 2 ( speed & time )
                # periods = 12 ( 향후 12 step을 예측 )
                self.tgnn = A3TGCN(in_channels=node_features, 
                                out_channels=96, 
                                periods=input_periods)
                # single-shot prediction

            def forward(self, x, edge_index):
                # x 크기 : (207, 2, 12)
                # edge_index 크기 : (2, 1722)
                h = self.tgnn(x, edge_index)
                # h 크기 : (207, 32)
                h = F.relu(h)
                # h 크기 : (207, 12)
                return h
        
        self.graph_model = TemporalGNN(node_features=2, 
            input_periods=12,
            output_periods=12)
        return self.transformer, self.graph_model

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch:int = None, iter_num: int = None, train:bool = True, **kwargs) -> tuple:
        """feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        # breakpoint()
        future_data, history_data, graph = data
        history_data    = self.to_running_device(history_data)      # B, L, N, C
        future_data     = self.to_running_device(future_data)       # B, L, N, C
        graph_data     = self.to_running_device(graph)       # B, L, N, C
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)

        # feed forward
        reconstruction_masked_tokens, label_masked_tokens, hideen_state, mask_index = self.transformer(history_data=history_data, future_data=None, batch_seen=iter_num, epoch=epoch)
        g = self.graph_model(graph_data.x, graph_data.edge_index)
        g = g.expand(batch_size, *g.shape)
        graph_unmasked = g[:, list(set(list(range(num_nodes))) - set(mask_index)), :]
        # assert list(prediction_data.shape)[:3] == [batch_size, length, num_nodes], \
            # "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        # prediction = self.select_target_features(prediction_data)
        # real_value = self.select_target_features(future_data)
        return reconstruction_masked_tokens, label_masked_tokens, hideen_state, graph_unmasked

    @torch.no_grad()
    @master_only
    def test(self, train_epoch):
        """Evaluate the model.

        Args:
            train_epoch (int, optional): current epoch if in training process.
        """
        for _, (data, graph) in enumerate(zip(*self.test_data_loader)):
            forward_return = self.forward(data=(*data, graph), epoch=None, iter_num=None, train=False)
            # re-scale data
            prediction_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[0], **self.scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(self.scaler["func"])(forward_return[1], **self.scaler["args"])
            # metrics
            for metric_name, metric_func in self.metrics.items():
                metric_item = metric_func(prediction_rescaled, real_value_rescaled, null_val=self.null_val)
                self.update_epoch_meter("test_"+metric_name, metric_item.item())
