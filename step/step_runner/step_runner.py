import time

import torch

from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape


class STEPRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.metrics = cfg.get("METRICS", {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})
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
        future_data, history_data, long_history_data = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        long_history_data   = self.to_running_device(long_history_data)       # B, L, N, C
        future_data         = self.to_running_device(future_data)       # B, L, N, C

        history_data = self.select_input_features(history_data)
        long_history_data = self.select_input_features(long_history_data)

        # feed forward
        prediction, pred_adj, prior_adj, gsl_coefficient, query, pos, neg = self.model(
            history_data=history_data,
            long_history_data=long_history_data,
            future_data=future_data,
            batch_seen=iter_num,
            epoch=epoch,
            src=kwargs.get("src"),
            k=kwargs.get("k")
        )

        batch_size, length, num_nodes, _ = future_data.shape
        assert list(prediction.shape)[:3] == [batch_size, length, num_nodes], \
            "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"

        # post process
        prediction = self.select_target_features(prediction)
        real_value = self.select_target_features(future_data)
        return prediction, real_value, pred_adj, prior_adj, gsl_coefficient, query, pos, neg

    # def test_process(self, cfg, train_epoch: int = None):
    #     if train_epoch is None:
    #         self.init_test(cfg)
    #
    #     self.on_test_start()
    #
    #     test_start_time = time.time()
    #     self.model.eval()
    #
    #     # test
    #     self.test(k=k, src=src)
    #
    #     test_end_time = time.time()
    #     self.update_epoch_meter("test_time", test_end_time - test_start_time)
    #     # print test meters
    #     self.print_epoch_meters("test")
    #     if train_epoch is not None:
    #         # tensorboard plt meters
    #         self.plt_epoch_meters("test", train_epoch // self.test_interval)