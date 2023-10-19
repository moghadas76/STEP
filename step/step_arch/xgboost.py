import xgboost as xgb

xgb.XGBRegressor()

import numpy as np


def masked_mae(preds, labels, null_val: float = np.nan):
    """Masked mean absolute error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    mask = ~np.isnan(labels)
    mask = mask.float()
    mask /= np.mean(mask)
    mask = np.where(np.isnan(mask), np.zeros_like(mask), mask)
    loss = np.abs(preds-labels)
    loss = loss * mask
    return np.mean(loss)

mse = np.mean((actual - predicted) ** 2)
rmse = np.sqrt(mse)

params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

n = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)