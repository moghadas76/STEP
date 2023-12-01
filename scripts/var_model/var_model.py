import copy
import random
import more_itertools as mit
import numpy as np
import pandas as pd
import subprocess
import tensorflow as tf, os, sys
import matplotlib.pyplot as plt
import networkx as nx
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VARResults
from multiprocessing import Pool
from typing import Tuple, List, TypeVar, Dict, Optional

from scripts.var_model.util import RandomWalkUti

NODE = TypeVar("NODE")

sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import re_standard_transform


def masked_mae_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    # preds = preds.values
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def fit_var_model(data: pd.DataFrame, test_data: pd.DataFrame, hold_out: pd.DataFrame, target_node_id: int, mean: float= 0.0, std: float=0.0) -> Tuple[
    Optional[VARResults], int | str]:
    # index = data.index
    # data.index.normalize()
    WINDOW = 12
    HORIZON = 12
    # for index, batch in enumerate(mit.chunked(test_data, WINDOW)):
    model = None

    # Forecast the next 10 time steps
    n_lags = 12
    max_n_forwards = 12
    n_test = len(test_data) + len(hold_out)
    n_forwards = [1, 3, 6, 12]
    n_output = 207
    n_sample = len(test_data) + len(hold_out) + len(data)
    n_train = n_sample - n_test
    try:
        # breakpoint()
        model = VAR(data[:n_sample - int(round(n_sample * 0.2))])
    except ValueError:
        return None, str(str(target_node_id) + "__Isolated")

    model_fitted = model.fit(maxlags=12, trend="ctt", verbose=True)
    # Convert the forecasted values to a DataFrame
    df = pd.concat([data, test_data, hold_out])
    print((len(n_forwards), n_test, n_output))
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    print(start, n_sample)
    for input_ind in range(start, n_sample - n_lags):
        prediction = model_fitted.forecast(df.values[input_ind: input_ind + n_lags], max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            # print(result_ind)
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    # Evaluate the model performance
    test = pd.concat([test_data, hold_out])
    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(re_standard_transform(result[i], mean=mean, std=std), index=test.index, columns=test.columns)
        df_predicts.append(df_predict)

    for steps in range(len(n_forwards)):
        # breakpoint()
        # rmse_result = rmse(re_standard_transform(test, mean=mean, std=std).values, df_predicts[steps].values)
        mae_result = masked_mae_tf(df_predicts[steps].values, re_standard_transform(test, mean=mean, std=std).values, 0)
        # print(f"Horizon {n_forwards[steps]} : Validatipon RMSE: {rmse_result}", "<------> Node ID:", target_node_id)
        print(f"Horizon {n_forwards[steps]} : Validatipon MAE: {mae_result}", "<------> Node ID:", target_node_id)

    # new_data = pd.concat([data, test_data])
    # model = VAR(new_data)
    # model_fitted = model.fit(maxlags=12, trend="ctt", verbose=True)
    #
    # # Forecast the next 10 time steps
    # forecast = model_fitted.forecast(new_data.values, steps=len(hold_out))
    #
    # # Convert the forecasted values to a DataFrame
    # forecast_df = re_standard_transform(pd.DataFrame(forecast, columns=data.columns), mean=mean, std=std)
    # # Evaluate the model performance
    #
    # rmse_result = rmse(re_standard_transform(hold_out, mean=mean, std=std), forecast_df)
    # mae_result = mean_absolute_error(re_standard_transform(hold_out, mean=mean, std=std), forecast_df)
    #
    # print(f"Test RMSE: {rmse_result}", "<------> Node ID:", target_node_id)
    # print(f"Test MAE: {mae_result}", "<------> Node ID:", target_node_id)

    return model_fitted, target_node_id


def train(data: pd.DataFrame, data_test: pd.DataFrame, hold_out: pd.DataFrame, node_count: int,
          neighbours: Dict[NODE, Dict[str, Dict[str, List[NODE]]]]):
    node_ids = list(range(node_count))
    data_repo = []
    for node_id in node_ids:
        data_repo.append((
            data[list(map(str, neighbours[node_id]["1_hop"]["nodes"]))],
            data_test[list(map(str, neighbours[node_id]["1_hop"]["nodes"]))],
            hold_out[list(map(str, neighbours[node_id]["1_hop"]["nodes"]))],
            node_id))
    with Pool() as pool:
        result = pool.starmap(fit_var_model, data_repo)
        for model, node_id in result:
            # print("saving the result for the node id", node_id)
            if model:
                model.save(f"/home/seyed/PycharmProjects/step/STEP/checkpoints/var_model/predictor_node_{node_id}.pkl")
            else:
                subprocess.check_output(f"touch /home/seyed/PycharmProjects/step/STEP/checkpoints/var_model/{node_id}.pkl", shell=True, text=True)
            print("Result", result)
    # fit_var_model(data, data_test, data_valid, 0, mean=54.45524850080463, std=19.514737115784587)


def load_dataset(output_dir="/home/seyed/PycharmProjects/step/STEP/datasets/raw_data/METR-LA") -> Tuple[
    pd.DataFrame, ...]:
    history_seq_len = 2016
    future_seq_len = 12
    from basicts.data.transform import standard_transform
    # df = pd.read_hdf("/home/seyed/PycharmProjects/step/STEP/datasets/raw_data/METR-LA/METR-LA.h5")
    df = np.load("/home/seyed/PycharmProjects/step/STEP/datasets/raw_data/METR-LA/metr.npz")
    data = np.expand_dims(df["x"], axis=-1)

    data = data[..., [0]]
    print("raw time series shape: {0}".format(data.shape))

    l, n, f = data.shape
    num_samples = l
    # keep same number of validation and test samples with Graph WaveNet (input 12, output 12)
    test_num_short = 3429
    valid_num_short = 3425
    train_num_short = num_samples - valid_num_short - test_num_short
    # train_num_short = round(num_samples * train_ratio)
    # valid_num_short = round(num_samples * valid_ratio)
    # test_num_short = num_samples - train_num_short - valid_num_short
    print("number of training samples:{0}".format(train_num_short))
    print("number of validation samples:{0}".format(valid_num_short))
    print("number of test samples:{0}".format(test_num_short))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t - history_seq_len, t, t + future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index = index_list[train_num_short +
                            valid_num_short: train_num_short + valid_num_short + test_num_short]

    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)
    shape_before_flatten = data_norm.shape
    data_norm = data_norm.reshape((shape_before_flatten[0], shape_before_flatten[1]))
    data_norm = pd.DataFrame(data_norm, pd.DatetimeIndex(df["z"].astype('datetime64[ns]'), freq="5T"))
    data_norm.columns = [f"{i}" for i in range(shape_before_flatten[1])]
    # add external feature
    feature_list = [data_norm]
    return data_norm.iloc[: train_num_short, :], data_norm.iloc[train_num_short: train_num_short + valid_num_short, :], \
        data_norm.iloc[train_num_short +
                       valid_num_short: train_num_short + valid_num_short + test_num_short, :]


def load_adj(show=False):
    G, net = RandomWalkUti.load_random_walk(
        "/home/seyed/PycharmProjects/step/STEP/datasets/raw_data/METR-LA/adj_METR-LA.pkl")
    orig_g = copy.deepcopy(G)
    if show:
        pos = nx.spring_layout(net)
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'))
        plt.show()
    node_nns = {}
    for node in range(207):
        node_nns[node] = {}
        node_nns[node]["1_hop"] = {}
        sub = G.subgraph(list(G.neighbors(node)))
        node_nns[node]["1_hop"]["nodes"] = list(sub.nodes())
        node_nns[node]["1_hop"]["edges"] = list(sub.edges())
    return node_nns, G, net, orig_g


if __name__ == '__main__':
    node_nns, _, _ = load_adj()
    data_train, data_valid, data_test = load_dataset()
    train(data_train, data_valid, data_test, len(node_nns), node_nns)
