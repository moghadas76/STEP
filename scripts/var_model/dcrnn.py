import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.var_model import VAR
from multiprocessing import Pool
from typing import Tuple, List, TypeVar, Dict, Optional
from scripts.var_model.var_model import load_adj
NODE = TypeVar("NODE")
import numpy as np
import matplotlib.pyplot as plt


def masked_mse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_mae_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_rmse_tf(preds, labels, null_val=np.nan):
    """
    Accuracy with masking.
    :param preds:
    :param labels:
    :param null_val:
    :return:
    """
    return tf.sqrt(masked_mse_tf(preds=preds, labels=labels, null_val=null_val))


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
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


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# Builds loss function.
def masked_mse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_mse_tf(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        return masked_rmse_tf(preds=preds, labels=labels, null_val=null_val)

    return loss


def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = masked_mae_tf(preds=preds, labels=labels, null_val=null_val)
        return mae

    return loss


def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    mape = masked_mape_np(preds=df_pred.values, labels=df_test.values, null_val=null_val)
    mae = masked_mae_np(preds=df_pred.values, labels=df_test.values, null_val=null_val)
    rmse = masked_rmse_np(preds=df_pred.values, labels=df_test.values, null_val=null_val)
    return mae, mape, rmse
import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf

from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters


def load_dataset(dataset_dir, batch_size, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data




def historical_average_predict(df, period=12 * 24 * 7, test_ratio=0.2, null_val=0.):
    """
    Calculates the historical average of sensor reading.
    :param df:
    :param period: default 1 week.
    :param test_ratio:
    :param null_val: default 0.
    :return:
    """
    n_sample, n_sensor = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    y_test = df[-n_test:]
    y_predict = pd.DataFrame.copy(y_test)

    for i in range(n_train, min(n_sample, n_train + period)):
        inds = [j for j in range(i % period, n_train, period)]
        historical = df.iloc[inds, :]
        y_predict.iloc[i - n_train, :] = historical[historical != null_val].mean()
    # Copy each period.
    for i in range(n_train + period, n_sample, period):
        size = min(period, n_sample - i)
        start = i - n_train
        y_predict.iloc[start:start + size, :] = y_predict.iloc[start - period: start + size - period, :].values
    return y_predict, y_test


def static_predict(df, n_forward, test_ratio=0.2):
    """
    Assumes $x^{t+1} = x^{t}$
    :param df:
    :param n_forward:
    :param test_ratio:
    :return:
    """
    test_num = int(round(df.shape[0] * test_ratio))
    y_test = df[-test_num:]
    y_predict = df.shift(n_forward).iloc[-test_num:]
    return y_predict, y_test


def var_predict(df, n_forwards=(1, 3), n_lags=3, test_ratio=0.2):
    """
    Multivariate time series forecasting using Vector Auto-Regressive Model.
    :param df: pandas.DataFrame, index: time, columns: sensor id, content: data.
    :param n_forwards: a tuple of horizons.
    :param n_lags: the order of the VAR model.
    :param test_ratio:
    :return: [list of prediction in different horizon], dt_test
    """
    n_sample, n_output = df.shape
    n_test = int(round(n_sample * test_ratio))
    n_train = n_sample - n_test
    df_train, df_test = df[:n_train], df[n_train:]
    scaler = StandardScaler(mean=df_train.values.mean(), std=df_train.values.std())
    data = scaler.transform(df_train.values)
    var_model = None
    try:
        var_model = VAR(data)
    except:
        return None, None, var_model
    var_result = var_model.fit(n_lags, ic="fpe")
    max_n_forwards = np.max(n_forwards)
    # n_forwards : [1, 3, 6, 12]
    # Do forecasting.
    result = np.zeros(shape=(len(n_forwards), n_test, n_output))
    start = n_train - n_lags - max_n_forwards + 1
    for input_ind in range(start, n_sample - n_lags):
        prediction = var_result.forecast(scaler.transform(df.values[input_ind: input_ind + n_lags]), max_n_forwards)
        for i, n_forward in enumerate(n_forwards):
            result_ind = input_ind - n_train + n_lags + n_forward - 1
            if 0 <= result_ind < n_test:
                result[i, result_ind, :] = prediction[n_forward - 1, :]

    df_predicts = []
    for i, n_forward in enumerate(n_forwards):
        df_predict = pd.DataFrame(scaler.inverse_transform(result[i]), index=df_test.index, columns=df_test.columns)
        df_predicts.append(df_predict)
    return df_predicts, df_test, var_result


def eval_static(traffic_reading_df):
    logger.info('Static')
    horizons = [1, 3, 6, 12]
    logger.info('\t'.join(['Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in horizons:
        y_predict, y_test = static_predict(traffic_reading_df, n_forward=horizon, test_ratio=0.2)
        rmse = masked_rmse_np(preds=y_predict.values, labels=y_test.values, null_val=0)
        mape = masked_mape_np(preds=y_predict.values, labels=y_test.values, null_val=0)
        mae = masked_mae_np(preds=y_predict.values, labels=y_test.values, null_val=0)
        line = 'Static\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_historical_average(traffic_reading_df, period):
    y_predict, y_test = historical_average_predict(traffic_reading_df, period=period, test_ratio=0.2)
    rmse = masked_rmse_np(preds=y_predict.values, labels=y_test.values, null_val=0)
    mape = masked_mape_np(preds=y_predict.values, labels=y_test.values, null_val=0)
    mae = masked_mae_np(preds=y_predict.values, labels=y_test.values, null_val=0)
    logger.info('Historical Average')
    logger.info('\t'.join(['Node ID','Model', 'Horizon', 'RMSE', 'MAPE', 'MAE']))
    for horizon in [1, 3, 6, 12]:
        line = 'HA\t%d\t%.2f\t%.2f\t%.2f' % (horizon, rmse, mape * 100, mae)
        logger.info(line)


def eval_var(traffic_reading_df, n_lags=3, node_id=0):
    n_forwards = [1, 3, 6, 12]
    y_predicts, y_test, model = var_predict(traffic_reading_df, n_forwards=n_forwards, n_lags=3,
                                     test_ratio=0.2)
    if not y_predicts:
        return -1, -1, -1, None, node_id
    logger.info('VAR (lag=%d)' % n_lags)
    logger.info('Node ID\tModel\tHorizon\tRMSE\tMAPE\tMAE')
    for i, horizon in enumerate(n_forwards):
        # breakpoint()
        rmse = masked_rmse_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mape = masked_mape_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        mae = masked_mae_np(preds=y_predicts[i].values, labels=y_test.values, null_val=0)
        line = 'Node\t%d VAR\t%d\t%.2f\t%.2f\t%.2f\n' % (node_id, horizon, rmse, mape * 100, mae)
        logger.info(line)
    return mae, mape, rmse, model, node_id

def post_processsing(result):
    df = pd.DataFrame(result, columns=["mae", "mape", "rmse"])
    # df.hist(bins=10, column="mae", figsize=(20, 6))
    df.to_csv("/home/seyed/PycharmProjects/step/STEP/checkpoints/var_model/results/res_2_hop.csv")


def train(data: pd.DataFrame, node_count: int,
          neighbours: Dict[NODE, Dict[str, Dict[str, List[NODE]]]], n_lags=12):
    def second_extractor(node_id: int):
        ls = []
        seconds = set()
        one_hop = neighbours[node_id]["1_hop"]["nodes"]
        for nd in one_hop:
            seconds.update(neighbours[nd]["1_hop"]["nodes"])
        ls.extend(list(seconds - set(one_hop)))
        return ls

    node_ids = list(range(node_count))
    data_repo = []
    for node_id in node_ids:
        data_repo.append((
            # data[list(map(int, neighbours[node_id]["1_hop"]["nodes"]))],
            data[second_extractor(node_id)],
            n_lags,
            node_id
        ))
    filterd_data_repo = [data_repo[i] for i in [47, 148, 127, 56, 137]]
    results = np.zeros((node_count, 3))
    with Pool() as pool:
        result = pool.starmap(eval_var, filterd_data_repo)
        for mae, mape, rmse, model, node_id in result:
            # print("saving the result for the node id", node_id)
            if model:
                results[node_id] = np.array([mae, mape, rmse])
                model.save(f"/home/seyed/PycharmProjects/step/STEP/checkpoints/var_model/preds/predictor_node_{node_id}_mae_{mae}.pkl")
            else:
                subprocess.check_output(f"touch /home/seyed/PycharmProjects/step/STEP/checkpoints/var_model/preds/{node_id}_{mae}.pkl", shell=True, text=True)
            # print("Result", result)
    post_processsing(results)


def analysis(csv_path):
    df = pd.read_csv(csv_path)
    df.hist(bins=10, column="mae",  figsize=(20, 6))
    plt.show()
    df_sorted = df.sort_values(by="mae")
    ranks = df_sorted.index.tolist()
    rank_map = {ind: ranks[ind] for ind, _ in enumerate(ranks)}
    with open("/home/seyed/PycharmProjects/step/STEP/checkpoints/var_model/results/rank_map.json", "w") as file:
        json.dump(rank_map, file)
    df_sorted.plot(kind='bar', y='mae', legend=False)
    plt.xlabel('node id')
    plt.ylabel('MAE')
    plt.title('Bar Chart - Sorted by MAE')
    plt.show()



def main(args):
    df = np.load("/home/seyed/PycharmProjects/step/STEP/datasets/raw_data/METR-LA/metr.npz")
    traffic_reading_df = pd.DataFrame(df["x"], pd.DatetimeIndex(df["z"].astype('datetime64[ns]'), freq="5T"))
    # eval_static(traffic_reading_df)
    # eval_historical_average(traffic_reading_df, period=7 * 24 * 12)
    neighbours, _, rw, _ = load_adj()
    if args.train and not args.analysis:
        train(traffic_reading_df, len(neighbours), neighbours, n_lags=12)
    else:
        analysis(args.csv)
    # eval_var(traffic_reading_df, n_lags=3, node_id=0)


if __name__ == '__main__':
    logger = get_logger('data/model', 'Baseline')
    parser = argparse.ArgumentParser()
    parser.add_argument('--traffic_reading_filename', default="data/metr-la.h5", type=str,
                        help='Path to the traffic Dataframe.')
    parser.add_argument('--train', default=True, type=bool,
                        help='Training')
    parser.add_argument('--analysis', default=False, type=bool,
                        help='Analysing')
    parser.add_argument('--csv', default="", type=str,
                        help='CSV result file')

    args = parser.parse_args()
    main(args)