import pandas as pd
import pickle
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.vector_ar.var_model import VARResults
from multiprocessing import Pool
from typing import Tuple, List, TypeVar

NODE = TypeVar("NODE")


def fit_var_model(data: pd.DataFrame, test_data: pd.DataFrame, target_node_id: int) -> Tuple[VARResults, int]:
    model = VAR(data)
    model_fitted = model.fit()

    # Forecast the next 10 time steps
    forecast = model_fitted.forecast(test_data, steps=len(test_data))

    # Convert the forecasted values to a DataFrame
    forecast_df = pd.DataFrame(forecast, columns=data.columns)

    # Evaluate the model performance
    rmse_result = rmse(test_data, forecast_df)
    mae_result = mean_absolute_error(test_data, forecast_df)

    print(f"RMSE: {rmse_result:.2f}")
    print(f"MAE: {mae_result:.2f}")
    return model_fitted, target_node_id


def data_splitter(data, node: NODE, neighbours: List[NODE]) -> Tuple[pd.DataFrame, pd.DataFrame, NODE]:
    pass


def train(data: pd.DataFrame, node_count: int, neighbours: List[List[NODE]]):
    node_ids = list(range(node_count))
    data_repo = []
    for node_id, neighbours in zip(node_ids, neighbours):
        data_repo.append(data_splitter(data[list(map(str, neighbours[node_id]))], node_id, neighbours[node_id]))
    with Pool() as pool:
        result = pool.starmap(fit_var_model, data_repo)
        for model, node_id in result:
            print("saving the result for the node id", node_id)
            pickle.dump(
                f"/home/seyed/PycharmProjects/step/STEP/checkpoints/var_model/predictor_node_{node_id}.pkl",
                model.model.state_dict()
            )
    print("result")