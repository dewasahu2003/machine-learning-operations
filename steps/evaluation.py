import logging


import pandas as pd
from zenml import step

from src.evaluation import MSE, R2Score, RMSE
from sklearn.base import RegressorMixin

from typing import Tuple
from typing_extensions import Annotated

import mlflow
from zenml.client import Client

client = Client()
client.activate_stack(stack_name_id_or_prefix="mlflow_stack")
experiment_tracker = client.active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def eval_model(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
    Annotated[float, "mse"],
]:
    """
    Evalutes model on the ingested data
    Args:
        model:RegressorMixin,
        X_test:pd.DataFrame,
        y_test:pd.Series
    Returns:
        r2: float | ndarray,
        rmse: float | ndarray,
        mse: float | ndarray,
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_sources(y_test, prediction)
        mlflow.log_metric("mse", mse)  # logging the metrics

        r2_class = R2Score()
        r2 = r2_class.calculate_sources(y_test, prediction)
        mlflow.log_metric("r2 score", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_sources(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse, mse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e
