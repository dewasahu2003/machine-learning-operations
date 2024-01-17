import logging

import pandas as pd
from zenml import step

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

import mlflow
from zenml.client import Client

client = Client()
client.activate_stack(stack_name_id_or_prefix="mlflow_stack")
experiment_tracker = client.active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Trains the model on the ingested data

    Args:
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ModelNameConfig
    Returns:


    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()  # auto logging model
            model = LinearRegressionModel()
            trained_model = model.train(X_train=X_train, y_train=y_train)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error in creating model:{e}")
        raise e
