from typing import cast
import numpy as np
import pandas as pd
import json

from zenml import pipeline, step
from zenml.config import DockerSettings  # for docker stuff
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from pipelines.utils import get_data_for_test

from steps.clean_data import clean_data
from steps.evaluation import eval_model
from steps.model_train import model_train
from steps.ingest_data import ingest_data


docker_setting = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    """Deployement Trigger Config"""

    min_accuracy: float = 0


# trigger
@step
def deployement_trigger(
    accuracy: float,
    config: DeploymentTriggerConfig,
):
    """Model deployement trigger based on its accuracy"""
    return accuracy >= config.min_accuracy


# continious pipeline
@pipeline(enable_cache=False, settings={"docker": docker_setting})
def continous_deployement_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    """
    CD Pipeline

    Args:
        data_path:str,
        min_accuracy:float
        worker:int
        timeout:int
    """
    data = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(data)
    model = model_train(X_train, y_train)
    r2, rmse, mse = eval_model(model, X_test, y_test)

    # here is the difference DEPLOYEMENT CONDITION based on r2 now
    deployement_decision = deployement_trigger(r2)

    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployement_decision,
        workers=workers,
        timeout=timeout,
    )


# for predicition
@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str, pipeline_step_name: str, model_name: str, running: bool = True
) -> MLFlowDeploymentService:
    """Get the prediction service started by deployement pipeline
    Attributes:
        pipeline_name:str
        step_name:str
        running:bool
        model_name:str
    """
    # get the MLFlow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    # fetch the exisiting service with the same pipeline name,step name,model_name

    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow deployment service found for pipeline {pipeline_name}"
            f"step {pipeline_step_name} and model {model_name}"
            f"pipeline is currently running:{running}"
        )
    return existing_services[0]


@step(enable_cache=True)
def dynamic_importer() -> str:
    """get the data for prediction"""
    data = get_data_for_test()
    return data


@step
def predictor(service: MLFlowDeploymentService, data: str) -> np.ndarray:
    """
    Run an inference request against a prediciton service
    Args:
        service: MLFlowDeploymentService
        data: np.ndarray
    """
    service.start(timeout=10)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]

    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction


@pipeline(enable_cache=False, settings={"docker": docker_setting})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer()
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name="model",
        running=False,
    )
    prediction = predictor(service=service, data=data)
    return prediction
