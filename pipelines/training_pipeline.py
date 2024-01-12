from zenml import pipelines
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.evaluation import eval_model


@pipelines
def train_pipeline(data_path: str):
    """
    Training Pipeline

    Args:
        data_path: take the path to the data
    """
    data = ingest_data("path")
    X_train, X_test, y_train, y_test=clean_data(data)
    model = model_train(X_train, X_test, y_train, y_test)
    r2, rmse, mse = eval_model(model,X_test,y_test)

