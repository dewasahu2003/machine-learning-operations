from zenml import pipelines
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.evaluation import eval_model


@pipelines
def train_pipeline(data_path:str):

    """
    Training Pipeline 

    Args:
        data_path: take the path to the data
    """
    data= ingest_data("path")
    clean_data = clean_data(data)
    model = model_train(clean_data)
    evaluation = eval_model(model)