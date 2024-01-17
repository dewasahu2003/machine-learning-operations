from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

client = Client()
client.activate_stack(stack_name_id_or_prefix="mlflow_stack")
experiment_tracker = client.active_stack.experiment_tracker

if __name__ == "__main__":
    print(experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="data/olist_customers_dataset.csv")
