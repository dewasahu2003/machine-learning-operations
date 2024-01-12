import logging
import pandas as pd
from zenml import step


class IngestData:
    """
    Ingesting the data from the data_path
    """

    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        """
        Args:
            Ingesting the data from the data_path
        Returns:
            pd.DataFrame: the ingested fata
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)


@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data_path
    Args:
        data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path=data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        raise e
