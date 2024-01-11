import logging

import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning,DataDivideStrategy, DataPreProcessingStrategy

from typing import Tuple
from typing_extensions import Annotated



@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],

]:
    """
    Cleaning the Raw data

    Args:
        df: Raw Data
    Returns:
        X_train: Training Data
        X_test: Testing Data
        y_train: Training Label
        y_test: Testing Label

    """
    try:
        preprocess_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaning(data=df,strategy=preprocess_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data,divide_strategy)

        X_train,X_test,y_train,y_test = data_cleaning.handle_data()
        logging.info("Data Cleaning Completed")
    except Exception as e:
        logging.error(f"Error in cleaning data:{e}")
        raise e



