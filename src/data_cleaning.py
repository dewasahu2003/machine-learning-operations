import logging

from abc import ABC, abstractmethod  # for abstraction
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Data Strategy Abstract Class
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy for preprocessing Data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess Data
        """

        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_customer_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )

            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True
            )
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True
            )
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True
            )
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True
            )
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(
                include=[np.number]
            )  # to include only selected_dtypes

            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data:{e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]  # bug? how to access after dropping
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=4
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data:{e}")
            raise e


# final class to make use of above strategy


class DataCleaning:
    """
    Class for cleaning data which process the data and divides the data
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        handle data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data:{e}")
            raise e
