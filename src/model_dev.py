import logging

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin


class Model(ABC):
    """
    ABstract Class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: training data
            y_train: training label
        Returns:
            None
        """
        pass


class LinearRegressionModel(Model):
    """
    Linear Regression Model
    """

    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        """
        Trains the model
        Args:
            X_train: training data
            y_train: training label
        Returns:
            None
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error(f"Error in training model:{e}")
            raise e
