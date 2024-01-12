import logging
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, r2_score

import numpy as np


class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our model
    """

    @abstractmethod
    def calculate_sources(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: True Labels
            y_pred: Predicted Labels
        Returns:
            None
        """
        pass


class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """

    def calculate_sources(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model using MSE
        Args:
            y_true: True Labels
            y_pred: Predicted Labels
        Returns:
            mse: float | ndarray
        """

        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in evaluating model via MSE:{e}")
            raise e


class R2Score(Evaluation):
    """
    Evaluation Strategy that uses R2Score
    """

    def calculate_sources(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model using R2Score
        Args:
            y_true: True Labels
            y_pred: Predicted Labels
        Returns:
            mse: float | ndarray
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"R2 Score:{r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in evaluating model via R2Score")
            raise e


class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared Error
    """

    def calculate_sources(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model using RMSE
        Args:
            y_true: True Labels
            y_pred: Predicted Labels
        Returns:
            rmse: float | ndarray
        """

        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in evaluating model via RMSE:{e}")
            raise e
