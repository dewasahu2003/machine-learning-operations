from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    """Model Configuration"""

    model_name: str = "LinearRegression"
