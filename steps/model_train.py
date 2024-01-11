import logging

import pandas as pd
from zenml import step

@step
def model_train(df:pd.DataFrame)->pd.DataFrame:
    pass