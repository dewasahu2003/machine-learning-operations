import logging

import pandas as pd
from zenml import step

@step
def eval_model(df:pd.DataFrame)->pd.DataFrame:
    pass