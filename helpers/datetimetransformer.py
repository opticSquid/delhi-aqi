import numpy as np
import pandas as pd


def datetime_to_unix(series: pd.Series) -> pd.Series:
    return (pd.to_datetime(series).astype(np.int64) // 10**9).astype(np.int64)
