import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from helpers.constants import CATEGORICAL_COLS, MI_THRESHOLD, RANDOM_STATE
from helpers.datetimetransformer import datetime_to_unix


def week_day_to_int(series: pd.Series) -> pd.Series:
    mapping = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    return series.map(mapping)


def season_to_int(series: pd.Series) -> pd.Series:
    mapping = {
        "winter": 0,
        "summer": 2,
        "monsoon": 1,
        "post_monsoon": 3,
    }
    return series.map(mapping)


def find_relevant_features(X_train, y_train):
    X_train_mi = X_train.copy()
    X_train_mi["day_of_week"] = week_day_to_int(X_train_mi["day_of_week"])
    X_train_mi["season"] = season_to_int(X_train_mi["season"])

    discrete_mask = [col in CATEGORICAL_COLS for col in X_train_mi.columns]

    mi_scores = mutual_info_regression(
        X_train_mi, y_train, discrete_features=discrete_mask, random_state=RANDOM_STATE
    )

    mi_results = pd.Series(mi_scores, index=X_train_mi.columns).sort_values(
        ascending=False
    )
    print("MI Results:")
    print(mi_results)
    relevant_features = mi_results[mi_results > MI_THRESHOLD].index.tolist()
    return relevant_features
