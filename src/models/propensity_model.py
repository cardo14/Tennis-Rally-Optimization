from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROPENSITY_FEATURES = [
    "hitter",
    "opponent",
    "hitter_hand",
    "hitter_is_server",
    "side_proxy",
    "shot_number_bucket",
    "score_state",
    "set_score",
    "game_score",
    "serve_number",
    "serve_direction",
    "previous_shot_family",
    "previous_shot_direction",
    "previous_shot_target_side",
    "previous_shot_depth",
    "previous_net_state",
    "incoming_wing_proxy",
    "match_prior_context_count",
    "match_prior_dtl_count",
    "match_prior_cc_count",
    "match_prior_dtl_rate",
    "recent_context_count",
    "recent_dtl_count",
    "recent_cc_count",
    "recent_dtl_rate",
    "has_match_exposure",
    "has_recent_exposure",
]

CATEGORICAL_FEATURES = [
    "hitter",
    "opponent",
    "hitter_hand",
    "side_proxy",
    "shot_number_bucket",
    "score_state",
    "set_score",
    "game_score",
    "previous_shot_family",
    "previous_shot_target_side",
    "previous_net_state",
    "incoming_wing_proxy",
]

NUMERIC_FEATURES = [
    "hitter_is_server",
    "serve_number",
    "serve_direction",
    "previous_shot_direction",
    "previous_shot_depth",
    "match_prior_context_count",
    "match_prior_dtl_count",
    "match_prior_cc_count",
    "match_prior_dtl_rate",
    "recent_context_count",
    "recent_dtl_count",
    "recent_cc_count",
    "recent_dtl_rate",
    "has_match_exposure",
    "has_recent_exposure",
]


def _build_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=25,
                ),
            ),
        ]
    )
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=-1))])

    return ColumnTransformer(
        [
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ]
    )


def fit_propensity_model(train_df: pd.DataFrame) -> dict[str, Any]:
    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    solver="liblinear",
                    C=0.5,
                ),
            ),
        ]
    )
    model.fit(train_df[PROPENSITY_FEATURES], train_df["action_dtl"])
    return {
        "model": model,
        "feature_columns": list(PROPENSITY_FEATURES),
    }


def predict_propensity(bundle: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    probs = bundle["model"].predict_proba(df[bundle["feature_columns"]])[:, 1]
    return pd.Series(probs, index=df.index)
