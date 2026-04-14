from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


Q_MODEL_FEATURES = [
    "action",
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
]

CATEGORICAL_FEATURES = [
    "action",
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


def fit_q_model(train_df: pd.DataFrame) -> dict[str, Any]:
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
    model.fit(train_df[Q_MODEL_FEATURES], train_df["point_win"])
    return {
        "model": model,
        "feature_columns": list(Q_MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "calibrator": None,
    }


def predict_q(bundle: dict[str, Any], df: pd.DataFrame, action: str | None = None) -> pd.Series:
    score_df = df[bundle["feature_columns"]].copy()
    if action is not None:
        score_df["action"] = action
    raw_pred = bundle["model"].predict_proba(score_df)[:, 1]
    calibrator = bundle.get("calibrator")
    if calibrator is not None:
        raw_pred = calibrator.predict(raw_pred)
    return pd.Series(raw_pred, index=df.index)

