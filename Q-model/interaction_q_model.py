from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    action = result["action"].astype(str)

    result["hitter_action"] = result["hitter"].astype(str) + "_" + action
    result["opponent_action"] = result["opponent"].astype(str) + "_" + action

    result["family_side_action"] = (
        result["stroke_family"].astype(str) + "_"
        + result["side_proxy"].astype(str) + "_"
        + action
    )

    result["shot_depth_action"] = (
        result["shot_number_bucket"].astype(str) + "_" + action
    )

    result["recent_dtl_bucket"] = pd.cut(
        result["recent_dtl_rate"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["low_dtl", "mid_dtl", "high_dtl"],
    ).astype(str)
    result["exposure_x_action"] = result["recent_dtl_bucket"] + "_" + action

    return result


BASE_FEATURES = [
    "action_label",
    "stroke_family",
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

INTERACTION_FEATURES = [
    "hitter_action",
    "opponent_action",
    "family_side_action",
    "shot_depth_action",
    "exposure_x_action",
]

ALL_FEATURES = BASE_FEATURES + INTERACTION_FEATURES

CATEGORICAL = [
    "action_label",
    "stroke_family",
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
    "hitter_action",
    "opponent_action",
    "family_side_action",
    "shot_depth_action",
    "exposure_x_action",
]

NUMERIC = [
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

CANDIDATE_ACTIONS = ["FH_CC", "FH_DTL", "BH_CC", "BH_DTL"]


def _build_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=15,
                ),
            ),
        ]
    )
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="constant", fill_value=-1))]
    )

    return ColumnTransformer(
        [
            ("categorical", categorical_pipe, CATEGORICAL),
            ("numeric", numeric_pipe, NUMERIC),
        ]
    )


def fit_interaction_q_model(train_df: pd.DataFrame) -> dict[str, Any]:
    train_with_interactions = add_interaction_features(train_df)

    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    solver="liblinear",
                    C=0.3,
                ),
            ),
        ]
    )

    model.fit(
        train_with_interactions[ALL_FEATURES],
        train_with_interactions["point_win"],
    )

    return {
        "model": model,
        "feature_columns": list(ALL_FEATURES),
        "candidate_actions": list(CANDIDATE_ACTIONS),
    }


def predict_q_interaction(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    action_label: str | None = None,
) -> pd.Series:
    score_df = df.copy()
    if action_label is not None:
        score_df["action_label"] = action_label
        raw_action = action_label.split("_")[1]
        score_df["action"] = raw_action

    score_with_interactions = add_interaction_features(score_df)

    probs = bundle["model"].predict_proba(
        score_with_interactions[bundle["feature_columns"]]
    )[:, 1]

    return pd.Series(probs, index=df.index)
