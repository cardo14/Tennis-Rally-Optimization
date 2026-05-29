from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


Q_ALL_SHOT_FEATURES = [
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

CATEGORICAL_FEATURES = [
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

CANDIDATE_ACTIONS = ["FH_CC", "FH_DTL", "BH_CC", "BH_DTL"]


def _build_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                ),
            ),
        ]
    )
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="constant", fill_value=-1))]
    )

    return ColumnTransformer(
        [
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ]
    )


def fit_pooled_q_model(train_df: pd.DataFrame) -> dict[str, Any]:
    model = Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    solver="liblinear",
                    C=0.5,
                ),
            ),
        ]
    )

    model.fit(train_df[Q_ALL_SHOT_FEATURES], train_df["point_win"])

    return {
        "model": model,
        "feature_columns": list(Q_ALL_SHOT_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "candidate_actions": list(CANDIDATE_ACTIONS),
    }


def predict_q(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    action_label: str | None = None,
) -> pd.Series:
    score_df = df[bundle["feature_columns"]].copy()
    if action_label is not None:
        score_df["action_label"] = action_label
    raw_pred = bundle["model"].predict_proba(score_df)[:, 1]
    return pd.Series(raw_pred, index=df.index)


def predict_q_all_actions(
    bundle: dict[str, Any],
    df: pd.DataFrame,
) -> pd.DataFrame:
    q_scores = {}
    for action in bundle["candidate_actions"]:
        q_scores[f"q_{action}"] = predict_q(bundle, df, action_label=action)
    q_df = pd.DataFrame(q_scores, index=df.index)
    return q_df


def score_decision_table(
    bundle: dict[str, Any],
    decision_df: pd.DataFrame,
) -> pd.DataFrame:
    scored = decision_df.copy()

    q_df = predict_q_all_actions(bundle, decision_df)
    for col in q_df.columns:
        scored[col] = q_df[col]

    scored["q_cc"] = np.where(
        scored["stroke_family"] == "FH",
        scored["q_FH_CC"],
        scored["q_BH_CC"],
    )
    scored["q_dtl"] = np.where(
        scored["stroke_family"] == "FH",
        scored["q_FH_DTL"],
        scored["q_BH_DTL"],
    )

    scored["q_observed"] = np.where(
        scored["action"] == "DTL",
        scored["q_dtl"],
        scored["q_cc"],
    )

    scored["q_best"] = scored[["q_cc", "q_dtl"]].max(axis=1)
    scored["regret"] = scored["q_best"] - scored["q_observed"]

    scored["recommended_action"] = np.where(
        scored["q_dtl"] > scored["q_cc"],
        "DTL",
        "CC",
    )

    scored["took_recommended"] = (
        scored["action"] == scored["recommended_action"]
    ).astype(int)

    scored["recommended_action_prob"] = scored.groupby(
        ["hitter", "stroke_family", "recommended_action"]
    )["took_recommended"].transform("mean")

    return scored
