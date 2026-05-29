from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


Q_TREE_FEATURES = [
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

CATEGORICAL_MASK = [
    True,   # action_label
    True,   # stroke_family
    True,   # hitter
    True,   # opponent
    True,   # hitter_hand
    False,  # hitter_is_server
    True,   # side_proxy
    True,   # shot_number_bucket
    True,   # score_state
    True,   # set_score
    True,   # game_score
    False,  # serve_number
    False,  # serve_direction
    True,   # previous_shot_family
    False,  # previous_shot_direction
    True,   # previous_shot_target_side
    False,  # previous_shot_depth
    True,   # previous_net_state
    True,   # incoming_wing_proxy
    False,  # match_prior_context_count
    False,  # match_prior_dtl_count
    False,  # match_prior_cc_count
    False,  # match_prior_dtl_rate
    False,  # recent_context_count
    False,  # recent_dtl_count
    False,  # recent_cc_count
    False,  # recent_dtl_rate
    False,  # has_match_exposure
    False,  # has_recent_exposure
]

CANDIDATE_ACTIONS = ["FH_CC", "FH_DTL", "BH_CC", "BH_DTL"]


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df[Q_TREE_FEATURES].copy()

    for i, col in enumerate(Q_TREE_FEATURES):
        if CATEGORICAL_MASK[i]:
            encoded[col] = encoded[col].astype("category").cat.codes
        else:
            encoded[col] = pd.to_numeric(encoded[col], errors="coerce")

    return encoded


def _build_encoder_mapping(df: pd.DataFrame) -> dict[str, dict]:
    mappings = {}
    for i, col in enumerate(Q_TREE_FEATURES):
        if CATEGORICAL_MASK[i]:
            cat = df[col].astype("category")
            mappings[col] = dict(zip(cat.cat.categories, range(len(cat.cat.categories))))
    return mappings


def _encode_with_mapping(df: pd.DataFrame, mappings: dict[str, dict]) -> pd.DataFrame:
    encoded = df[Q_TREE_FEATURES].copy()

    for i, col in enumerate(Q_TREE_FEATURES):
        if CATEGORICAL_MASK[i]:
            mapping = mappings.get(col, {})
            encoded[col] = encoded[col].map(mapping).fillna(-1).astype(int)
        else:
            encoded[col] = pd.to_numeric(encoded[col], errors="coerce")

    return encoded


def fit_xgb_q_model(
    train_df: pd.DataFrame,
    max_iter: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 50,
) -> dict[str, Any]:
    mappings = _build_encoder_mapping(train_df)
    X_train = _encode_with_mapping(train_df, mappings)
    y_train = train_df["point_win"].values

    cat_indices = [i for i, is_cat in enumerate(CATEGORICAL_MASK) if is_cat]

    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        categorical_features=cat_indices,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    model.fit(X_train, y_train)

    return {
        "model": model,
        "mappings": mappings,
        "feature_columns": list(Q_TREE_FEATURES),
        "candidate_actions": list(CANDIDATE_ACTIONS),
        "n_trees": model.n_iter_,
    }


def predict_q_xgb(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    action_label: str | None = None,
) -> pd.Series:
    score_df = df.copy()
    if action_label is not None:
        score_df = score_df.copy()
        score_df["action_label"] = action_label

    X = _encode_with_mapping(score_df, bundle["mappings"])
    probs = bundle["model"].predict_proba(X)[:, 1]
    return pd.Series(probs, index=df.index)
