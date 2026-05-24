from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ADAPTATION_OUTCOMES = [
    "opponent_made_next_ball",
    "opponent_won_within_2",
    "opponent_neutralized",
]

CATEGORICAL_FEATURES = [
    "match_id",
    "hitter",
    "opponent",
    "hitter_hand",
    "hitter_is_server",
    "side_proxy",
    "score_state",
    "set_score",
    "game_score",
    "serve_number",
    "serve_direction",
    "previous_shot_family",
    "previous_shot_target_side",
    "previous_net_state",
    "incoming_wing_proxy",
]

NUMERIC_FEATURES = [
    "action_dtl",
    "high_recent_dtl",
    "action_high_recent_dtl",
    "future_high_dtl",
    "action_future_high_dtl",
    "match_prior_context_count",
    "match_prior_dtl_rate",
    "recent_context_count",
    "recent_dtl_rate",
    "previous_shot_direction",
    "previous_shot_depth",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _build_adaptation_model() -> Pipeline:
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0))])
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2500, solver="liblinear", C=0.5)),
        ]
    )


def _scenario_frame(df: pd.DataFrame, action_dtl: int, high_recent_dtl: int) -> pd.DataFrame:
    scenario = df.copy()
    scenario["action_dtl"] = action_dtl
    scenario["high_recent_dtl"] = high_recent_dtl
    scenario["action_high_recent_dtl"] = action_dtl * high_recent_dtl
    return scenario


def _future_scenario_frame(df: pd.DataFrame, action_dtl: int, future_high_dtl: int) -> pd.DataFrame:
    scenario = df.copy()
    scenario["action_dtl"] = action_dtl
    scenario["future_high_dtl"] = future_high_dtl
    scenario["action_future_high_dtl"] = action_dtl * future_high_dtl
    return scenario


def _predict_mean(model: Pipeline, df: pd.DataFrame) -> float:
    return float(model.predict_proba(df[MODEL_FEATURES])[:, 1].mean())


def _adaptation_contrasts(model: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    dtl_low = _predict_mean(model, _scenario_frame(df, action_dtl=1, high_recent_dtl=0))
    dtl_high = _predict_mean(model, _scenario_frame(df, action_dtl=1, high_recent_dtl=1))
    cc_low = _predict_mean(model, _scenario_frame(df, action_dtl=0, high_recent_dtl=0))
    cc_high = _predict_mean(model, _scenario_frame(df, action_dtl=0, high_recent_dtl=1))

    placebo_dtl_low = _predict_mean(model, _future_scenario_frame(df, action_dtl=1, future_high_dtl=0))
    placebo_dtl_high = _predict_mean(model, _future_scenario_frame(df, action_dtl=1, future_high_dtl=1))
    placebo_cc_low = _predict_mean(model, _future_scenario_frame(df, action_dtl=0, future_high_dtl=0))
    placebo_cc_high = _predict_mean(model, _future_scenario_frame(df, action_dtl=0, future_high_dtl=1))

    return {
        "dtl_effect_high_minus_low": dtl_high - dtl_low,
        "cc_effect_high_minus_low": cc_high - cc_low,
        "adaptation_did": (dtl_high - dtl_low) - (cc_high - cc_low),
        "placebo_did": (placebo_dtl_high - placebo_dtl_low) - (placebo_cc_high - placebo_cc_low),
    }


def _bootstrap_contrast_ci(df: pd.DataFrame, outcome: str, n_boot: int = 80, random_state: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    match_ids = np.asarray(df["match_id"].dropna().unique())
    adaptation_values = []
    placebo_values = []
    for _ in range(n_boot):
        sampled_matches = rng.choice(match_ids, size=len(match_ids), replace=True)
        boot_df = pd.concat([df[df["match_id"] == match_id] for match_id in sampled_matches], ignore_index=True)
        if boot_df[outcome].nunique() < 2:
            continue
        model = _build_adaptation_model()
        model.fit(boot_df[MODEL_FEATURES], boot_df[outcome])
        contrasts = _adaptation_contrasts(model, boot_df)
        adaptation_values.append(contrasts["adaptation_did"])
        placebo_values.append(contrasts["placebo_did"])

    if not adaptation_values:
        return {
            "adaptation_ci_low": np.nan,
            "adaptation_ci_high": np.nan,
            "placebo_ci_low": np.nan,
            "placebo_ci_high": np.nan,
        }

    return {
        "adaptation_ci_low": float(np.percentile(adaptation_values, 2.5)),
        "adaptation_ci_high": float(np.percentile(adaptation_values, 97.5)),
        "placebo_ci_low": float(np.percentile(placebo_values, 2.5)),
        "placebo_ci_high": float(np.percentile(placebo_values, 97.5)),
    }


def fit_opponent_adaptation_models(
    response_df: pd.DataFrame,
    outcomes: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outcomes = outcomes or list(ADAPTATION_OUTCOMES)
    rows = []
    models = {}
    model_df = response_df.copy()

    for outcome in outcomes:
        if model_df[outcome].nunique() < 2:
            continue
        model = _build_adaptation_model()
        model.fit(model_df[MODEL_FEATURES], model_df[outcome])
        predicted = model.predict_proba(model_df[MODEL_FEATURES])[:, 1]
        contrasts = _adaptation_contrasts(model, model_df)
        ci = _bootstrap_contrast_ci(model_df, outcome)
        rows.append(
            {
                "outcome": outcome,
                "sample_size": int(len(model_df)),
                "positive_rate": float(model_df[outcome].mean()),
                "log_loss": float(log_loss(model_df[outcome], np.clip(predicted, 1e-6, 1 - 1e-6))),
                "brier_score": float(brier_score_loss(model_df[outcome], predicted)),
                **contrasts,
                **ci,
            }
        )
        models[outcome] = {
            "model": model,
            "feature_columns": list(MODEL_FEATURES),
            "outcome": outcome,
        }

    return pd.DataFrame(rows), models


def observed_response_summary(response_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = response_df.copy()
    summary_df["recent_exposure_bucket"] = "mixed_or_none"
    summary_df.loc[summary_df["low_recent_dtl"] == 1, "recent_exposure_bucket"] = "low_recent_dtl"
    summary_df.loc[summary_df["high_recent_dtl"] == 1, "recent_exposure_bucket"] = "high_recent_dtl"
    return (
        summary_df.groupby(["action", "recent_exposure_bucket"], dropna=False)
        .agg(
            sample_size=("action", "size"),
            opponent_made_next_ball=("opponent_made_next_ball", "mean"),
            opponent_won_within_2=("opponent_won_within_2", "mean"),
            opponent_neutralized=("opponent_neutralized", "mean"),
            hitter_forced_error_next=("hitter_forced_error_next", "mean"),
            opponent_next_winner=("opponent_next_winner", "mean"),
            mean_recent_dtl_rate=("recent_dtl_rate", "mean"),
        )
        .reset_index()
        .sort_values(["action", "recent_exposure_bucket"])
    )
