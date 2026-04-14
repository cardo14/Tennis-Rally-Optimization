from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.offline_policy_eval import (
    bootstrap_interval,
    clip_probability,
    doubly_robust_scores,
    ipw_scores,
    soft_recommendation_policy,
)
from src.evaluation.overlap import overlap_summary
from src.evaluation.regret import add_regret_columns
from src.features.state_builder import (
    build_forehand_direction_dataset,
    build_shot_level_table,
    load_filtered_points,
)
from src.models.baseline_glm import fit_q_model, predict_q
from src.models.calibration import calibration_summary, fit_isotonic_calibrator
from src.models.propensity_model import fit_propensity_model, predict_propensity
from src.parsing.mcp_parser import parse_rally
from src.parsing.validation import validate_points
from src.reporting.recommendation_tables import build_player_recommendations


OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_MODELS = Path("outputs/models")
INTERIM_DATA = Path("data/interim")


def _write_model_metadata(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2))


def _split_context_dataset(context_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fit_df = context_df[context_df["match_date"] < "2023-07-01"].copy()
    calibration_df = context_df[
        (context_df["match_date"] >= "2023-07-01") & (context_df["match_date"] < "2024-01-01")
    ].copy()
    test_df = context_df[context_df["match_date"] >= "2024-01-01"].copy()

    if fit_df.empty or calibration_df.empty or test_df.empty:
        raise ValueError("Time-based split produced an empty fold. Check the filtered context dataset.")
    return fit_df, calibration_df, test_df


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline() -> dict[str, pd.DataFrame]:
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA.mkdir(parents=True, exist_ok=True)

    filtered_points = load_filtered_points()
    validation_df, validation_summary_df = validate_points(filtered_points)
    _save_frame(validation_df, OUTPUT_TABLES / "parser_validation_rows.csv")
    _save_frame(validation_summary_df, OUTPUT_TABLES / "parser_validation_summary.csv")

    shot_df = build_shot_level_table(filtered_points)
    _save_frame(shot_df, INTERIM_DATA / "shot_level_table.csv")

    context_df = build_forehand_direction_dataset(shot_df, shot_index=5)
    if context_df.empty:
        raise ValueError("No fifth-shot forehand CC/DTL contexts were found.")
    _save_frame(context_df, INTERIM_DATA / "fifth_shot_forehand_context.csv")

    context_summary_df = (
        context_df.groupby(["side_proxy", "hitter"])
        .agg(
            sample_size=("action", "size"),
            dtl_rate=("action_dtl", "mean"),
            point_win_rate=("point_win", "mean"),
        )
        .reset_index()
        .sort_values(["sample_size", "hitter"], ascending=[False, True])
    )
    _save_frame(context_summary_df, OUTPUT_TABLES / "context_summary.csv")

    fit_df, calibration_df, test_df = _split_context_dataset(context_df)

    q_bundle = fit_q_model(fit_df)
    raw_calibration_pred = predict_q(q_bundle, calibration_df)
    q_bundle["calibrator"] = fit_isotonic_calibrator(raw_calibration_pred, calibration_df["point_win"])
    joblib.dump(q_bundle, OUTPUT_MODELS / "q_model.joblib")
    _write_model_metadata(
        OUTPUT_MODELS / "q_model_metadata.json",
        {
            "train_start": str(fit_df["match_date"].min().date()),
            "train_end": str(fit_df["match_date"].max().date()),
            "calibration_start": str(calibration_df["match_date"].min().date()),
            "calibration_end": str(calibration_df["match_date"].max().date()),
            "feature_columns": q_bundle["feature_columns"],
            "context": "fifth_shot_forehand_direction",
            "model": "ridge-logistic-regression",
        },
    )

    propensity_train_df = pd.concat([fit_df, calibration_df], ignore_index=True)
    propensity_bundle = fit_propensity_model(propensity_train_df)
    joblib.dump(propensity_bundle, OUTPUT_MODELS / "behavior_policy_model.joblib")
    _write_model_metadata(
        OUTPUT_MODELS / "behavior_policy_model_metadata.json",
        {
            "train_start": str(propensity_train_df["match_date"].min().date()),
            "train_end": str(propensity_train_df["match_date"].max().date()),
            "feature_columns": propensity_bundle["feature_columns"],
            "context": "fifth_shot_forehand_direction",
            "model": "ridge-logistic-regression",
        },
    )

    scored_test = test_df.copy()
    scored_test["q_cc"] = predict_q(q_bundle, scored_test, action="CC")
    scored_test["q_dtl"] = predict_q(q_bundle, scored_test, action="DTL")
    scored_test["mu_dtl"] = clip_probability(predict_propensity(propensity_bundle, scored_test))
    scored_test["mu_cc"] = 1.0 - scored_test["mu_dtl"]
    scored_test["pi_dtl"] = soft_recommendation_policy(scored_test["q_cc"], scored_test["q_dtl"], tau=0.02)
    scored_test["pi_cc"] = 1.0 - scored_test["pi_dtl"]
    scored_test = add_regret_columns(scored_test)
    _save_frame(scored_test, INTERIM_DATA / "fifth_shot_forehand_scored_test.csv")

    calibration_metrics_df, reliability_df = calibration_summary(scored_test["point_win"], scored_test["q_obs"])
    _save_frame(calibration_metrics_df, OUTPUT_TABLES / "q_model_calibration_metrics.csv")
    _save_frame(reliability_df, OUTPUT_TABLES / "q_model_reliability.csv")

    observed_values = scored_test["point_win"]
    dm_scores = scored_test["pi_dtl"] * scored_test["q_dtl"] + (1.0 - scored_test["pi_dtl"]) * scored_test["q_cc"]
    ipw = ipw_scores(
        observed_action_dtl=scored_test["action_dtl"],
        observed_reward=scored_test["point_win"],
        mu_dtl=scored_test["mu_dtl"],
        pi_dtl=scored_test["pi_dtl"],
    )
    dr = doubly_robust_scores(
        observed_action_dtl=scored_test["action_dtl"],
        observed_reward=scored_test["point_win"],
        mu_dtl=scored_test["mu_dtl"],
        q_cc=scored_test["q_cc"],
        q_dtl=scored_test["q_dtl"],
        pi_dtl=scored_test["pi_dtl"],
    )

    evaluation_summary_df = pd.DataFrame(
        [
            {
                "metric": "observed_policy_value",
                "estimate": float(observed_values.mean()),
                "ci_low": bootstrap_interval(observed_values)[0],
                "ci_high": bootstrap_interval(observed_values)[1],
            },
            {
                "metric": "recommended_policy_value_dm",
                "estimate": float(dm_scores.mean()),
                "ci_low": bootstrap_interval(dm_scores)[0],
                "ci_high": bootstrap_interval(dm_scores)[1],
            },
            {
                "metric": "recommended_policy_value_ipw",
                "estimate": float(ipw.mean()),
                "ci_low": bootstrap_interval(ipw)[0],
                "ci_high": bootstrap_interval(ipw)[1],
            },
            {
                "metric": "recommended_policy_value_dr",
                "estimate": float(dr.mean()),
                "ci_low": bootstrap_interval(dr)[0],
                "ci_high": bootstrap_interval(dr)[1],
            },
            {
                "metric": "recommended_minus_observed_uplift_dr",
                "estimate": float(dr.mean() - observed_values.mean()),
                "ci_low": float(bootstrap_interval(dr - observed_values)[0]),
                "ci_high": float(bootstrap_interval(dr - observed_values)[1]),
            },
        ]
    )
    _save_frame(evaluation_summary_df, OUTPUT_TABLES / "policy_evaluation_summary.csv")

    overlap_df = overlap_summary(scored_test)
    _save_frame(overlap_df, OUTPUT_TABLES / "overlap_summary.csv")

    recommendation_df = build_player_recommendations(scored_test)
    _save_frame(recommendation_df, OUTPUT_TABLES / "player_recommendations.csv")

    return {
        "validation_summary": validation_summary_df,
        "context_summary": context_summary_df,
        "calibration_metrics": calibration_metrics_df,
        "evaluation_summary": evaluation_summary_df,
        "player_recommendations": recommendation_df,
    }


if __name__ == "__main__":
    results = run_pipeline()
    for name, frame in results.items():
        print(f"{name}: {len(frame)} rows")
