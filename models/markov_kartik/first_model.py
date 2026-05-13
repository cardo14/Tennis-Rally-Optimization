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
from src.evaluation.sequence_model_comparison import evaluate_sequence_models
from src.features.sequence_builder import build_next_shot_examples, build_rally_sequence_table
from src.features.state_builder import (
    build_forehand_direction_dataset,
    build_shot_level_table,
    load_filtered_points,
)
from src.features.opponent_response import build_opponent_response_dataset
from src.models.baseline_glm import fit_q_model, predict_q
from src.models.calibration import calibration_summary, fit_isotonic_calibrator
from src.models.next_shot_policy import NextShotPolicyModel, next_shot_confusion_table
from src.models.opponent_adaptation import fit_opponent_adaptation_models, observed_response_summary
from src.models.propensity_model import fit_propensity_model, predict_propensity
from src.models.sequence_baselines import EmpiricalShotWinRateModel, MarkovPairWinRateModel
from src.parsing.mcp_parser import parse_rally
from src.parsing.validation import validate_points
from src.reporting.recommendation_tables import build_player_recommendations
from src.reporting.exposure_tables import build_recent_usage_q_contrast_table, build_recent_usage_q_table
from src.reporting.player_adaptive_evaluation import (
    add_shot_adaptive_evaluation_columns,
    build_match_adaptive_evaluation,
    build_player_adaptive_evaluation,
)


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


def _sample_points(shot_df: pd.DataFrame, max_points: int, random_state: int) -> pd.DataFrame:
    point_ids = pd.Series(shot_df["point_id"].unique())
    if len(point_ids) > max_points:
        point_ids = point_ids.sample(n=max_points, random_state=random_state)
    return shot_df[shot_df["point_id"].isin(set(point_ids))].copy()


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

    sequence_train_source_df = _sample_points(
        shot_df[shot_df["match_date"] < "2024-01-01"],
        max_points=15000,
        random_state=40,
    )
    sequence_test_source_df = _sample_points(
        shot_df[shot_df["match_date"] >= "2024-01-01"],
        max_points=7500,
        random_state=41,
    )
    sequence_df = pd.concat(
        [
            build_rally_sequence_table(sequence_train_source_df),
            build_rally_sequence_table(sequence_test_source_df),
        ],
        ignore_index=True,
    )
    _save_frame(sequence_df, INTERIM_DATA / "rally_sequence_table.csv")
    sequence_train_df = sequence_df[sequence_df["match_date"] < "2024-01-01"].copy()
    sequence_test_df = sequence_df[sequence_df["match_date"] >= "2024-01-01"].copy()
    sequence_models = [EmpiricalShotWinRateModel(), MarkovPairWinRateModel()]
    train_sequences = sequence_train_df["encoded_sequence"].tolist()
    train_labels = sequence_train_df["server_win"].astype(int).tolist()
    test_sequences = sequence_test_df["encoded_sequence"].tolist()
    test_labels = sequence_test_df["server_win"].astype(int).tolist()
    candidate_tokens = sorted({token for sequence in train_sequences for token in sequence})
    for sequence_model in sequence_models:
        sequence_model.fit(train_sequences, train_labels)
    sequence_comparison_df = evaluate_sequence_models(sequence_models, test_sequences, test_labels, candidate_tokens)
    _save_frame(sequence_comparison_df, OUTPUT_TABLES / "sequence_model_comparison.csv")

    next_shot_train_source_df = _sample_points(
        shot_df[shot_df["match_date"] < "2024-01-01"],
        max_points=12000,
        random_state=42,
    )
    next_shot_test_source_df = _sample_points(
        shot_df[shot_df["match_date"] >= "2024-01-01"],
        max_points=6000,
        random_state=43,
    )
    next_shot_examples_df = pd.concat(
        [
            build_next_shot_examples(next_shot_train_source_df, max_examples=75000),
            build_next_shot_examples(next_shot_test_source_df, max_examples=40000),
        ],
        ignore_index=True,
    )
    _save_frame(next_shot_examples_df, INTERIM_DATA / "next_shot_examples.csv")
    next_shot_train_df = next_shot_examples_df[next_shot_examples_df["match_date"] < "2024-01-01"].copy()
    next_shot_test_df = next_shot_examples_df[next_shot_examples_df["match_date"] >= "2024-01-01"].copy()
    next_shot_policy = NextShotPolicyModel(alpha=0.0005, max_iter=1000)
    next_shot_policy.fit(next_shot_train_df)
    next_shot_policy_summary_df = next_shot_policy.evaluate(next_shot_test_df)
    _save_frame(next_shot_policy_summary_df, OUTPUT_TABLES / "next_shot_policy_summary.csv")
    next_shot_confusion_df = next_shot_confusion_table(next_shot_policy, next_shot_test_df)
    _save_frame(next_shot_confusion_df, OUTPUT_TABLES / "next_shot_policy_confusion.csv")
    joblib.dump(next_shot_policy, OUTPUT_MODELS / "next_shot_policy_model.joblib")
    _write_model_metadata(
        OUTPUT_MODELS / "next_shot_policy_model_metadata.json",
        {
            "train_start": str(next_shot_train_df["match_date"].min().date()),
            "train_end": str(next_shot_train_df["match_date"].max().date()),
            "train_examples_available": int(len(next_shot_train_df)),
            "train_examples_used": int(len(next_shot_train_df)),
            "test_start": str(next_shot_test_df["match_date"].min().date()),
            "test_end": str(next_shot_test_df["match_date"].max().date()),
            "test_examples_used": int(len(next_shot_test_df)),
            "model": "sgd-logistic-next-shot-policy",
            "source_inspiration": "origin/sai next_shot_model.py",
            "parser_source": "src.parsing.mcp_parser via shot_level_table",
        },
    )

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

    recent_usage_q_df = build_recent_usage_q_table(scored_test)
    _save_frame(recent_usage_q_df, OUTPUT_TABLES / "recent_usage_q_summary.csv")
    recent_usage_q_contrast_df = build_recent_usage_q_contrast_table(recent_usage_q_df)
    _save_frame(recent_usage_q_contrast_df, OUTPUT_TABLES / "recent_usage_q_contrasts.csv")

    response_df = build_opponent_response_dataset(scored_test, shot_df)
    response_observed_df = observed_response_summary(response_df)
    _save_frame(response_observed_df, OUTPUT_TABLES / "opponent_response_observed_summary.csv")
    opponent_adaptation_df, opponent_adaptation_models = fit_opponent_adaptation_models(response_df)
    _save_frame(opponent_adaptation_df, OUTPUT_TABLES / "opponent_adaptation_model_summary.csv")
    shot_adaptive_eval_df = add_shot_adaptive_evaluation_columns(response_df, opponent_adaptation_df)
    _save_frame(shot_adaptive_eval_df, INTERIM_DATA / "fifth_shot_forehand_opponent_response.csv")
    player_adaptive_eval_df = build_player_adaptive_evaluation(shot_adaptive_eval_df)
    _save_frame(player_adaptive_eval_df, OUTPUT_TABLES / "player_adaptive_evaluation.csv")
    match_adaptive_eval_df = build_match_adaptive_evaluation(shot_adaptive_eval_df)
    _save_frame(match_adaptive_eval_df, OUTPUT_TABLES / "match_adaptive_evaluation.csv")
    joblib.dump(opponent_adaptation_models, OUTPUT_MODELS / "opponent_adaptation_models.joblib")
    _write_model_metadata(
        OUTPUT_MODELS / "opponent_adaptation_model_metadata.json",
        {
            "context": "fifth_shot_forehand_direction",
            "model": "ridge-logistic-regression-with-match-fixed-effects",
            "outcomes": list(opponent_adaptation_models.keys()),
            "adaptation_term": "action_dtl x high_recent_dtl",
            "placebo_term": "action_dtl x future_high_dtl",
            "bootstrap_unit": "match_id",
        },
    )

    return {
        "validation_summary": validation_summary_df,
        "sequence_model_comparison": sequence_comparison_df,
        "next_shot_policy_summary": next_shot_policy_summary_df,
        "context_summary": context_summary_df,
        "calibration_metrics": calibration_metrics_df,
        "evaluation_summary": evaluation_summary_df,
        "player_recommendations": recommendation_df,
        "recent_usage_q_summary": recent_usage_q_df,
        "recent_usage_q_contrasts": recent_usage_q_contrast_df,
        "opponent_response_observed": response_observed_df,
        "opponent_adaptation": opponent_adaptation_df,
        "player_adaptive_evaluation": player_adaptive_eval_df,
        "match_adaptive_evaluation": match_adaptive_eval_df,
    }


if __name__ == "__main__":
    results = run_pipeline()
    for name, frame in results.items():
        print(f"{name}: {len(frame)} rows")
