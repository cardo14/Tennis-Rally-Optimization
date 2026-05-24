from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.overlap import label_support


def _neutralized_adaptation_row(opponent_adaptation_df: pd.DataFrame) -> dict[str, float | str]:
    if opponent_adaptation_df.empty:
        return {
            "adaptation_signal": "unavailable",
            "opponent_neutralized_adaptation_did": np.nan,
            "opponent_neutralized_adaptation_ci_low": np.nan,
            "opponent_neutralized_adaptation_ci_high": np.nan,
            "opponent_neutralized_placebo_did": np.nan,
            "opponent_neutralized_placebo_ci_low": np.nan,
            "opponent_neutralized_placebo_ci_high": np.nan,
        }

    rows = opponent_adaptation_df[opponent_adaptation_df["outcome"] == "opponent_neutralized"]
    if rows.empty:
        return {
            "adaptation_signal": "unavailable",
            "opponent_neutralized_adaptation_did": np.nan,
            "opponent_neutralized_adaptation_ci_low": np.nan,
            "opponent_neutralized_adaptation_ci_high": np.nan,
            "opponent_neutralized_placebo_did": np.nan,
            "opponent_neutralized_placebo_ci_low": np.nan,
            "opponent_neutralized_placebo_ci_high": np.nan,
        }

    row = rows.iloc[0]
    ci_above_zero = row["adaptation_ci_low"] > 0
    placebo_crosses_zero = row["placebo_ci_low"] <= 0 <= row["placebo_ci_high"]
    if ci_above_zero and placebo_crosses_zero:
        signal = "supported_adaptation_signal"
    elif ci_above_zero:
        signal = "adaptation_signal_with_placebo_risk"
    else:
        signal = "weak_or_uncertain_adaptation_signal"

    return {
        "adaptation_signal": signal,
        "opponent_neutralized_adaptation_did": float(row["adaptation_did"]),
        "opponent_neutralized_adaptation_ci_low": float(row["adaptation_ci_low"]),
        "opponent_neutralized_adaptation_ci_high": float(row["adaptation_ci_high"]),
        "opponent_neutralized_placebo_did": float(row["placebo_did"]),
        "opponent_neutralized_placebo_ci_low": float(row["placebo_ci_low"]),
        "opponent_neutralized_placebo_ci_high": float(row["placebo_ci_high"]),
    }


def add_shot_adaptive_evaluation_columns(
    response_df: pd.DataFrame,
    opponent_adaptation_df: pd.DataFrame,
) -> pd.DataFrame:
    shot_df = response_df.copy()
    adaptation = _neutralized_adaptation_row(opponent_adaptation_df)

    shot_df["recommended_cc_pct"] = shot_df["pi_cc"]
    shot_df["observed_cc"] = (shot_df["action"] == "CC").astype(int)
    shot_df["cc_policy_gap"] = shot_df["recommended_cc_pct"] - shot_df["observed_cc"]
    shot_df["adaptive_pressure"] = (
        (shot_df["high_recent_dtl"] == 1)
        & (adaptation["opponent_neutralized_adaptation_did"] > 0)
    ).astype(int)
    shot_df["adaptation_signal"] = adaptation["adaptation_signal"]
    shot_df["opponent_neutralized_adaptation_did"] = adaptation["opponent_neutralized_adaptation_did"]
    shot_df["opponent_neutralized_placebo_did"] = adaptation["opponent_neutralized_placebo_did"]
    shot_df["adaptive_cc_timing_error"] = np.where(
        shot_df["adaptive_pressure"] == 1,
        shot_df["cc_policy_gap"],
        0.0,
    )
    return shot_df


def _mean_or_nan(group: pd.DataFrame, column: str) -> float:
    if group.empty:
        return np.nan
    return float(group[column].mean())


def _rate_or_nan(group: pd.DataFrame, mask_column: str) -> float:
    if group.empty:
        return np.nan
    return float(group[mask_column].mean())


def _metrics_for_group(group: pd.DataFrame, keys: dict[str, object]) -> dict[str, object]:
    high = group[group["high_recent_dtl"] == 1]
    low = group[group["low_recent_dtl"] == 1]
    high_dtl = high[high["action"] == "DTL"]
    high_cc = high[high["action"] == "CC"]

    cc_count = int((group["action"] == "CC").sum())
    dtl_count = int((group["action"] == "DTL").sum())
    mean_min_propensity = float(group[["mu_cc", "mu_dtl"]].min(axis=1).mean())
    high_policy_gap_cc = _mean_or_nan(high, "cc_policy_gap")
    low_policy_gap_cc = _mean_or_nan(low, "cc_policy_gap")

    metrics = {
        **keys,
        "sample_size": int(len(group)),
        "observed_cc_pct": float(group["observed_cc"].mean()),
        "recommended_cc_pct": float(group["recommended_cc_pct"].mean()),
        "policy_gap_cc": float(group["cc_policy_gap"].mean()),
        "average_regret": float(group["regret"].mean()),
        "estimated_uplift": float(group["expected_uplift"].mean()),
        "mean_q_edge_cc": float(group["q_edge_cc"].mean()),
        "adaptation_pressure_rate": float(group["adaptive_pressure"].mean()),
        "adaptive_cc_timing_error": float(group["adaptive_cc_timing_error"].mean()),
        "high_recent_dtl_sample_size": int(len(high)),
        "high_recent_dtl_observed_cc_pct": _rate_or_nan(high, "observed_cc"),
        "high_recent_dtl_recommended_cc_pct": _mean_or_nan(high, "recommended_cc_pct"),
        "high_recent_dtl_policy_gap_cc": high_policy_gap_cc,
        "high_recent_dtl_average_regret": _mean_or_nan(high, "regret"),
        "high_recent_dtl_estimated_uplift": _mean_or_nan(high, "expected_uplift"),
        "high_recent_dtl_q_edge_cc": _mean_or_nan(high, "q_edge_cc"),
        "low_recent_dtl_sample_size": int(len(low)),
        "low_recent_dtl_observed_cc_pct": _rate_or_nan(low, "observed_cc"),
        "low_recent_dtl_recommended_cc_pct": _mean_or_nan(low, "recommended_cc_pct"),
        "low_recent_dtl_policy_gap_cc": low_policy_gap_cc,
        "low_recent_dtl_average_regret": _mean_or_nan(low, "regret"),
        "low_recent_dtl_estimated_uplift": _mean_or_nan(low, "expected_uplift"),
        "low_recent_dtl_q_edge_cc": _mean_or_nan(low, "q_edge_cc"),
        "adaptive_timing_gap_cc": (
            high_policy_gap_cc - low_policy_gap_cc
            if not np.isnan(high_policy_gap_cc) and not np.isnan(low_policy_gap_cc)
            else np.nan
        ),
        "opponent_neutralized_after_high_dtl_action": _mean_or_nan(high_dtl, "opponent_neutralized"),
        "opponent_neutralized_after_high_cc_action": _mean_or_nan(high_cc, "opponent_neutralized"),
        "mean_min_propensity": mean_min_propensity,
        "support_label": label_support(mean_min_propensity, min(cc_count, dtl_count), int(len(group))),
    }
    for column in [
        "adaptation_signal",
        "opponent_neutralized_adaptation_did",
        "opponent_neutralized_placebo_did",
    ]:
        metrics[column] = group[column].iloc[0]
    return metrics


def build_player_adaptive_evaluation(
    shot_eval_df: pd.DataFrame,
    min_context_rows: int = 25,
) -> pd.DataFrame:
    rows = []
    for (hitter, side_proxy), group in shot_eval_df.groupby(["hitter", "side_proxy"], dropna=False):
        if len(group) < min_context_rows:
            continue
        rows.append(_metrics_for_group(group, {"hitter": hitter, "side_proxy": side_proxy}))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["sample_size", "adaptive_cc_timing_error"], ascending=[False, False]).reset_index(drop=True)


def build_match_adaptive_evaluation(
    shot_eval_df: pd.DataFrame,
    min_context_rows: int = 5,
) -> pd.DataFrame:
    rows = []
    group_columns = ["match_id", "match_date", "hitter", "opponent", "side_proxy"]
    for group_key, group in shot_eval_df.groupby(group_columns, dropna=False):
        if len(group) < min_context_rows:
            continue
        keys = dict(zip(group_columns, group_key))
        rows.append(_metrics_for_group(group, keys))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["adaptive_cc_timing_error", "sample_size"], ascending=[False, False]).reset_index(drop=True)
