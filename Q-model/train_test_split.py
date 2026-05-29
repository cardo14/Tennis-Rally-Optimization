from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.calibration import calibration_summary


def time_split(
    decision_df: pd.DataFrame,
    split_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_df = decision_df.copy()

    if not pd.api.types.is_datetime64_any_dtype(decision_df["match_date"]):
        decision_df["match_date"] = pd.to_datetime(decision_df["match_date"])

    train_mask = decision_df["match_date"].dt.year < split_year
    test_mask = decision_df["match_date"].dt.year >= split_year

    train_df = decision_df[train_mask].reset_index(drop=True)
    test_df = decision_df[test_mask].reset_index(drop=True)

    return train_df, test_df


def evaluate_on_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], dict[str, Any]],
    predict_fn: Callable[[dict[str, Any], pd.DataFrame, str], pd.Series],
    model_name: str = "model",
) -> dict[str, Any]:
    bundle = fit_fn(train_df)

    results = {"model_name": model_name, "bundle": bundle}

    for split_name, df in [("train", train_df), ("test", test_df)]:
        scored = df.copy()

        q_cc_vals = np.full(len(df), np.nan)
        q_dtl_vals = np.full(len(df), np.nan)

        for family in ["FH", "BH"]:
            mask = df["stroke_family"] == family
            if mask.sum() == 0:
                continue
            cc_action = f"{family}_CC"
            dtl_action = f"{family}_DTL"
            q_cc_vals[mask] = predict_fn(bundle, df[mask], cc_action).values
            q_dtl_vals[mask] = predict_fn(bundle, df[mask], dtl_action).values

        scored["q_cc"] = q_cc_vals
        scored["q_dtl"] = q_dtl_vals
        scored["q_observed"] = np.where(scored["action"] == "DTL", scored["q_dtl"], scored["q_cc"])
        scored["q_best"] = scored[["q_cc", "q_dtl"]].max(axis=1)
        scored["regret"] = scored["q_best"] - scored["q_observed"]
        scored["recommended_action"] = np.where(scored["q_dtl"] > scored["q_cc"], "DTL", "CC")

        metrics, reliability = calibration_summary(scored["point_win"], scored["q_observed"])

        results[f"{split_name}_scored"] = scored
        results[f"{split_name}_metrics"] = metrics
        results[f"{split_name}_reliability"] = reliability

    return results


def print_comparison(results: dict[str, Any]) -> None:
    name = results["model_name"]
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    for split in ["train", "test"]:
        scored = results[f"{split}_scored"]
        metrics = results[f"{split}_metrics"]

        brier = metrics.loc[metrics["metric"] == "brier_score", "value"].iloc[0]
        ll = metrics.loc[metrics["metric"] == "log_loss", "value"].iloc[0]
        ece = metrics.loc[metrics["metric"] == "expected_calibration_error", "value"].iloc[0]

        n = len(scored)
        mean_regret = scored["regret"].mean()
        pct_dtl_rec = (scored["recommended_action"] == "DTL").mean()
        regret_gt_1pct = (scored["regret"] > 0.01).mean()

        print(f"\n  [{split.upper()}] ({n:,} rows)")
        print(f"    Brier score:    {brier:.5f}")
        print(f"    Log loss:       {ll:.5f}")
        print(f"    ECE:            {ece:.5f}")
        print(f"    Mean regret:    {mean_regret:.5f}")
        print(f"    Recommend DTL:  {pct_dtl_rec:.1%}")
        print(f"    Regret > 1%:    {regret_gt_1pct:.1%} of rows")

    train_brier = results["train_metrics"].loc[
        results["train_metrics"]["metric"] == "brier_score", "value"
    ].iloc[0]
    test_brier = results["test_metrics"].loc[
        results["test_metrics"]["metric"] == "brier_score", "value"
    ].iloc[0]
    degradation = test_brier - train_brier

    print(f"\n  Train->Test Brier degradation: {degradation:+.5f}", end="")
    if abs(degradation) < 0.002:
        print("  (minimal -- model generalizes well)")
    elif degradation > 0.005:
        print("  (significant -- model may be overfitting)")
    else:
        print("  (moderate)")
