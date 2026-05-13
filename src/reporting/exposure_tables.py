from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.overlap import label_support


def _recent_usage_bucket(row: pd.Series) -> str:
    if row["recent_context_count"] <= 0:
        return "no_prior"
    if row["recent_dtl_rate"] <= 1.0 / 3.0:
        return "low_recent_dtl"
    if row["recent_dtl_rate"] >= 2.0 / 3.0:
        return "high_recent_dtl"
    return "mixed_recent_dtl"


def build_recent_usage_q_table(scored_df: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    exposure_df = scored_df.copy()
    exposure_df["recent_usage_bucket"] = exposure_df.apply(_recent_usage_bucket, axis=1)

    rows = []
    for (hitter, side_proxy, bucket), group in exposure_df.groupby(
        ["hitter", "side_proxy", "recent_usage_bucket"], dropna=False
    ):
        if len(group) < min_rows:
            continue
        cc_count = int((group["action"] == "CC").sum())
        dtl_count = int((group["action"] == "DTL").sum())
        rows.append(
            {
                "hitter": hitter,
                "side_proxy": side_proxy,
                "recent_usage_bucket": bucket,
                "sample_size": int(len(group)),
                "mean_recent_dtl_rate": float(group["recent_dtl_rate"].mean()),
                "mean_recent_context_count": float(group["recent_context_count"].mean()),
                "observed_dtl_pct": float(group["action_dtl"].mean()),
                "recommended_dtl_pct": float(group["pi_dtl"].mean()),
                "mean_q_cc": float(group["q_cc"].mean()),
                "mean_q_dtl": float(group["q_dtl"].mean()),
                "mean_q_edge_dtl": float((group["q_dtl"] - group["q_cc"]).mean()),
                "estimated_uplift": float(group["expected_uplift"].mean()),
                "support_label": label_support(
                    float(group[["mu_cc", "mu_dtl"]].min(axis=1).mean()),
                    min(cc_count, dtl_count),
                    int(len(group)),
                ),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    bucket_order = {
        "no_prior": 0,
        "low_recent_dtl": 1,
        "mixed_recent_dtl": 2,
        "high_recent_dtl": 3,
    }
    table["bucket_order"] = table["recent_usage_bucket"].map(bucket_order).fillna(99)
    return (
        table.sort_values(["sample_size", "hitter", "side_proxy", "bucket_order"], ascending=[False, True, True, True])
        .drop(columns=["bucket_order"])
        .reset_index(drop=True)
    )


def build_recent_usage_q_contrast_table(exposure_table: pd.DataFrame) -> pd.DataFrame:
    if exposure_table.empty:
        return exposure_table.copy()

    usable = exposure_table[exposure_table["recent_usage_bucket"].isin(["low_recent_dtl", "high_recent_dtl"])]
    if usable.empty:
        return pd.DataFrame()

    pivot = usable.pivot_table(
        index=["hitter", "side_proxy"],
        columns="recent_usage_bucket",
        values=["sample_size", "mean_q_cc", "mean_q_dtl", "mean_q_edge_dtl", "recommended_dtl_pct"],
        aggfunc="first",
    )
    required = [("mean_q_edge_dtl", "low_recent_dtl"), ("mean_q_edge_dtl", "high_recent_dtl")]
    if not all(column in pivot.columns for column in required):
        return pd.DataFrame()

    contrast = pd.DataFrame(index=pivot.index)
    for metric in ["sample_size", "mean_q_cc", "mean_q_dtl", "mean_q_edge_dtl", "recommended_dtl_pct"]:
        for bucket in ["low_recent_dtl", "high_recent_dtl"]:
            column = (metric, bucket)
            if column in pivot.columns:
                contrast[f"{bucket}_{metric}"] = pivot[column]

    contrast = contrast.dropna(subset=["low_recent_dtl_mean_q_edge_dtl", "high_recent_dtl_mean_q_edge_dtl"])
    if contrast.empty:
        return pd.DataFrame()

    contrast["q_edge_shift_high_minus_low"] = (
        contrast["high_recent_dtl_mean_q_edge_dtl"] - contrast["low_recent_dtl_mean_q_edge_dtl"]
    )
    contrast["recommended_dtl_shift_high_minus_low"] = (
        contrast["high_recent_dtl_recommended_dtl_pct"] - contrast["low_recent_dtl_recommended_dtl_pct"]
    )
    contrast["total_contrast_sample"] = (
        contrast["high_recent_dtl_sample_size"] + contrast["low_recent_dtl_sample_size"]
    )
    contrast = contrast.replace([np.inf, -np.inf], np.nan).dropna()
    return contrast.reset_index().sort_values("total_contrast_sample", ascending=False).reset_index(drop=True)
