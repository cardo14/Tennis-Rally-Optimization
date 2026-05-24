from __future__ import annotations

import pandas as pd


def label_support(mean_min_propensity: float, minority_count: int, sample_size: int) -> str:
    if mean_min_propensity >= 0.20 and minority_count >= 20 and sample_size >= 40:
        return "high support"
    if mean_min_propensity >= 0.10 and minority_count >= 10 and sample_size >= 20:
        return "moderate support"
    return "low support"


def overlap_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (hitter, side_proxy), group in df.groupby(["hitter", "side_proxy"], dropna=False):
        cc_count = int((group["action"] == "CC").sum())
        dtl_count = int((group["action"] == "DTL").sum())
        minority_count = min(cc_count, dtl_count)
        mean_min_propensity = float(group[["mu_cc", "mu_dtl"]].min(axis=1).mean())
        rows.append(
            {
                "hitter": hitter,
                "side_proxy": side_proxy,
                "sample_size": int(len(group)),
                "cc_count": cc_count,
                "dtl_count": dtl_count,
                "mean_min_propensity": mean_min_propensity,
                "support_label": label_support(mean_min_propensity, minority_count, int(len(group))),
            }
        )
    return pd.DataFrame(rows)

