from __future__ import annotations

import pandas as pd

from src.evaluation.offline_policy_eval import bootstrap_interval
from src.evaluation.overlap import label_support


def build_player_recommendations(scored_df: pd.DataFrame, min_context_rows: int = 25) -> pd.DataFrame:
    recommendation_rows = []

    for (hitter, side_proxy), group in scored_df.groupby(["hitter", "side_proxy"], dropna=False):
        if len(group) < min_context_rows:
            continue

        uplift_low, uplift_high = bootstrap_interval(group["expected_uplift"])
        cc_count = int((group["action"] == "CC").sum())
        dtl_count = int((group["action"] == "DTL").sum())
        minority_count = min(cc_count, dtl_count)
        mean_min_propensity = float(group[["mu_cc", "mu_dtl"]].min(axis=1).mean())

        recommendation_rows.append(
            {
                "hitter": hitter,
                "side_proxy": side_proxy,
                "sample_size": int(len(group)),
                "observed_dtl_pct": float(group["action_dtl"].mean()),
                "recommended_dtl_pct": float(group["pi_dtl"].mean()),
                "observed_cc_pct": float((group["action"] == "CC").mean()),
                "recommended_cc_pct": float((1.0 - group["pi_dtl"]).mean()),
                "estimated_uplift": float(group["expected_uplift"].mean()),
                "uplift_ci_low": uplift_low,
                "uplift_ci_high": uplift_high,
                "average_regret": float(group["regret"].mean()),
                "mean_q_cc": float(group["q_cc"].mean()),
                "mean_q_dtl": float(group["q_dtl"].mean()),
                "mean_min_propensity": mean_min_propensity,
                "support_label": label_support(mean_min_propensity, minority_count, int(len(group))),
            }
        )

    recommendation_df = pd.DataFrame(recommendation_rows)
    if recommendation_df.empty:
        return recommendation_df
    return recommendation_df.sort_values(["sample_size", "estimated_uplift"], ascending=[False, False]).reset_index(drop=True)

