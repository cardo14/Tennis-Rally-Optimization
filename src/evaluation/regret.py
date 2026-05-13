from __future__ import annotations

import numpy as np
import pandas as pd


def add_regret_columns(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["q_obs"] = np.where(scored["action"] == "DTL", scored["q_dtl"], scored["q_cc"])
    scored["best_q"] = scored[["q_cc", "q_dtl"]].max(axis=1)
    scored["regret"] = scored["best_q"] - scored["q_obs"]
    scored["policy_gap_dtl"] = scored["pi_dtl"] - scored["action_dtl"]
    scored["action_cc"] = (scored["action"] == "CC").astype(int)
    scored["policy_gap_cc"] = scored["pi_cc"] - scored["action_cc"]
    scored["q_edge_cc"] = scored["q_cc"] - scored["q_dtl"]
    scored["q_edge_dtl"] = scored["q_dtl"] - scored["q_cc"]
    scored["expected_uplift"] = (
        scored["pi_dtl"] * scored["q_dtl"]
        + (1.0 - scored["pi_dtl"]) * scored["q_cc"]
        - scored["q_obs"]
    )
    return scored
