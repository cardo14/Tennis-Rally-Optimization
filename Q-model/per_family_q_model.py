from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pooled_q_model import (
    Q_ALL_SHOT_FEATURES,
    CANDIDATE_ACTIONS,
    fit_pooled_q_model,
    predict_q,
)


def fit_per_family_q_models(decision_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    family_models = {}

    for family in ["FH", "BH"]:
        family_df = decision_df[decision_df["stroke_family"] == family].copy()
        if len(family_df) < 100:
            continue
        bundle = fit_pooled_q_model(family_df)
        bundle["stroke_family"] = family
        family_models[family] = bundle

    return family_models


def score_with_per_family_models(
    family_models: dict[str, dict[str, Any]],
    decision_df: pd.DataFrame,
) -> pd.DataFrame:
    scored = decision_df.copy()
    scored["pf_q_cc"] = np.nan
    scored["pf_q_dtl"] = np.nan

    for family, bundle in family_models.items():
        mask = scored["stroke_family"] == family

        if family == "FH":
            cc_action, dtl_action = "FH_CC", "FH_DTL"
        else:
            cc_action, dtl_action = "BH_CC", "BH_DTL"

        family_rows = scored.loc[mask]
        scored.loc[mask, "pf_q_cc"] = predict_q(bundle, family_rows, action_label=cc_action).values
        scored.loc[mask, "pf_q_dtl"] = predict_q(bundle, family_rows, action_label=dtl_action).values

    scored["pf_q_observed"] = np.where(
        scored["action"] == "DTL", scored["pf_q_dtl"], scored["pf_q_cc"]
    )
    scored["pf_q_best"] = scored[["pf_q_cc", "pf_q_dtl"]].max(axis=1)
    scored["pf_regret"] = scored["pf_q_best"] - scored["pf_q_observed"]
    scored["pf_recommended"] = np.where(
        scored["pf_q_dtl"] > scored["pf_q_cc"], "DTL", "CC"
    )

    return scored
