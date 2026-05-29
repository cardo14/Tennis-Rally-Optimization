from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.sequence_builder import (
    build_rally_sequence_table,
    encode_shot_token,
)
from src.models.sequence_baselines import (
    EmpiricalShotWinRateModel,
    MarkovPairWinRateModel,
)


def _build_prefix_lookup(shot_df: pd.DataFrame) -> dict[str, dict[int, list[int]]]:
    lookup: dict[str, dict[int, list[int]]] = {}
    ordered = shot_df.sort_values(["match_id", "point_number", "shot_index"])

    for point_id, group in ordered.groupby("point_id", sort=False):
        group = group.sort_values("shot_index")
        tokens = [
            encode_shot_token(row.current_family, row.current_direction)
            for row in group.itertuples(index=False)
        ]
        indices = group["shot_index"].tolist()

        prefix_map: dict[int, list[int]] = {}
        for i, shot_idx in enumerate(indices):
            prefix_map[shot_idx] = tokens[:i]
        lookup[point_id] = prefix_map

    return lookup


def fit_sequence_baselines(shot_df: pd.DataFrame) -> dict[str, object]:
    seq_table = build_rally_sequence_table(shot_df)
    sequences = seq_table["encoded_sequence"].tolist()
    labels = seq_table["server_win"].tolist()

    empirical = EmpiricalShotWinRateModel()
    empirical.fit(sequences, labels)

    markov = MarkovPairWinRateModel()
    markov.fit(sequences, labels)

    return {"empirical": empirical, "markov": markov}


def score_with_sequence_baselines(
    seq_models: dict[str, object],
    decision_df: pd.DataFrame,
    shot_df: pd.DataFrame,
    model_name: str = "markov",
) -> pd.DataFrame:
    model = seq_models[model_name]
    prefix_lookup = _build_prefix_lookup(shot_df)

    seq_q_cc = np.full(len(decision_df), np.nan)
    seq_q_dtl = np.full(len(decision_df), np.nan)

    for i, row in enumerate(decision_df.itertuples(index=False)):
        point_id = row.point_id
        shot_idx = row.shot_index
        family = row.stroke_family
        is_server = row.hitter_is_server

        prefix_map = prefix_lookup.get(point_id)
        if prefix_map is None:
            continue
        prefix = prefix_map.get(shot_idx)
        if prefix is None:
            continue

        if family == "FH":
            if row.side_proxy == "deuce":
                cc_token = encode_shot_token("FH", 1)
                dtl_token = encode_shot_token("FH", 3)
            else:
                cc_token = encode_shot_token("FH", 3)
                dtl_token = encode_shot_token("FH", 1)
        else:
            if row.side_proxy == "ad":
                cc_token = encode_shot_token("BH", 3)
                dtl_token = encode_shot_token("BH", 1)
            else:
                cc_token = encode_shot_token("BH", 1)
                dtl_token = encode_shot_token("BH", 3)

        cc_seq = prefix + [cc_token]
        dtl_seq = prefix + [dtl_token]

        cc_probs = model.predict_proba(cc_seq)
        dtl_probs = model.predict_proba(dtl_seq)

        p_server_win_cc = cc_probs[-1] if cc_probs else 0.5
        p_server_win_dtl = dtl_probs[-1] if dtl_probs else 0.5

        if is_server:
            seq_q_cc[i] = p_server_win_cc
            seq_q_dtl[i] = p_server_win_dtl
        else:
            seq_q_cc[i] = 1.0 - p_server_win_cc
            seq_q_dtl[i] = 1.0 - p_server_win_dtl

    scored = decision_df.copy()
    scored["seq_q_cc"] = seq_q_cc
    scored["seq_q_dtl"] = seq_q_dtl
    scored["seq_q_observed"] = np.where(
        scored["action"] == "DTL", scored["seq_q_dtl"], scored["seq_q_cc"]
    )
    scored["seq_q_best"] = scored[["seq_q_cc", "seq_q_dtl"]].max(axis=1)
    scored["seq_regret"] = scored["seq_q_best"] - scored["seq_q_observed"]
    scored["seq_recommended"] = np.where(
        scored["seq_q_dtl"] > scored["seq_q_cc"], "DTL", "CC"
    )

    return scored
