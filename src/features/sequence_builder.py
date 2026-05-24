from __future__ import annotations

from typing import Any

import pandas as pd


PAD_TOKEN = 0
OTHER_TOKEN = 9

SHOT_TOKEN_LABELS = {
    PAD_TOKEN: "PAD",
    1: "FH_0",
    2: "FH_1",
    3: "FH_2",
    4: "FH_3",
    5: "BH_0",
    6: "BH_1",
    7: "BH_2",
    8: "BH_3",
    OTHER_TOKEN: "OTHER",
}

TOKEN_TO_SHOT_LABEL = {token: label for token, label in SHOT_TOKEN_LABELS.items() if token != PAD_TOKEN}


def encode_shot_token(family: Any, direction: Any) -> int:
    direction_value = None if pd.isna(direction) else int(direction)
    if direction_value not in {0, 1, 2, 3}:
        direction_value = 0

    if family == "FH":
        return 1 + direction_value
    if family == "BH":
        return 5 + direction_value
    return OTHER_TOKEN


def build_rally_sequence_table(shot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = shot_df.sort_values(["match_date", "match_id", "point_number", "shot_index"])

    for point_id, group in ordered.groupby("point_id", sort=False):
        group = group.sort_values("shot_index")
        first = group.iloc[0]
        sequence = [
            encode_shot_token(row.current_family, row.current_direction)
            for row in group.itertuples(index=False)
        ]
        if not sequence:
            continue
        rows.append(
            {
                "match_id": first["match_id"],
                "point_id": point_id,
                "match_date": first["match_date"],
                "surface": first["surface"],
                "server": first["server"],
                "returner": first["returner"],
                "point_winner": first["point_winner"],
                "server_win": int(first["point_winner"] == first["server"]),
                "sequence_length": len(sequence),
                "encoded_sequence": sequence,
                "token_sequence": " ".join(TOKEN_TO_SHOT_LABEL[token] for token in sequence),
            }
        )

    return pd.DataFrame(rows)


def build_next_shot_examples(
    shot_df: pd.DataFrame,
    window: int = 5,
    max_examples: int | None = None,
) -> pd.DataFrame:
    rows = []
    ordered = shot_df.sort_values(["match_date", "match_id", "point_number", "shot_index"])

    for point_id, group in ordered.groupby("point_id", sort=False):
        group = group.sort_values("shot_index").reset_index(drop=True)
        tokens = [
            encode_shot_token(row.current_family, row.current_direction)
            for row in group.itertuples(index=False)
        ]
        for i in range(len(tokens) - 1):
            state = tokens[max(0, i - window + 1): i + 1]
            state_tokens = [PAD_TOKEN] * (window - len(state)) + state
            next_row = group.iloc[i + 1]
            rows.append(
                {
                    "match_id": next_row["match_id"],
                    "point_id": point_id,
                    "match_date": next_row["match_date"],
                    "point_number": next_row["point_number"],
                    "target_shot_index": int(next_row["shot_index"]),
                    "hitter": next_row["hitter"],
                    "opponent": next_row["opponent"],
                    "server": next_row["server"],
                    "hitter_is_server": int(next_row["hitter_is_server"]),
                    "state_tokens": tuple(state_tokens),
                    "state_token_text": " ".join(SHOT_TOKEN_LABELS[token] for token in state_tokens),
                    "next_token": int(tokens[i + 1]),
                    "next_token_text": TOKEN_TO_SHOT_LABEL[int(tokens[i + 1])],
                    "next_family": next_row["current_family"],
                    "next_direction": next_row["current_direction"],
                }
            )
            if max_examples is not None and len(rows) >= max_examples:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows)
