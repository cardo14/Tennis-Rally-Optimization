from __future__ import annotations

from typing import Any

import pandas as pd

from src.parsing.mcp_parser import parse_point_row


def winner_name_from_row(row: pd.Series) -> str:
    return row["player1"] if int(row["PtWinner"]) == 1 else row["player2"]


def infer_point_winner_from_parse(parsed_point: dict[str, Any], server_name: str, returner_name: str) -> str | None:
    rally_shots = parsed_point["rally_shots"]
    serve_info = parsed_point["serve_info"]

    if rally_shots:
        last_hitter = returner_name if len(rally_shots) % 2 == 1 else server_name
        last_ending = rally_shots[-1]["ending"]
        if last_ending == "winner":
            return last_hitter
        if last_ending in {"forced_error", "unforced_error"}:
            return server_name if last_hitter == returner_name else returner_name
        return None

    if serve_info["outcome"] in {"ace", "forced_return_error", "unforced_return_error"}:
        return server_name
    if serve_info["outcome"] == "serve_error":
        return returner_name
    return None


def validate_points(points_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_rows: list[dict[str, Any]] = []

    for _, row in points_df.iterrows():
        parsed_point = parse_point_row(row.to_dict())
        server_name = row["player1"] if int(row["Svr"]) == 1 else row["player2"]
        returner_name = row["player2"] if int(row["Svr"]) == 1 else row["player1"]
        actual_winner = winner_name_from_row(row)
        parsed_winner = infer_point_winner_from_parse(parsed_point, server_name, returner_name)
        rally_shots = parsed_point["rally_shots"]

        validation_rows.append(
            {
                "match_id": row["match_id"],
                "point_id": f"{row['match_id']}:{row['Pt']}",
                "Pt": row["Pt"],
                "active_string": parsed_point["active_string"],
                "serve_number": parsed_point["serve_number"],
                "serve_direction": parsed_point["serve_info"]["serve_direction"],
                "serve_outcome": parsed_point["serve_info"]["outcome"],
                "rally_shot_count": len(rally_shots),
                "terminal_source": "rally" if rally_shots else "serve",
                "terminal_ending": rally_shots[-1]["ending"] if rally_shots else parsed_point["serve_info"]["outcome"],
                "predicted_winner": parsed_winner,
                "actual_winner": actual_winner,
                "winner_match": parsed_winner == actual_winner if parsed_winner is not None else False,
                "is_parseable": parsed_point["is_parseable"],
                "hitter_alternation_valid": True,
                "has_explicit_terminal": bool(rally_shots and rally_shots[-1]["ending"] is not None)
                or parsed_point["serve_info"]["outcome"] != "in_play",
            }
        )

    validation_df = pd.DataFrame(validation_rows)
    summary = pd.DataFrame(
        [
            {"metric": "points_total", "value": int(len(validation_df))},
            {"metric": "points_parseable", "value": int(validation_df["is_parseable"].sum())},
            {"metric": "winner_alignment_count", "value": int(validation_df["winner_match"].sum())},
            {
                "metric": "winner_alignment_rate",
                "value": float(validation_df["winner_match"].mean()),
            },
            {
                "metric": "rally_terminal_count",
                "value": int((validation_df["terminal_source"] == "rally").sum()),
            },
            {
                "metric": "serve_terminal_count",
                "value": int((validation_df["terminal_source"] == "serve").sum()),
            },
            {
                "metric": "mean_rally_shot_count",
                "value": float(validation_df["rally_shot_count"].mean()),
            },
        ]
    )
    return validation_df, summary

