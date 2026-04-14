from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.action_builder import classify_forehand_action, infer_side_proxy
from src.parsing.mcp_parser import (
    clean_cell,
    direction_to_court_side,
    infer_net_state,
    is_forehand_groundstroke,
    parse_point_row,
)


POINT_USECOLS = [
    "match_id",
    "Pt",
    "Set1",
    "Set2",
    "Gm1",
    "Gm2",
    "Pts",
    "Svr",
    "1st",
    "2nd",
    "Notes",
    "PtWinner",
]

MATCH_USECOLS = [
    "match_id",
    "Player 1",
    "Player 2",
    "Pl 1 hand",
    "Pl 2 hand",
    "Date",
    "Surface",
    "Tournament",
    "Round",
]


def load_filtered_points(
    points_path: str = "data/raw/charting-m-points-2020s.csv",
    matches_path: str = "data/raw/charting-m-matches.csv",
) -> pd.DataFrame:
    points = pd.read_csv(points_path, usecols=POINT_USECOLS)
    matches = pd.read_csv(matches_path, usecols=MATCH_USECOLS)
    matches = matches.rename(
        columns={
            "Player 1": "player1",
            "Player 2": "player2",
            "Pl 1 hand": "player1_hand",
            "Pl 2 hand": "player2_hand",
            "Date": "match_date",
            "Surface": "surface",
            "Tournament": "tournament",
            "Round": "round",
        }
    )
    merged = points.merge(matches, on="match_id", how="left")
    merged = merged[merged["match_date"].astype(str).str.fullmatch(r"\d{8}")]
    merged["match_date"] = pd.to_datetime(merged["match_date"], format="%Y%m%d")
    merged = merged[
        (merged["surface"] == "Hard")
        & (merged["match_date"].between("2022-01-01", "2024-12-31"))
    ].copy()
    merged = merged.sort_values(["match_date", "match_id", "Pt"]).reset_index(drop=True)
    return merged


def _shot_number_bucket(shot_index: int) -> str:
    if shot_index <= 1:
        return "1"
    if shot_index == 2:
        return "2"
    if shot_index == 3:
        return "3"
    if shot_index in {4, 5}:
        return "4-5"
    return "6+"


def _winner_name(row: pd.Series) -> str:
    return row["player1"] if int(row["PtWinner"]) == 1 else row["player2"]


def build_shot_level_table(points_df: pd.DataFrame) -> pd.DataFrame:
    shot_rows: list[dict[str, Any]] = []

    for _, row in points_df.iterrows():
        parsed_point = parse_point_row(row.to_dict())
        if not parsed_point["is_parseable"] or parsed_point["is_placeholder"]:
            continue

        active_string = parsed_point["active_string"]
        rally_shots = parsed_point["rally_shots"]
        if not rally_shots:
            continue

        server_name = row["player1"] if int(row["Svr"]) == 1 else row["player2"]
        returner_name = row["player2"] if int(row["Svr"]) == 1 else row["player1"]
        server_hand = row["player1_hand"] if int(row["Svr"]) == 1 else row["player2_hand"]
        returner_hand = row["player2_hand"] if int(row["Svr"]) == 1 else row["player1_hand"]
        point_winner = _winner_name(row)
        serve_info = parsed_point["serve_info"]

        for rally_index, shot in enumerate(rally_shots):
            shot_index = rally_index + 2
            hitter = returner_name if rally_index % 2 == 0 else server_name
            opponent = server_name if rally_index % 2 == 0 else returner_name
            hitter_hand = returner_hand if rally_index % 2 == 0 else server_hand
            opponent_hand = server_hand if rally_index % 2 == 0 else returner_hand
            hitter_is_server = int(hitter == server_name)

            if rally_index == 0:
                previous_raw_code = "serve"
                previous_family = "SERVE"
                previous_direction = serve_info["serve_direction"]
                previous_depth = None
                previous_net_state = "baseline"
            else:
                previous_shot = rally_shots[rally_index - 1]
                previous_raw_code = previous_shot["raw_code"]
                previous_family = previous_shot["family"]
                previous_direction = previous_shot["dir"]
                previous_depth = previous_shot["depth"]
                previous_net_state = infer_net_state(previous_shot["raw_code"], previous_shot["position"])

            side_proxy, side_proxy_confidence, incoming_wing = infer_side_proxy(
                previous_direction,
                hitter_hand,
                shot["family"],
            )
            action = None
            if is_forehand_groundstroke(shot["raw_code"]):
                action = classify_forehand_action(shot["dir"], side_proxy)

            shot_rows.append(
                {
                    "match_id": row["match_id"],
                    "point_id": f"{row['match_id']}:{row['Pt']}",
                    "match_date": row["match_date"],
                    "surface": row["surface"],
                    "tournament": row["tournament"],
                    "round": row["round"],
                    "point_number": row["Pt"],
                    "shot_index": shot_index,
                    "shot_number_bucket": _shot_number_bucket(shot_index),
                    "rally_length_so_far": shot_index - 1,
                    "score_state": clean_cell(row["Pts"]),
                    "set_score": f"{int(row['Set1'])}-{int(row['Set2'])}",
                    "game_score": f"{int(row['Gm1'])}-{int(row['Gm2'])}",
                    "server": server_name,
                    "returner": returner_name,
                    "hitter": hitter,
                    "opponent": opponent,
                    "hitter_hand": hitter_hand,
                    "opponent_hand": opponent_hand,
                    "hitter_is_server": hitter_is_server,
                    "serve_number": parsed_point["serve_number"],
                    "serve_direction": serve_info["serve_direction"],
                    "serve_outcome": serve_info["outcome"],
                    "active_string": active_string,
                    "previous_shot_code": previous_raw_code,
                    "previous_shot_family": previous_family,
                    "previous_shot_direction": previous_direction,
                    "previous_shot_target_side": direction_to_court_side(previous_direction),
                    "previous_shot_depth": previous_depth,
                    "previous_net_state": previous_net_state,
                    "incoming_wing_proxy": incoming_wing,
                    "current_raw_code": shot["raw_code"],
                    "current_family": shot["family"],
                    "current_stroke_type": shot["stroke_type"],
                    "current_spin": shot["spin"],
                    "current_direction": shot["dir"],
                    "current_target_side": direction_to_court_side(shot["dir"]),
                    "current_depth": shot["depth"],
                    "current_net_state": infer_net_state(shot["raw_code"], shot["position"]),
                    "current_position_flag": shot["position"],
                    "approach": shot["approach"],
                    "net_cord": shot["net_cord"],
                    "stop_volley": shot["stop_volley"],
                    "ending": shot["ending"],
                    "error_type": shot["error_type"],
                    "point_winner": point_winner,
                    "point_win": int(hitter == point_winner),
                    "side_proxy": side_proxy,
                    "side_proxy_confidence": side_proxy_confidence,
                    "action": action,
                }
            )

    return pd.DataFrame(shot_rows)


def build_forehand_direction_dataset(shot_df: pd.DataFrame, shot_index: int = 5) -> pd.DataFrame:
    dataset = shot_df[
        (shot_df["shot_index"] == shot_index)
        & (shot_df["current_raw_code"] == "f")
        & (shot_df["action"].isin(["CC", "DTL"]))
        & (shot_df["side_proxy_confidence"] == "high")
    ].copy()
    dataset["action_dtl"] = (dataset["action"] == "DTL").astype(int)
    dataset["context_name"] = "fifth_shot_forehand_direction"
    return dataset.reset_index(drop=True)

