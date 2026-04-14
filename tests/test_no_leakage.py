import pandas as pd

from src.features.state_builder import build_shot_level_table


def _points_with_same_prefix() -> pd.DataFrame:
    base = {
        "match_id": "m2",
        "Set1": 0,
        "Set2": 0,
        "Gm1": 0,
        "Gm2": 0,
        "Pts": "15-15",
        "Svr": 1,
        "2nd": None,
        "Notes": None,
        "player1": "Server",
        "player2": "Returner",
        "player1_hand": "R",
        "player2_hand": "R",
        "match_date": pd.Timestamp("2024-01-02"),
        "surface": "Hard",
        "tournament": "Test",
        "round": "R1",
    }
    return pd.DataFrame(
        [
            {**base, "Pt": 1, "1st": "4b1f3b1f3b1*", "PtWinner": 2},
            {**base, "Pt": 2, "1st": "4b1f3b1f3b3@", "PtWinner": 1},
        ]
    )


def test_state_features_do_not_depend_on_future_shots():
    shot_df = build_shot_level_table(_points_with_same_prefix())
    shot5 = shot_df[shot_df["shot_index"] == 5].sort_values("point_number").reset_index(drop=True)

    comparable_columns = [
        "shot_index",
        "hitter",
        "opponent",
        "hitter_is_server",
        "serve_number",
        "serve_direction",
        "previous_shot_code",
        "previous_shot_family",
        "previous_shot_direction",
        "previous_shot_target_side",
        "side_proxy",
        "side_proxy_confidence",
        "action",
        "score_state",
    ]

    assert shot5.loc[0, comparable_columns].to_dict() == shot5.loc[1, comparable_columns].to_dict()

