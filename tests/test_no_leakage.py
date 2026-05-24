import pandas as pd

from src.features.opponent_response import build_opponent_response_dataset
from src.features.state_builder import build_forehand_direction_dataset, build_shot_level_table


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


def test_match_local_exposure_uses_only_prior_actions():
    shot_df = build_shot_level_table(_points_with_same_prefix())
    context_df = build_forehand_direction_dataset(shot_df, shot_index=5).sort_values("point_number")

    assert context_df.loc[0, "match_prior_context_count"] == 0
    assert context_df.loc[0, "recent_context_count"] == 0
    assert context_df.loc[0, "recent_dtl_rate"] == 0.5
    assert context_df.loc[1, "match_prior_context_count"] == 1
    assert context_df.loc[1, "match_prior_dtl_count"] == 1
    assert context_df.loc[1, "recent_context_count"] == 1
    assert context_df.loc[1, "recent_dtl_rate"] == 1.0


def test_opponent_response_features_attach_next_shot_and_placebo_separately():
    shot_df = build_shot_level_table(_points_with_same_prefix())
    context_df = build_forehand_direction_dataset(shot_df, shot_index=5)
    response_df = build_opponent_response_dataset(context_df, shot_df).sort_values("point_number").reset_index(drop=True)

    assert response_df.loc[0, "opponent_response_exists"] == 1
    assert response_df.loc[0, "opponent_next_winner"] == 1
    assert response_df.loc[0, "future_context_count"] == 1
    assert response_df.loc[0, "future_dtl_rate"] == 1.0
    assert response_df.loc[1, "opponent_response_exists"] == 1
    assert response_df.loc[1, "opponent_next_error"] == 1
    assert response_df.loc[1, "future_context_count"] == 0
