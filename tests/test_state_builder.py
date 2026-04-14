import pandas as pd

from src.features.state_builder import build_shot_level_table


def _sample_points() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "m1",
                "Pt": 1,
                "Set1": 0,
                "Set2": 0,
                "Gm1": 0,
                "Gm2": 0,
                "Pts": "0-0",
                "Svr": 1,
                "1st": "4b1f3b1f3b1*",
                "2nd": None,
                "Notes": None,
                "PtWinner": 2,
                "player1": "Server",
                "player2": "Returner",
                "player1_hand": "R",
                "player2_hand": "R",
                "match_date": pd.Timestamp("2024-01-01"),
                "surface": "Hard",
                "tournament": "Test",
                "round": "R1",
            }
        ]
    )


def test_build_shot_level_table_assigns_shot_indices_and_players():
    shot_df = build_shot_level_table(_sample_points())
    first_shot = shot_df.iloc[0]
    fifth_shot = shot_df[shot_df["shot_index"] == 5].iloc[0]

    assert first_shot["shot_index"] == 2
    assert first_shot["hitter"] == "Returner"
    assert first_shot["previous_shot_code"] == "serve"
    assert fifth_shot["hitter"] == "Server"
    assert fifth_shot["action"] == "DTL"
    assert fifth_shot["side_proxy_confidence"] == "high"

