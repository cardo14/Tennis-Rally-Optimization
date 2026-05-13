import pandas as pd

from src.evaluation.sequence_model_comparison import evaluate_sequence_models
from src.features.sequence_builder import (
    build_next_shot_examples,
    build_rally_sequence_table,
    encode_shot_token,
)
from src.features.state_builder import build_shot_level_table
from src.models.sequence_baselines import EmpiricalShotWinRateModel, MarkovPairWinRateModel


def _sample_points() -> pd.DataFrame:
    base = {
        "match_id": "m3",
        "Set1": 0,
        "Set2": 0,
        "Gm1": 0,
        "Gm2": 0,
        "Pts": "0-0",
        "Svr": 1,
        "2nd": None,
        "Notes": None,
        "player1": "Server",
        "player2": "Returner",
        "player1_hand": "R",
        "player2_hand": "R",
        "match_date": pd.Timestamp("2024-01-01"),
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


def test_sequence_builder_uses_parsed_shot_table():
    shot_df = build_shot_level_table(_sample_points())
    sequence_df = build_rally_sequence_table(shot_df)

    assert len(sequence_df) == 2
    assert sequence_df.loc[0, "encoded_sequence"] == [6, 4, 6, 4, 6]
    assert sequence_df.loc[0, "server_win"] == 0


def test_next_shot_examples_use_only_prefix_tokens():
    shot_df = build_shot_level_table(_sample_points())
    examples = build_next_shot_examples(shot_df, window=3)
    first = examples.iloc[0]

    assert first["state_tokens"] == (0, 0, encode_shot_token("BH", 1))
    assert first["next_token"] == encode_shot_token("FH", 3)


def test_sequence_evaluator_accepts_integrated_model_interface():
    sequences = [[1, 5, 2], [1, 5, 3], [2, 6, 4]]
    labels = [1, 1, 0]
    models = [EmpiricalShotWinRateModel(), MarkovPairWinRateModel()]
    for model in models:
        model.fit(sequences, labels)

    summary = evaluate_sequence_models(models, sequences, labels, candidate_tokens=[1, 2, 3, 4, 5, 6])

    assert set(summary["model"]) == {"EmpiricalShotWinRate", "MarkovPairWinRate"}
    assert {"brier_score", "log_loss", "expected_value_gap"}.issubset(summary.columns)
