from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.models.interfaces import ShotSequenceWinProbabilityModel


def _brier_score(labels: list[int], probabilities: list[float]) -> float:
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    return float(np.mean((p - y) ** 2))


def _expected_value_gap(
    model: ShotSequenceWinProbabilityModel,
    sequences: list[list[int]],
    candidate_tokens: list[int],
    max_sequences: int = 200,
) -> float:
    total_gap = 0.0
    total_decisions = 0

    for sequence in sequences[:max_sequences]:
        if not sequence:
            continue
        actual_probs = model.predict_proba(sequence)
        for i, actual_token in enumerate(sequence):
            prev_prob = actual_probs[i - 1] if i > 0 else 0.5
            actual_delta = actual_probs[i] - prev_prob
            best_delta = actual_delta
            prefix = sequence[:i]
            for token in candidate_tokens:
                alt_sequence = prefix + [token]
                alt_probs = model.predict_proba(alt_sequence)
                if not alt_probs:
                    continue
                best_delta = max(best_delta, alt_probs[-1] - prev_prob)
            total_gap += best_delta - actual_delta
            total_decisions += 1

    return float(total_gap / total_decisions) if total_decisions else 0.0


def evaluate_sequence_models(
    models: list[ShotSequenceWinProbabilityModel],
    test_sequences: list[list[int]],
    test_labels: list[int],
    candidate_tokens: list[int],
) -> pd.DataFrame:
    rows = []
    for model in models:
        final_probabilities = [
            float(np.clip(model.predict_final(sequence), 1e-7, 1.0 - 1e-7))
            for sequence in test_sequences
        ]
        try:
            auc = float(roc_auc_score(test_labels, final_probabilities))
        except ValueError:
            auc = np.nan

        rows.append(
            {
                "model": model.get_name(),
                "test_points": int(len(test_sequences)),
                "brier_score": _brier_score(test_labels, final_probabilities),
                "log_loss": float(log_loss(test_labels, final_probabilities)),
                "auc": auc,
                "expected_value_gap": _expected_value_gap(model, test_sequences, candidate_tokens),
            }
        )

    return pd.DataFrame(rows).sort_values(["log_loss", "brier_score"]).reset_index(drop=True)
