from __future__ import annotations

from collections import defaultdict

from src.models.interfaces import ShotSequenceWinProbabilityModel


class EmpiricalShotWinRateModel(ShotSequenceWinProbabilityModel):
    """Single-shot empirical server-win baseline."""

    def __init__(self) -> None:
        self.win_counts: dict[int, int] = defaultdict(int)
        self.total_counts: dict[int, int] = defaultdict(int)
        self.baseline_win_rate = 0.5

    def get_name(self) -> str:
        return "EmpiricalShotWinRate"

    def fit(self, sequences: list[list[int]], labels: list[int]) -> None:
        self.baseline_win_rate = sum(labels) / len(labels) if labels else 0.5
        for sequence, label in zip(sequences, labels):
            for shot in sequence:
                self.total_counts[shot] += 1
                self.win_counts[shot] += int(label)

    def predict_proba(self, shot_sequence: list[int]) -> list[float]:
        probabilities = []
        for shot in shot_sequence:
            total = self.total_counts[shot]
            if total == 0:
                probabilities.append(self.baseline_win_rate)
            else:
                probabilities.append(self.win_counts[shot] / total)
        return probabilities


class MarkovPairWinRateModel(ShotSequenceWinProbabilityModel):
    """Pairwise Markov empirical server-win baseline."""

    def __init__(self) -> None:
        self.pair_win_counts: dict[tuple[int, int], int] = defaultdict(int)
        self.pair_total_counts: dict[tuple[int, int], int] = defaultdict(int)
        self.single_win_counts: dict[int, int] = defaultdict(int)
        self.single_total_counts: dict[int, int] = defaultdict(int)
        self.baseline_win_rate = 0.5

    def get_name(self) -> str:
        return "MarkovPairWinRate"

    def fit(self, sequences: list[list[int]], labels: list[int]) -> None:
        self.baseline_win_rate = sum(labels) / len(labels) if labels else 0.5
        for sequence, label in zip(sequences, labels):
            for i, shot in enumerate(sequence):
                self.single_total_counts[shot] += 1
                self.single_win_counts[shot] += int(label)
                if i > 0:
                    pair = (sequence[i - 1], shot)
                    self.pair_total_counts[pair] += 1
                    self.pair_win_counts[pair] += int(label)

    def _single_probability(self, shot: int) -> float:
        total = self.single_total_counts[shot]
        if total == 0:
            return self.baseline_win_rate
        return self.single_win_counts[shot] / total

    def predict_proba(self, shot_sequence: list[int]) -> list[float]:
        probabilities = []
        for i, shot in enumerate(shot_sequence):
            if i == 0:
                probabilities.append(self._single_probability(shot))
                continue

            pair = (shot_sequence[i - 1], shot)
            pair_total = self.pair_total_counts[pair]
            if pair_total == 0:
                probabilities.append(self._single_probability(shot))
            else:
                probabilities.append(self.pair_win_counts[pair] / pair_total)
        return probabilities
