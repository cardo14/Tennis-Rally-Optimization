from __future__ import annotations

from abc import ABC, abstractmethod


class ShotSequenceWinProbabilityModel(ABC):
    """Common interface for sequence-level win-probability models.

    These models predict the server's point-win probability after each observed
    shot prefix. Shot-level decision models can wrap or transform this later,
    but the interface keeps architecture comparisons consistent.
    """

    @abstractmethod
    def fit(self, sequences: list[list[int]], labels: list[int]) -> None:
        """Fit on encoded rally shot sequences and terminal server-win labels."""

    @abstractmethod
    def predict_proba(self, shot_sequence: list[int]) -> list[float]:
        """Return server-win probability after each prefix in shot_sequence."""

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable model name."""

    def predict_final(self, shot_sequence: list[int]) -> float:
        probabilities = self.predict_proba(shot_sequence)
        return probabilities[-1] if probabilities else 0.5
