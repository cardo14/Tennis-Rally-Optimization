from abc import ABC, abstractmethod
import numpy as np


class WinProbabilityModel(ABC):
    """
    Base class for all win probability models.
    Every model needs to implement predict_proba and get_name.
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, shot_sequence):
        """
        Given a sequence of shots, return win probability at each step.
        shot_sequence: list of shot encodings
        returns: list of floats (win probabilities), one per shot
        """
        pass

    @abstractmethod
    def get_name(self):
        pass

    def predict_final(self, shot_sequence):
        """Just the final win probability for a rally."""
        probs = self.predict_proba(shot_sequence)
        return probs[-1] if probs else 0.5