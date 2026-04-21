import numpy as np
from collections import defaultdict
from base_model import WinProbabilityModel


class HMMModel(WinProbabilityModel):
    """
    Hidden Markov Model for win probability.
    States are shot types. Win probability at each step
    estimated from empirical win rates per state.
    """

    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.win_counts = defaultdict(int)
        self.total_counts = defaultdict(int)
        self.baseline_win_rate = 0.5

    def get_name(self):
        return "HMM"

    def fit(self, rallies, outcomes):
        for rally, outcome in zip(rallies, outcomes):
            for i in range(len(rally)):
                state = rally[i]
                self.total_counts[state] += 1
                if outcome == 1:
                    self.win_counts[state] += 1
                if i < len(rally) - 1:
                    next_state = rally[i + 1]
                    self.transition_counts[state][next_state] += 1

        self.baseline_win_rate = sum(outcomes) / len(outcomes) if outcomes else 0.5

    def _state_win_rate(self, state):
        if self.total_counts[state] == 0:
            return self.baseline_win_rate
        return self.win_counts[state] / self.total_counts[state]

    def predict_proba(self, shot_sequence):
        probs = []
        for shot in shot_sequence:
            probs.append(self._state_win_rate(shot))
        return probs


class MarkovChainModel(WinProbabilityModel):
    """
    Markov chain — win probability based on (prev_shot, curr_shot) pairs.
    Falls back to single-shot win rate if pair not seen in training.
    """

    def __init__(self):
        self.pair_win_counts = defaultdict(int)
        self.pair_total_counts = defaultdict(int)
        self.single_win_counts = defaultdict(int)
        self.single_total_counts = defaultdict(int)
        self.baseline_win_rate = 0.5

    def get_name(self):
        return "MarkovChain"

    def fit(self, rallies, outcomes):
        for rally, outcome in zip(rallies, outcomes):
            for i in range(len(rally)):
                s = rally[i]
                self.single_total_counts[s] += 1
                if outcome == 1:
                    self.single_win_counts[s] += 1
                if i > 0:
                    pair = (rally[i - 1], rally[i])
                    self.pair_total_counts[pair] += 1
                    if outcome == 1:
                        self.pair_win_counts[pair] += 1

        self.baseline_win_rate = sum(outcomes) / len(outcomes) if outcomes else 0.5

    def predict_proba(self, shot_sequence):
        probs = []
        for i, shot in enumerate(shot_sequence):
            if i == 0:
                if self.single_total_counts[shot] > 0:
                    p = self.single_win_counts[shot] / self.single_total_counts[shot]
                else:
                    p = self.baseline_win_rate
            else:
                pair = (shot_sequence[i - 1], shot)
                if self.pair_total_counts[pair] > 0:
                    p = self.pair_win_counts[pair] / self.pair_total_counts[pair]
                elif self.single_total_counts[shot] > 0:
                    p = self.single_win_counts[shot] / self.single_total_counts[shot]
                else:
                    p = self.baseline_win_rate
            probs.append(p)
        return probs