from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from src.features.sequence_builder import PAD_TOKEN, SHOT_TOKEN_LABELS


def _state_matrix(examples: pd.DataFrame, num_tokens: int = 10) -> np.ndarray:
    states = list(examples["state_tokens"])
    if not states:
        return np.empty((0, 0))

    window = len(states[0])
    matrix = np.zeros((len(states), window * num_tokens), dtype=float)
    for row_idx, state in enumerate(states):
        for pos, token in enumerate(state):
            token_value = int(token)
            if token_value == PAD_TOKEN:
                continue
            if 0 <= token_value < num_tokens:
                matrix[row_idx, pos * num_tokens + token_value] = 1.0
    return matrix


@dataclass
class NextShotPolicyModel:
    """Fast all-shot behavior-policy model inspired by Sai's next-shot work.

    This is intentionally lightweight for the core pipeline. It estimates the
    logged next-shot distribution from prior shot history without using future
    shots. Neural versions can implement the same fit/predict API later.
    """

    num_tokens: int = 10
    max_iter: int = 1000
    alpha: float = 0.0005
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = SGDClassifier(
            loss="log_loss",
            alpha=self.alpha,
            max_iter=self.max_iter,
            tol=1e-3,
            random_state=self.random_state,
        )
        self.classes_: np.ndarray | None = None

    def fit(self, examples: pd.DataFrame) -> None:
        x = _state_matrix(examples, self.num_tokens)
        y = examples["next_token"].astype(int).to_numpy()
        self.model.fit(x, y)
        self.classes_ = self.model.classes_

    def predict_proba(self, examples: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(_state_matrix(examples, self.num_tokens))

    def evaluate(self, examples: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba(examples)
        y_true = examples["next_token"].astype(int).to_numpy()
        classes = np.asarray(self.classes_, dtype=int)
        top_order = np.argsort(probabilities, axis=1)[:, ::-1]
        top1 = classes[top_order[:, 0]]
        top3 = classes[top_order[:, : min(3, len(classes))]]
        top3_match = np.array([target in row for target, row in zip(y_true, top3)])

        return pd.DataFrame(
            [
                {
                    "model": "NextShotPolicyLogistic",
                    "test_examples": int(len(examples)),
                    "cross_entropy": float(log_loss(y_true, probabilities, labels=classes)),
                    "top1_accuracy": float((top1 == y_true).mean()),
                    "top3_accuracy": float(top3_match.mean()),
                    "num_classes": int(len(classes)),
                }
            ]
        )


def next_shot_confusion_table(model: NextShotPolicyModel, examples: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(examples)
    classes = np.asarray(model.classes_, dtype=int)
    predictions = classes[np.argmax(probabilities, axis=1)]
    table = pd.crosstab(
        pd.Series(examples["next_token"].astype(int).map(SHOT_TOKEN_LABELS), name="actual"),
        pd.Series(pd.Series(predictions).map(SHOT_TOKEN_LABELS), name="predicted"),
        dropna=False,
    )
    return table.reset_index()
