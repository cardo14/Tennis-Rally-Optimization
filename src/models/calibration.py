from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


def fit_isotonic_calibrator(raw_probabilities: pd.Series, labels: pd.Series) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probabilities, labels)
    return calibrator


def calibration_summary(
    labels: pd.Series,
    probabilities: pd.Series,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.Series(labels).astype(float)
    probabilities = pd.Series(probabilities).clip(1e-6, 1 - 1e-6)

    metrics = pd.DataFrame(
        [
            {"metric": "log_loss", "value": float(log_loss(labels, probabilities))},
            {"metric": "brier_score", "value": float(brier_score_loss(labels, probabilities))},
        ]
    )

    bins = pd.cut(probabilities, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    reliability = (
        pd.DataFrame({"label": labels, "probability": probabilities, "bin": bins})
        .groupby("bin", observed=False)
        .agg(
            predicted_mean=("probability", "mean"),
            observed_rate=("label", "mean"),
            sample_size=("label", "size"),
        )
        .reset_index()
    )
    reliability["abs_gap"] = (reliability["predicted_mean"] - reliability["observed_rate"]).abs()
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {
                        "metric": "expected_calibration_error",
                        "value": float(
                            (reliability["abs_gap"] * reliability["sample_size"]).sum()
                            / reliability["sample_size"].sum()
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return metrics, reliability

