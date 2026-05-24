from __future__ import annotations

import numpy as np
import pandas as pd


def clip_probability(probabilities: pd.Series | np.ndarray, floor: float = 0.025) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), floor, 1.0 - floor)


def soft_recommendation_policy(q_cc: pd.Series, q_dtl: pd.Series, tau: float = 0.02) -> pd.Series:
    delta = (q_dtl - q_cc) / tau
    return pd.Series(1.0 / (1.0 + np.exp(-delta)), index=q_cc.index)


def doubly_robust_scores(
    observed_action_dtl: pd.Series,
    observed_reward: pd.Series,
    mu_dtl: pd.Series,
    q_cc: pd.Series,
    q_dtl: pd.Series,
    pi_dtl: pd.Series,
    weight_clip: float = 10.0,
) -> pd.Series:
    observed_action_dtl = pd.Series(observed_action_dtl).astype(int)
    observed_reward = pd.Series(observed_reward).astype(float)
    mu_dtl = pd.Series(clip_probability(mu_dtl), index=observed_action_dtl.index)
    q_cc = pd.Series(q_cc, index=observed_action_dtl.index)
    q_dtl = pd.Series(q_dtl, index=observed_action_dtl.index)
    pi_dtl = pd.Series(pi_dtl, index=observed_action_dtl.index)

    q_obs = np.where(observed_action_dtl == 1, q_dtl, q_cc)
    q_target = pi_dtl * q_dtl + (1.0 - pi_dtl) * q_cc
    pi_obs = np.where(observed_action_dtl == 1, pi_dtl, 1.0 - pi_dtl)
    mu_obs = np.where(observed_action_dtl == 1, mu_dtl, 1.0 - mu_dtl)
    weights = np.clip(pi_obs / mu_obs, 0.0, weight_clip)
    dr = q_target + weights * (observed_reward - q_obs)
    return pd.Series(dr, index=observed_action_dtl.index)


def ipw_scores(
    observed_action_dtl: pd.Series,
    observed_reward: pd.Series,
    mu_dtl: pd.Series,
    pi_dtl: pd.Series,
    weight_clip: float = 10.0,
) -> pd.Series:
    observed_action_dtl = pd.Series(observed_action_dtl).astype(int)
    observed_reward = pd.Series(observed_reward).astype(float)
    mu_dtl = pd.Series(clip_probability(mu_dtl), index=observed_action_dtl.index)
    pi_dtl = pd.Series(pi_dtl, index=observed_action_dtl.index)
    pi_obs = np.where(observed_action_dtl == 1, pi_dtl, 1.0 - pi_dtl)
    mu_obs = np.where(observed_action_dtl == 1, mu_dtl, 1.0 - mu_dtl)
    weights = np.clip(pi_obs / mu_obs, 0.0, weight_clip)
    return pd.Series(weights * observed_reward, index=observed_action_dtl.index)


def bootstrap_interval(values: pd.Series, n_boot: int = 300, random_state: int = 42) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)
    means = []
    for _ in range(n_boot):
        draw = sample[rng.integers(0, len(sample), size=len(sample))]
        means.append(draw.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

