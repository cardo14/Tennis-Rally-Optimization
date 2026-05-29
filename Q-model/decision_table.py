from __future__ import annotations

import pandas as pd

from src.parsing.mcp_parser import direction_to_court_side


def classify_groundstroke_action(
    current_direction: int | None,
    side_proxy: str | None,
    current_family: str,
) -> str | None:
    if current_family not in {"FH", "BH"}:
        return None
    if current_direction is None or side_proxy not in {"deuce", "ad"}:
        return None
    if current_direction == 2:
        return "BODY"
    target_side = direction_to_court_side(current_direction)
    if target_side is None:
        return None
    return "CC" if target_side == side_proxy else "DTL"


def add_general_exposure_features(dataset: pd.DataFrame, recent_window: int = 8) -> pd.DataFrame:
    if dataset.empty:
        return dataset.copy()

    exposure_df = dataset.sort_values(
        ["match_id", "hitter", "opponent", "stroke_family", "point_number", "shot_index"]
    ).copy()
    exposure_df["action_dtl"] = (exposure_df["action"] == "DTL").astype(int)

    feature_frames = []
    group_columns = ["match_id", "hitter", "opponent", "stroke_family"]

    for _, group in exposure_df.groupby(group_columns, sort=False, dropna=False):
        group = group.sort_values(["point_number", "shot_index"]).copy()

        prior_count = pd.Series(range(len(group)), index=group.index, dtype=float)
        prior_dtl_count = group["action_dtl"].cumsum().shift(fill_value=0).astype(float)
        prior_cc_count = prior_count - prior_dtl_count

        shifted_actions = group["action_dtl"].shift()
        recent_dtl_count = (
            shifted_actions.rolling(window=recent_window, min_periods=1).sum().fillna(0.0).astype(float)
        )
        recent_context_count = (
            shifted_actions.rolling(window=recent_window, min_periods=1).count().fillna(0.0).astype(float)
        )

        group["match_prior_context_count"] = prior_count
        group["match_prior_dtl_count"] = prior_dtl_count
        group["match_prior_cc_count"] = prior_cc_count
        group["match_prior_dtl_rate"] = (
            prior_dtl_count / prior_count.where(prior_count > 0)
        ).fillna(0.5)
        group["recent_context_count"] = recent_context_count
        group["recent_dtl_count"] = recent_dtl_count
        group["recent_cc_count"] = recent_context_count - recent_dtl_count
        group["recent_dtl_rate"] = (
            recent_dtl_count / recent_context_count.where(recent_context_count > 0)
        ).fillna(0.5)
        group["has_match_exposure"] = (prior_count > 0).astype(int)
        group["has_recent_exposure"] = (recent_context_count > 0).astype(int)
        feature_frames.append(group)

    return pd.concat(feature_frames).sort_index()


def build_all_shot_decision_table(
    shot_df: pd.DataFrame,
    min_shot_index: int = 2,
    require_high_confidence: bool = True,
) -> pd.DataFrame:
    df = shot_df[
        (shot_df["shot_index"] >= min_shot_index)
        & (shot_df["current_family"].isin(["FH", "BH"]))
        & (shot_df["current_raw_code"].isin(["f", "b", "r", "s"]))
        & (shot_df["current_direction"].notna())
    ].copy()

    df["action"] = df.apply(
        lambda row: classify_groundstroke_action(
            row["current_direction"],
            row["side_proxy"],
            row["current_family"],
        ),
        axis=1,
    )

    if require_high_confidence:
        df = df[df["side_proxy_confidence"] == "high"].copy()

    df = df[df["action"].isin(["CC", "DTL"])].copy()

    df["stroke_family"] = df["current_family"]
    df["action_dtl"] = (df["action"] == "DTL").astype(int)
    df["action_label"] = df["stroke_family"] + "_" + df["action"]

    df = add_general_exposure_features(df)

    df["context_name"] = "all_shot_groundstroke_direction"

    return df.reset_index(drop=True)
