from __future__ import annotations

import pandas as pd


def _future_exposure_features(context_df: pd.DataFrame, future_window: int = 5) -> pd.DataFrame:
    exposure_df = context_df.sort_values(["match_id", "hitter", "opponent", "side_proxy", "point_number"]).copy()
    frames = []
    group_columns = ["match_id", "hitter", "opponent", "side_proxy"]
    for _, group in exposure_df.groupby(group_columns, sort=False, dropna=False):
        group = group.sort_values("point_number").copy()
        next_actions = group["action_dtl"].shift(-1)
        future_dtl_count = (
            next_actions.iloc[::-1].rolling(window=future_window, min_periods=1).sum().iloc[::-1].fillna(0.0)
        )
        future_context_count = (
            next_actions.iloc[::-1].rolling(window=future_window, min_periods=1).count().iloc[::-1].fillna(0.0)
        )
        group["future_context_count"] = future_context_count.astype(float)
        group["future_dtl_count"] = future_dtl_count.astype(float)
        group["future_dtl_rate"] = (
            group["future_dtl_count"] / group["future_context_count"].where(group["future_context_count"] > 0)
        ).fillna(0.5)
        group["has_future_exposure"] = (group["future_context_count"] > 0).astype(int)
        frames.append(group)
    return pd.concat(frames).sort_index()


def build_opponent_response_dataset(context_df: pd.DataFrame, shot_df: pd.DataFrame) -> pd.DataFrame:
    response_columns = [
        "point_id",
        "current_raw_code",
        "current_family",
        "current_stroke_type",
        "current_direction",
        "current_target_side",
        "current_depth",
        "current_net_state",
        "ending",
        "error_type",
    ]
    response_df = shot_df[shot_df["shot_index"] == 6][response_columns].rename(
        columns={
            "current_raw_code": "opponent_response_raw_code",
            "current_family": "opponent_response_family",
            "current_stroke_type": "opponent_response_stroke_type",
            "current_direction": "opponent_response_direction",
            "current_target_side": "opponent_response_target_side",
            "current_depth": "opponent_response_depth",
            "current_net_state": "opponent_response_net_state",
            "ending": "opponent_response_ending",
            "error_type": "opponent_response_error_type",
        }
    )
    terminal_df = (
        shot_df.groupby("point_id", as_index=False)
        .agg(terminal_shot_index=("shot_index", "max"))
    )

    response_model_df = context_df.merge(response_df, on="point_id", how="left").merge(
        terminal_df, on="point_id", how="left"
    )
    response_model_df = _future_exposure_features(response_model_df)

    response_model_df["high_recent_dtl"] = (
        (response_model_df["recent_context_count"] > 0) & (response_model_df["recent_dtl_rate"] >= 2.0 / 3.0)
    ).astype(int)
    response_model_df["low_recent_dtl"] = (
        (response_model_df["recent_context_count"] > 0) & (response_model_df["recent_dtl_rate"] <= 1.0 / 3.0)
    ).astype(int)
    response_model_df["future_high_dtl"] = (
        (response_model_df["future_context_count"] > 0) & (response_model_df["future_dtl_rate"] >= 2.0 / 3.0)
    ).astype(int)
    response_model_df["action_high_recent_dtl"] = (
        response_model_df["action_dtl"] * response_model_df["high_recent_dtl"]
    )
    response_model_df["action_future_high_dtl"] = (
        response_model_df["action_dtl"] * response_model_df["future_high_dtl"]
    )

    response_model_df["opponent_response_exists"] = response_model_df["opponent_response_raw_code"].notna().astype(int)
    response_error = response_model_df["opponent_response_ending"].isin(["forced_error", "unforced_error"])
    response_model_df["opponent_made_next_ball"] = (
        (response_model_df["opponent_response_exists"] == 1) & ~response_error
    ).astype(int)
    response_model_df["opponent_next_error"] = response_error.astype(int)
    response_model_df["opponent_next_winner"] = (
        response_model_df["opponent_response_ending"].eq("winner")
    ).astype(int)
    response_model_df["hitter_forced_error_next"] = (
        response_model_df["opponent_response_ending"].eq("forced_error")
    ).astype(int)
    response_model_df["opponent_won_within_2"] = (
        response_model_df["point_winner"].eq(response_model_df["opponent"])
        & response_model_df["terminal_shot_index"].le(7)
    ).astype(int)
    response_model_df["hitter_won_within_2"] = (
        response_model_df["point_winner"].eq(response_model_df["hitter"])
        & response_model_df["terminal_shot_index"].le(7)
    ).astype(int)
    response_model_df["opponent_neutralized"] = (
        (response_model_df["opponent_made_next_ball"] == 1)
        & (response_model_df["hitter_won_within_2"] == 0)
    ).astype(int)

    response_model_df["opponent_response_quality_proxy"] = "no_response"
    response_model_df.loc[response_model_df["opponent_next_error"] == 1, "opponent_response_quality_proxy"] = "error"
    response_model_df.loc[
        (response_model_df["opponent_made_next_ball"] == 1)
        & (response_model_df["opponent_next_winner"] == 0),
        "opponent_response_quality_proxy",
    ] = "in_play"
    response_model_df.loc[response_model_df["opponent_next_winner"] == 1, "opponent_response_quality_proxy"] = "winner"

    return response_model_df.reset_index(drop=True)
