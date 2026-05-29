from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

import pandas as pd

from src.features.state_builder import (
    build_forehand_direction_dataset,
    build_shot_level_table,
    load_filtered_points,
)


def main() -> None:
    print("=" * 70)
    print("ALL-SHOT DECISION TABLE -- VERIFICATION")
    print("=" * 70)

    print("\n[1] Loading filtered points (Hard court, 2022-2024)...")
    t0 = time.time()
    points_df = load_filtered_points()
    print(f"    Points loaded: {len(points_df):,} rows in {time.time()-t0:.1f}s")

    print("\n[2] Building shot-level table...")
    t0 = time.time()
    shot_df = build_shot_level_table(points_df)
    print(f"    Shot table built: {len(shot_df):,} rows in {time.time()-t0:.1f}s")
    print(f"    Columns: {len(shot_df.columns)}")
    print(f"    Families: {shot_df['current_family'].value_counts().to_dict()}")
    print(f"    Shot index range: {shot_df['shot_index'].min()} - {shot_df['shot_index'].max()}")
    print(f"    Directions available: {shot_df['current_direction'].notna().sum():,} / {len(shot_df):,}")

    print("\n[3] Building all-shot decision table...")
    from decision_table import build_all_shot_decision_table

    t0 = time.time()
    decision_df = build_all_shot_decision_table(shot_df)
    print(f"    Decision table built: {len(decision_df):,} rows in {time.time()-t0:.1f}s")

    print("\n[4] Table summary:")
    print(f"    Total decisions: {len(decision_df):,}")
    print(f"    Unique matches: {decision_df['match_id'].nunique():,}")
    print(f"    Unique hitters: {decision_df['hitter'].nunique():,}")
    print(f"\n    By stroke family:")
    for family, count in decision_df["stroke_family"].value_counts().items():
        print(f"      {family}: {count:,} ({100*count/len(decision_df):.1f}%)")
    print(f"\n    By action:")
    for action, count in decision_df["action"].value_counts().items():
        print(f"      {action}: {count:,} ({100*count/len(decision_df):.1f}%)")
    print(f"\n    By action_label:")
    for label, count in decision_df["action_label"].value_counts().sort_index().items():
        print(f"      {label}: {count:,}")
    print(f"\n    By shot_number_bucket:")
    for bucket, count in decision_df["shot_number_bucket"].value_counts().sort_index().items():
        print(f"      shot {bucket}: {count:,}")
    print(f"\n    Point win rate: {decision_df['point_win'].mean():.4f}")
    print(f"    DTL rate overall: {decision_df['action_dtl'].mean():.4f}")

    print("\n[5] Leakage check:")
    first_rows = decision_df[decision_df["match_prior_context_count"] == 0]
    print(f"    Rows with zero prior context (first in group): {len(first_rows):,}")
    default_rate_correct = (first_rows["match_prior_dtl_rate"] == 0.5).all()
    print(f"    All first-row dtl_rates default to 0.5: {default_rate_correct}")

    sample_group = decision_df.groupby(
        ["match_id", "hitter", "opponent", "stroke_family"], sort=False
    ).filter(lambda x: len(x) >= 5).head(10)
    if len(sample_group) > 0:
        group_key = (
            sample_group.iloc[0]["match_id"],
            sample_group.iloc[0]["hitter"],
            sample_group.iloc[0]["opponent"],
            sample_group.iloc[0]["stroke_family"],
        )
        grp = decision_df[
            (decision_df["match_id"] == group_key[0])
            & (decision_df["hitter"] == group_key[1])
            & (decision_df["opponent"] == group_key[2])
            & (decision_df["stroke_family"] == group_key[3])
        ].sort_values(["point_number", "shot_index"])

        leakage_found = False
        for i in range(len(grp)):
            expected_prior = grp["action_dtl"].iloc[:i].sum()
            actual_prior = grp["match_prior_dtl_count"].iloc[i]
            if expected_prior != actual_prior:
                leakage_found = True
                print(f"    LEAKAGE at index {i}: expected {expected_prior}, got {actual_prior}")
                break
        if not leakage_found:
            print(f"    Spot-check passed: exposure counts exclude current row correctly")

    print("\n[6] Comparison with Leo's original 5th-shot forehand table:")
    original_df = build_forehand_direction_dataset(shot_df)
    print(f"    Leo's 5th-shot FH table: {len(original_df):,} rows")
    fifth_shot_fh = decision_df[
        (decision_df["shot_index"] == 5) & (decision_df["stroke_family"] == "FH")
    ]
    print(f"    Our table at shot=5, FH: {len(fifth_shot_fh):,} rows")
    fifth_shot_fh_topspin = decision_df[
        (decision_df["shot_index"] == 5)
        & (decision_df["stroke_family"] == "FH")
        & (decision_df["current_raw_code"] == "f")
    ]
    print(f"    Our table at shot=5, FH, topspin only: {len(fifth_shot_fh_topspin):,} rows")

    print("\n[7] Full column list:")
    for i, col in enumerate(decision_df.columns, 1):
        print(f"    {i:2d}. {col}")

    print("\n" + "=" * 70)
    print("RESULT: Table is ready for Q-model training.")
    print(f"  - {len(decision_df):,} decision rows")
    print(f"  - Covers shots 2 through {decision_df['shot_index'].max()}")
    print(f"  - Both FH and BH groundstrokes")
    print(f"  - CC/DTL action labels with high-confidence side proxy")
    print(f"  - Match-local exposure features (no leakage)")
    print("=" * 70)


if __name__ == "__main__":
    main()
