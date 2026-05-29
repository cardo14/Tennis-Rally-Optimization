from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "Q-model")

import numpy as np
import pandas as pd

from src.features.state_builder import build_shot_level_table, load_filtered_points
from decision_table import build_all_shot_decision_table
from pooled_q_model import fit_pooled_q_model, score_decision_table


def main() -> None:
    print("=" * 70)
    print("POOLED ALL-SHOT Q-MODEL -- VERIFICATION")
    print("=" * 70)

    print("\n[1] Building decision table...")
    t0 = time.time()
    points_df = load_filtered_points()
    shot_df = build_shot_level_table(points_df)
    decision_df = build_all_shot_decision_table(shot_df)
    print(f"    Decision table: {len(decision_df):,} rows in {time.time()-t0:.1f}s")

    print("\n[2] Training pooled Q-model...")
    t0 = time.time()
    q_bundle = fit_pooled_q_model(decision_df)
    print(f"    Model trained in {time.time()-t0:.1f}s")
    print(f"    Feature columns: {len(q_bundle['feature_columns'])}")
    print(f"    Candidate actions: {q_bundle['candidate_actions']}")

    print("\n[3] Scoring all rows under all actions...")
    t0 = time.time()
    scored_df = score_decision_table(q_bundle, decision_df)
    print(f"    Scoring complete in {time.time()-t0:.1f}s")

    print("\n[4] Validity checks:")

    q_cols = ["q_FH_CC", "q_FH_DTL", "q_BH_CC", "q_BH_DTL", "q_cc", "q_dtl", "q_observed", "q_best"]
    for col in q_cols:
        lo, hi = scored_df[col].min(), scored_df[col].max()
        in_range = (lo >= 0) and (hi <= 1)
        status = "PASS" if in_range else "FAIL"
        print(f"    [{status}] {col}: range [{lo:.4f}, {hi:.4f}]")

    regret_min = scored_df["regret"].min()
    regret_pass = regret_min >= -1e-10
    print(f"    [{'PASS' if regret_pass else 'FAIL'}] regret >= 0: min = {regret_min:.6f}")

    best_ge_obs = (scored_df["q_best"] >= scored_df["q_observed"] - 1e-10).all()
    print(f"    [{'PASS' if best_ge_obs else 'FAIL'}] q_best >= q_observed always")

    dtl_better = scored_df["q_dtl"] > scored_df["q_cc"]
    rec_matches = (scored_df["recommended_action"] == np.where(dtl_better, "DTL", "CC")).all()
    print(f"    [{'PASS' if rec_matches else 'FAIL'}] recommended_action matches argmax(q_cc, q_dtl)")

    print("\n[5] Summary statistics:")
    print(f"    Mean Q (observed):     {scored_df['q_observed'].mean():.4f}")
    print(f"    Mean Q (best):         {scored_df['q_best'].mean():.4f}")
    print(f"    Mean regret:           {scored_df['regret'].mean():.4f}")
    print(f"    Median regret:         {scored_df['regret'].median():.4f}")
    print(f"    Regret > 0.05:         {(scored_df['regret'] > 0.05).sum():,} rows ({100*(scored_df['regret'] > 0.05).mean():.1f}%)")
    print(f"    Regret > 0.10:         {(scored_df['regret'] > 0.10).sum():,} rows ({100*(scored_df['regret'] > 0.10).mean():.1f}%)")

    print(f"\n    Recommendation breakdown:")
    rec_counts = scored_df["recommended_action"].value_counts()
    for action, count in rec_counts.items():
        print(f"      Recommend {action}: {count:,} ({100*count/len(scored_df):.1f}%)")

    print(f"\n    Players follow recommendation: {scored_df['took_recommended'].mean():.1%} of the time")

    print("\n[6] Breakdown by stroke family:")
    for family in ["FH", "BH"]:
        subset = scored_df[scored_df["stroke_family"] == family]
        print(f"\n    {family} ({len(subset):,} rows):")
        print(f"      Mean Q(CC):     {subset['q_cc'].mean():.4f}")
        print(f"      Mean Q(DTL):    {subset['q_dtl'].mean():.4f}")
        print(f"      Mean regret:    {subset['regret'].mean():.4f}")
        print(f"      DTL rate:       {subset['action_dtl'].mean():.3f}")
        print(f"      Recommend DTL:  {(subset['recommended_action'] == 'DTL').mean():.1%}")
        print(f"      Follow rec:     {subset['took_recommended'].mean():.1%}")

    print("\n[7] Breakdown by shot depth:")
    for bucket in ["3", "4-5", "6+"]:
        subset = scored_df[scored_df["shot_number_bucket"] == bucket]
        if len(subset) == 0:
            continue
        print(f"\n    Shot {bucket} ({len(subset):,} rows):")
        print(f"      Mean regret:    {subset['regret'].mean():.4f}")
        print(f"      Actual win%:    {subset['point_win'].mean():.4f}")
        print(f"      Mean Q(obs):    {subset['q_observed'].mean():.4f}")
        print(f"      Recommend DTL:  {(subset['recommended_action'] == 'DTL').mean():.1%}")

    print("\n[8] Sample output (5 random rows):")
    sample_cols = [
        "hitter", "stroke_family", "shot_number_bucket", "action",
        "q_cc", "q_dtl", "q_observed", "q_best", "regret",
        "recommended_action", "point_win",
    ]
    sample = scored_df[sample_cols].sample(5, random_state=42)
    print(sample.to_string(index=False))

    print("\n" + "=" * 70)
    print("POOLED Q-MODEL VERIFICATION COMPLETE")
    print(f"Output columns added: {[c for c in scored_df.columns if c not in decision_df.columns]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
