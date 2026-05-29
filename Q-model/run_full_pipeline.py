from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "Q-model")

import numpy as np
import pandas as pd

from src.features.state_builder import build_shot_level_table, load_filtered_points
from src.models.calibration import calibration_summary

from decision_table import build_all_shot_decision_table
from pooled_q_model import fit_pooled_q_model, score_decision_table
from per_family_q_model import fit_per_family_q_models, score_with_per_family_models
from sequence_q_baseline import fit_sequence_baselines, score_with_sequence_baselines


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_calibration(name: str, labels: pd.Series, probs: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = probs.notna()
    if mask.sum() < 100:
        print(f"  {name}: insufficient data ({mask.sum()} rows)")
        empty_m = pd.DataFrame(columns=["metric", "value"])
        empty_r = pd.DataFrame()
        return empty_m, empty_r

    metrics, reliability = calibration_summary(labels[mask], probs[mask])

    brier = metrics.loc[metrics["metric"] == "brier_score", "value"].iloc[0]
    ll = metrics.loc[metrics["metric"] == "log_loss", "value"].iloc[0]
    ece = metrics.loc[metrics["metric"] == "expected_calibration_error", "value"].iloc[0]

    print(f"  {name:30s}  Brier={brier:.5f}  LogLoss={ll:.5f}  ECE={ece:.5f}  n={mask.sum():,}")
    return metrics, reliability


def main() -> None:
    total_start = time.time()

    print_section("BUILDING DATA")

    print("Loading points...")
    t0 = time.time()
    points_df = load_filtered_points()
    print(f"  {len(points_df):,} points loaded ({time.time()-t0:.0f}s)")

    print("Building shot-level table...")
    t0 = time.time()
    shot_df = build_shot_level_table(points_df)
    print(f"  {len(shot_df):,} shots built ({time.time()-t0:.0f}s)")

    print("Building all-shot decision table...")
    t0 = time.time()
    decision_df = build_all_shot_decision_table(shot_df)
    print(f"  {len(decision_df):,} decisions built ({time.time()-t0:.0f}s)")

    print_section("STEP 2: POOLED Q-MODEL")

    t0 = time.time()
    pooled_bundle = fit_pooled_q_model(decision_df)
    pooled_scored = score_decision_table(pooled_bundle, decision_df)
    print(f"  Pooled model trained + scored in {time.time()-t0:.0f}s")

    print_section("STEP 3: PER-FAMILY Q-MODELS")

    t0 = time.time()
    family_models = fit_per_family_q_models(decision_df)
    for fam, bundle in family_models.items():
        n = len(decision_df[decision_df["stroke_family"] == fam])
        print(f"  {fam} model trained on {n:,} rows")
    family_scored = score_with_per_family_models(family_models, decision_df)
    print(f"  Per-family models trained + scored in {time.time()-t0:.0f}s")

    print_section("STEP 4: SEQUENCE BASELINES")

    t0 = time.time()
    print("  Training sequence models on full shot data...")
    seq_models = fit_sequence_baselines(shot_df)
    print(f"  Sequence models trained in {time.time()-t0:.0f}s")

    print("  Scoring decisions with Markov baseline (this is slow)...")
    t0 = time.time()
    seq_scored = score_with_sequence_baselines(seq_models, decision_df, shot_df, model_name="markov")
    seq_coverage = seq_scored["seq_q_cc"].notna().sum()
    print(f"  Scored {seq_coverage:,}/{len(decision_df):,} rows in {time.time()-t0:.0f}s")

    print_section("STEP 5: CALIBRATION COMPARISON")

    labels = decision_df["point_win"]

    print("\n--- Overall calibration (all rows) ---")
    print_calibration("Pooled Q-model", labels, pooled_scored["q_observed"])
    print_calibration("Per-family Q-model", labels, family_scored["pf_q_observed"])
    print_calibration("Sequence (Markov)", labels, seq_scored["seq_q_observed"])

    print("\n--- Calibration by stroke family ---")
    for family in ["FH", "BH"]:
        mask = decision_df["stroke_family"] == family
        print(f"\n  [{family}]")
        print_calibration(f"  Pooled", labels[mask], pooled_scored.loc[mask, "q_observed"])
        print_calibration(f"  Per-family", labels[mask], family_scored.loc[mask, "pf_q_observed"])
        print_calibration(f"  Sequence", labels[mask], seq_scored.loc[mask, "seq_q_observed"])

    print("\n--- Calibration by shot depth ---")
    for bucket in ["3", "4-5", "6+"]:
        mask = decision_df["shot_number_bucket"] == bucket
        if mask.sum() == 0:
            continue
        print(f"\n  [Shot {bucket}]")
        print_calibration(f"  Pooled", labels[mask], pooled_scored.loc[mask, "q_observed"])
        print_calibration(f"  Per-family", labels[mask], family_scored.loc[mask, "pf_q_observed"])
        print_calibration(f"  Sequence", labels[mask], seq_scored.loc[mask, "seq_q_observed"])

    print("\n--- Calibration by server/returner ---")
    for role, val in [("Server", 1), ("Returner", 0)]:
        mask = decision_df["hitter_is_server"] == val
        print(f"\n  [{role}]")
        print_calibration(f"  Pooled", labels[mask], pooled_scored.loc[mask, "q_observed"])
        print_calibration(f"  Per-family", labels[mask], family_scored.loc[mask, "pf_q_observed"])

    print("\n--- Reliability table (Pooled model) ---")
    _, reliability = calibration_summary(labels, pooled_scored["q_observed"])
    print(reliability[["bin", "predicted_mean", "observed_rate", "sample_size", "abs_gap"]].to_string(index=False))

    print_section("STEP 6: OUTPUT TABLE")

    output = decision_df.copy()

    output["q_observed"] = pooled_scored["q_observed"]
    output["q_best"] = pooled_scored["q_best"]
    output["regret"] = pooled_scored["regret"]
    output["recommended_action"] = pooled_scored["recommended_action"]
    output["recommended_action_prob"] = pooled_scored["recommended_action_prob"]

    output["pf_q_observed"] = family_scored["pf_q_observed"]
    output["pf_q_best"] = family_scored["pf_q_best"]
    output["pf_regret"] = family_scored["pf_regret"]
    output["pf_recommended"] = family_scored["pf_recommended"]

    output["seq_q_observed"] = seq_scored["seq_q_observed"]
    output["seq_q_best"] = seq_scored["seq_q_best"]
    output["seq_regret"] = seq_scored["seq_regret"]
    output["seq_recommended"] = seq_scored["seq_recommended"]

    output_path = "outputs/tables/all_shot_q_model_output.csv"
    output.to_csv(output_path, index=False)
    print(f"  Saved {len(output):,} rows to {output_path}")
    print(f"  Columns: {len(output.columns)}")

    print("\n  Sample rows (key columns):")
    sample_cols = [
        "hitter", "stroke_family", "shot_number_bucket", "action", "point_win",
        "q_observed", "regret", "recommended_action",
        "pf_q_observed", "pf_regret", "pf_recommended",
        "seq_q_observed", "seq_regret", "seq_recommended",
    ]
    print(output[sample_cols].sample(5, random_state=42).to_string(index=False))

    print_section("STEP 7: THE FOUR QUESTIONS")

    print("\n--- Q1: Does one pooled model work across shot types? ---\n")

    pooled_metrics_fh, _ = calibration_summary(
        labels[decision_df["stroke_family"] == "FH"],
        pooled_scored.loc[decision_df["stroke_family"] == "FH", "q_observed"],
    )
    pooled_metrics_bh, _ = calibration_summary(
        labels[decision_df["stroke_family"] == "BH"],
        pooled_scored.loc[decision_df["stroke_family"] == "BH", "q_observed"],
    )
    pf_metrics_fh, _ = calibration_summary(
        labels[decision_df["stroke_family"] == "FH"],
        family_scored.loc[decision_df["stroke_family"] == "FH", "pf_q_observed"],
    )
    pf_metrics_bh, _ = calibration_summary(
        labels[decision_df["stroke_family"] == "BH"],
        family_scored.loc[decision_df["stroke_family"] == "BH", "pf_q_observed"],
    )

    comparison_rows = []
    for name, m_fh, m_bh in [
        ("Pooled", pooled_metrics_fh, pooled_metrics_bh),
        ("Per-family", pf_metrics_fh, pf_metrics_bh),
    ]:
        for family, m in [("FH", m_fh), ("BH", m_bh)]:
            comparison_rows.append({
                "model": name,
                "family": family,
                "brier": m.loc[m["metric"] == "brier_score", "value"].iloc[0],
                "log_loss": m.loc[m["metric"] == "log_loss", "value"].iloc[0],
                "ece": m.loc[m["metric"] == "expected_calibration_error", "value"].iloc[0],
            })
    comp_df = pd.DataFrame(comparison_rows)
    print(comp_df.to_string(index=False))

    pooled_mean_regret = pooled_scored["regret"].mean()
    pf_mean_regret = family_scored["pf_regret"].mean()
    print(f"\n  Pooled mean regret:     {pooled_mean_regret:.5f}")
    print(f"  Per-family mean regret: {pf_mean_regret:.5f}")

    if pf_mean_regret > pooled_mean_regret * 1.1:
        print("  -> Per-family finds MORE situations where the non-default action is better.")
        print("     The pooled model is too blunt to detect wing-specific edges.")
    else:
        print("  -> Both models find similar (small) regret. The pooled model's averaging")
        print("     is not losing much -- the CC/DTL gap may be similar across wings.")

    print("\n\n--- Q2: Which contexts are poorly calibrated? ---\n")

    print("  Slicing pooled model calibration by (stroke_family x shot_depth):\n")
    slice_rows = []
    for family in ["FH", "BH"]:
        for bucket in ["3", "4-5", "6+"]:
            mask = (decision_df["stroke_family"] == family) & (decision_df["shot_number_bucket"] == bucket)
            n = mask.sum()
            if n < 50:
                continue
            m, r = calibration_summary(labels[mask], pooled_scored.loc[mask, "q_observed"])
            ece = m.loc[m["metric"] == "expected_calibration_error", "value"].iloc[0]
            brier = m.loc[m["metric"] == "brier_score", "value"].iloc[0]
            mean_pred = pooled_scored.loc[mask, "q_observed"].mean()
            mean_actual = labels[mask].mean()
            slice_rows.append({
                "family": family,
                "shot_bucket": bucket,
                "n": n,
                "mean_predicted": round(mean_pred, 4),
                "mean_actual": round(mean_actual, 4),
                "gap": round(mean_pred - mean_actual, 4),
                "brier": round(brier, 5),
                "ece": round(ece, 5),
            })
    slice_df = pd.DataFrame(slice_rows)
    print(slice_df.to_string(index=False))

    worst_slices = slice_df.nlargest(3, "ece")
    print(f"\n  Worst calibrated contexts (by ECE):")
    for _, row in worst_slices.iterrows():
        print(f"    {row['family']} shot {row['shot_bucket']}: ECE={row['ece']:.5f}, "
              f"predicted {row['mean_predicted']:.4f} vs actual {row['mean_actual']:.4f}")

    print("\n\n--- Q3: Does adding shot history improve Q estimates? ---\n")

    seq_valid = seq_scored["seq_q_observed"].notna()
    if seq_valid.sum() > 100:
        print(f"  Comparing on {seq_valid.sum():,} rows where sequence model has predictions:\n")
        print_calibration("Pooled (filtered)", labels[seq_valid], pooled_scored.loc[seq_valid, "q_observed"])
        print_calibration("Per-family (filtered)", labels[seq_valid], family_scored.loc[seq_valid, "pf_q_observed"])
        print_calibration("Sequence (Markov)", labels[seq_valid], seq_scored.loc[seq_valid, "seq_q_observed"])

        seq_brier = calibration_summary(labels[seq_valid], seq_scored.loc[seq_valid, "seq_q_observed"])[0]
        pooled_brier = calibration_summary(labels[seq_valid], pooled_scored.loc[seq_valid, "q_observed"])[0]

        seq_b = seq_brier.loc[seq_brier["metric"] == "brier_score", "value"].iloc[0]
        pooled_b = pooled_brier.loc[pooled_brier["metric"] == "brier_score", "value"].iloc[0]

        if seq_b < pooled_b:
            print(f"\n  -> Sequence model has LOWER Brier score ({seq_b:.5f} vs {pooled_b:.5f}).")
            print("     Shot history DOES help predict outcomes.")
        else:
            print(f"\n  -> Sequence model has HIGHER Brier score ({seq_b:.5f} vs {pooled_b:.5f}).")
            print("     The tabular features capture more information than raw shot history alone.")
    else:
        print("  Insufficient sequence predictions for comparison.")

    print("\n\n--- Q4: Are recommendations stable enough to trust? ---\n")

    both_valid = seq_scored["seq_recommended"].notna()
    if both_valid.sum() > 0:
        agree_pooled_pf = (pooled_scored["recommended_action"] == family_scored["pf_recommended"]).mean()
        agree_pooled_seq = (
            pooled_scored.loc[both_valid, "recommended_action"] == seq_scored.loc[both_valid, "seq_recommended"]
        ).mean()
        agree_pf_seq = (
            family_scored.loc[both_valid, "pf_recommended"] == seq_scored.loc[both_valid, "seq_recommended"]
        ).mean()

        print(f"  Model agreement rates:")
        print(f"    Pooled vs Per-family:   {agree_pooled_pf:.1%}")
        print(f"    Pooled vs Sequence:     {agree_pooled_seq:.1%}")
        print(f"    Per-family vs Sequence: {agree_pf_seq:.1%}")

    print(f"\n  Regret magnitude (how much is at stake):")
    for name, regret_col in [
        ("Pooled", pooled_scored["regret"]),
        ("Per-family", family_scored["pf_regret"]),
        ("Sequence", seq_scored["seq_regret"]),
    ]:
        valid = regret_col.dropna()
        if len(valid) == 0:
            continue
        print(f"    {name:15s}: mean={valid.mean():.5f}  median={valid.median():.5f}  "
              f"p95={valid.quantile(0.95):.5f}  max={valid.max():.5f}")

    print(f"\n  Rows with regret > 1% win probability:")
    for name, regret_col in [
        ("Pooled", pooled_scored["regret"]),
        ("Per-family", family_scored["pf_regret"]),
        ("Sequence", seq_scored["seq_regret"]),
    ]:
        valid = regret_col.dropna()
        above = (valid > 0.01).sum()
        print(f"    {name:15s}: {above:,} ({100*above/len(valid):.1f}%)")

    print(f"\n  Stability assessment:")
    max_regret = pooled_scored["regret"].max()
    pf_max_regret = family_scored["pf_regret"].max()
    if max_regret < 0.05 and pf_max_regret < 0.05:
        print("    All models produce small regret (<5%). This means either:")
        print("    a) Players are already close to optimal in CC/DTL choices, OR")
        print("    b) The models are too simple to detect the real edges.")
    else:
        pct_high = (family_scored["pf_regret"] > 0.03).mean()
        print(f"    {pct_high:.1%} of decisions have per-family regret > 3%.")

    elapsed = time.time() - total_start
    print_section(f"COMPLETE -- {elapsed:.0f}s total")

    print(f"\n  Files created:")
    print(f"    {output_path}")
    print(f"\n  Output table: {len(output):,} rows x {len(output.columns)} columns")
    print(f"  Three model variants compared: Pooled, Per-family, Sequence (Markov)")


if __name__ == "__main__":
    main()
