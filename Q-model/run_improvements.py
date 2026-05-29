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
from train_test_split import time_split, evaluate_on_test, print_comparison
from pooled_q_model import fit_pooled_q_model, predict_q
from interaction_q_model import (
    add_interaction_features,
    fit_interaction_q_model,
    predict_q_interaction,
)
from xgboost_q_model import fit_xgb_q_model, predict_q_xgb


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main() -> None:
    total_start = time.time()

    print_section("BUILDING DATA")

    t0 = time.time()
    points_df = load_filtered_points()
    shot_df = build_shot_level_table(points_df)
    decision_df = build_all_shot_decision_table(shot_df)
    print(f"  Decision table: {len(decision_df):,} rows ({time.time()-t0:.0f}s)")

    print_section("TRAIN/TEST SPLIT (Improvement 1)")

    train_df, test_df = time_split(decision_df, split_year=2024)
    print(f"  Train (2022-2023): {len(train_df):,} rows, {train_df['match_id'].nunique()} matches")
    print(f"  Test  (2024):      {len(test_df):,} rows, {test_df['match_id'].nunique()} matches")
    print(f"  Train win rate:    {train_df['point_win'].mean():.4f}")
    print(f"  Test  win rate:    {test_df['point_win'].mean():.4f}")

    train_players = set(train_df["hitter"].unique())
    test_players = set(test_df["hitter"].unique())
    overlap = train_players & test_players
    new_in_test = test_players - train_players
    print(f"  Players in train:  {len(train_players)}")
    print(f"  Players in test:   {len(test_players)}")
    print(f"  Overlap:           {len(overlap)} ({100*len(overlap)/len(test_players):.0f}% of test)")
    print(f"  New in test:       {len(new_in_test)} (unseen players)")

    print_section("MODEL A: Pooled Logistic Regression")

    t0 = time.time()
    results_a = evaluate_on_test(train_df, test_df, fit_pooled_q_model, predict_q, "A: Pooled LogReg")
    print(f"  Completed in {time.time()-t0:.0f}s")
    print_comparison(results_a)

    print_section("MODEL B: LogReg + Interactions")

    t0 = time.time()
    results_b = evaluate_on_test(
        train_df, test_df,
        fit_interaction_q_model, predict_q_interaction,
        "B: LogReg + Interactions",
    )
    print(f"  Completed in {time.time()-t0:.0f}s")
    print_comparison(results_b)

    print_section("MODEL C: Gradient-Boosted Trees")

    t0 = time.time()
    results_c = evaluate_on_test(
        train_df, test_df,
        fit_xgb_q_model, predict_q_xgb,
        "C: HistGBT",
    )
    n_trees = results_c["bundle"].get("n_trees", "?")
    print(f"  Completed in {time.time()-t0:.0f}s (used {n_trees} trees)")
    print_comparison(results_c)

    print_section("MODEL D: GBT + Interactions")

    def fit_xgb_with_interactions(df: pd.DataFrame):
        df_with = add_interaction_features(df)
        from xgboost_q_model import Q_TREE_FEATURES, CATEGORICAL_MASK
        from sklearn.ensemble import HistGradientBoostingClassifier

        extra_features = [
            "hitter_action", "opponent_action", "family_side_action",
            "shot_depth_action", "exposure_x_action",
        ]
        extra_is_cat = [
            False,
            False,
            True,
            True,
            True,
        ]
        extended_features = Q_TREE_FEATURES + extra_features
        extended_mask = list(CATEGORICAL_MASK) + extra_is_cat

        mappings = {}
        encoded = df_with[extended_features].copy()
        for i, col in enumerate(extended_features):
            if extended_mask[i] or col in extra_features:
                cat = encoded[col].astype("category")
                mappings[col] = dict(zip(cat.cat.categories, range(len(cat.cat.categories))))
                encoded[col] = cat.cat.codes
            else:
                encoded[col] = pd.to_numeric(encoded[col], errors="coerce")

        cat_indices = [i for i, is_cat in enumerate(extended_mask) if is_cat]

        model = HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.05,
            min_samples_leaf=50, categorical_features=cat_indices,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
        )
        model.fit(encoded, df_with["point_win"].values)

        return {
            "model": model,
            "mappings": mappings,
            "feature_columns": extended_features,
            "feature_mask": extended_mask,
            "candidate_actions": ["FH_CC", "FH_DTL", "BH_CC", "BH_DTL"],
            "n_trees": model.n_iter_,
        }

    def predict_xgb_with_interactions(bundle, df, action_label=None):
        score_df = df.copy()
        if action_label is not None:
            score_df = score_df.copy()
            score_df["action_label"] = action_label
            score_df["action"] = action_label.split("_")[1]

        score_with = add_interaction_features(score_df)
        features = bundle["feature_columns"]
        mask = bundle["feature_mask"]
        mappings = bundle["mappings"]

        encoded = score_with[features].copy()
        for i, col in enumerate(features):
            mapping = mappings.get(col)
            if mapping is not None:
                encoded[col] = encoded[col].map(mapping).fillna(-1).astype(int)
            else:
                encoded[col] = pd.to_numeric(encoded[col], errors="coerce")

        probs = bundle["model"].predict_proba(encoded)[:, 1]
        return pd.Series(probs, index=df.index)

    t0 = time.time()
    results_d = evaluate_on_test(
        train_df, test_df,
        fit_xgb_with_interactions, predict_xgb_with_interactions,
        "D: HistGBT + Interactions",
    )
    n_trees = results_d["bundle"].get("n_trees", "?")
    print(f"  Completed in {time.time()-t0:.0f}s (used {n_trees} trees)")
    print_comparison(results_d)

    print_section("SIDE-BY-SIDE COMPARISON (TEST SET)")

    rows = []
    for label, res in [
        ("A: Pooled LogReg", results_a),
        ("B: LogReg+Interact", results_b),
        ("C: HistGBT", results_c),
        ("D: HistGBT+Interact", results_d),
    ]:
        test_m = res["test_metrics"]
        test_s = res["test_scored"]
        train_m = res["train_metrics"]

        test_brier = test_m.loc[test_m["metric"] == "brier_score", "value"].iloc[0]
        test_ll = test_m.loc[test_m["metric"] == "log_loss", "value"].iloc[0]
        test_ece = test_m.loc[test_m["metric"] == "expected_calibration_error", "value"].iloc[0]
        train_brier = train_m.loc[train_m["metric"] == "brier_score", "value"].iloc[0]

        pct_dtl = (test_s["recommended_action"] == "DTL").mean()
        mean_regret = test_s["regret"].mean()
        regret_1pct = (test_s["regret"] > 0.01).mean()

        rows.append({
            "model": label,
            "test_brier": round(test_brier, 5),
            "test_logloss": round(test_ll, 5),
            "test_ece": round(test_ece, 5),
            "train_brier": round(train_brier, 5),
            "overfit": round(test_brier - train_brier, 5),
            "mean_regret": round(mean_regret, 5),
            "pct_rec_DTL": f"{pct_dtl:.1%}",
            "regret_gt1pct": f"{regret_1pct:.1%}",
        })

    summary = pd.DataFrame(rows)
    print("\n" + summary.to_string(index=False))

    print_section("INTERPRETATION")

    best_model_idx = summary["test_brier"].idxmin()
    best = summary.iloc[best_model_idx]
    worst = summary.iloc[summary["test_brier"].idxmax()]

    print(f"\n  Best model on test set:  {best['model']} (Brier={best['test_brier']})")
    print(f"  Worst model on test set: {worst['model']} (Brier={worst['test_brier']})")
    print(f"  Improvement:             {worst['test_brier'] - best['test_brier']:.5f} Brier points")

    any_dtl = any(r["pct_rec_DTL"] != "0.0%" for r in rows)
    if any_dtl:
        print("\n  PROGRESS: At least one model now recommends DTL in some contexts!")
        for r in rows:
            if r["pct_rec_DTL"] != "0.0%":
                print(f"    {r['model']}: recommends DTL in {r['pct_rec_DTL']} of test decisions")
    else:
        print("\n  All models still recommend CC for every decision.")

    print("\n  Overfitting check (test Brier - train Brier):")
    for r in rows:
        status = "OK" if r["overfit"] < 0.003 else "WARNING"
        print(f"    [{status}] {r['model']}: {r['overfit']:+.5f}")

    elapsed = time.time() - total_start
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
