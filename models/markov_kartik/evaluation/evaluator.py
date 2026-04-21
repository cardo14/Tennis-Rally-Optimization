import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def brier_score(y_true, y_pred):
    return np.mean((np.array(y_pred) - np.array(y_true)) ** 2)


def compute_evg(model, rallies, outcomes, all_possible_shots):
    total_gain = 0
    total_shots = 0

    for rally, outcome in zip(rallies, outcomes):
        if len(rally) == 0:
            continue

        probs = model.predict_proba(rally)
        
        if len(probs) == 0:
            continue

        for i in range(min(len(rally), len(probs))):
            prev_prob = probs[i - 1] if i > 0 else 0.5
            actual_delta = probs[i] - prev_prob

            best_delta = actual_delta
            for alt_shot in all_possible_shots:
                alt_rally = rally[:i] + [alt_shot] + rally[i+1:]
                alt_probs = model.predict_proba(alt_rally[:i+1])
                if len(alt_probs) == 0:
                    continue
                alt_delta = alt_probs[-1] - prev_prob
                if alt_delta > best_delta:
                    best_delta = alt_delta

            total_gain += best_delta - actual_delta
            total_shots += 1

    return total_gain / total_shots if total_shots > 0 else 0.0


def compute_shot_decision_accuracy(model, rallies, outcomes, all_possible_shots):
    """
    % of shots where the model's recommendation matches
    the shot that actually maximized win probability.
    """
    correct = 0
    total = 0

    for rally, outcome in zip(rallies, outcomes):
        if len(rally) < 2:
            continue

        for i in range(1, len(rally)):
            prev_prob = model.predict_proba(rally[:i])[-1]
            actual_shot = rally[i]

            best_shot = None
            best_delta = -999
            for alt_shot in all_possible_shots:
                alt_rally = rally[:i] + [alt_shot]
                alt_prob = model.predict_proba(alt_rally)[-1]
                delta = alt_prob - prev_prob
                if delta > best_delta:
                    best_delta = delta
                    best_shot = alt_shot

            if best_shot == actual_shot:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def evaluate_models(models, test_rallies, test_outcomes, all_possible_shots, verbose=True):
    results = []

    for model in models:
        if verbose:
            print(f"\nEvaluating {model.get_name()}...")

        y_pred_final = [model.predict_final(r) for r in test_rallies]
        y_pred_clipped = np.clip(y_pred_final, 1e-7, 1 - 1e-7)

        bs = brier_score(test_outcomes, y_pred_final)
        ll = log_loss(test_outcomes, y_pred_clipped)

        try:
            auc = roc_auc_score(test_outcomes, y_pred_final)
        except Exception:
            auc = float("nan")

        sample = min(100, len(test_rallies))
        evg = compute_evg(model, test_rallies[:sample], test_outcomes[:sample], all_possible_shots)
        sda = compute_shot_decision_accuracy(model, test_rallies[:sample], test_outcomes[:sample], all_possible_shots)

        if verbose:
            print(f"  Brier Score: {bs:.4f}")
            print(f"  Log Loss:    {ll:.4f}")
            print(f"  AUC:         {auc:.4f}")
            print(f"  EVG:         {evg:.4f}")
            print(f"  Shot Decision Acc: {sda:.4f}")

        results.append({
            "Model": model.get_name(),
            "Brier Score ↓": round(bs, 4),
            "Log Loss ↓": round(ll, 4),
            "AUC ↑": round(auc, 4),
            "EVG ↑": round(evg, 4),
            "Shot Decision Acc ↑": round(sda, 4),
        })

    return results


def print_table(results):
    headers = list(results[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in results)) for h in headers}

    header_row = "  ".join(h.ljust(col_widths[h]) for h in headers)
    print("\n" + "=" * len(header_row))
    print("MODEL COMPARISON TABLE")
    print("=" * len(header_row))
    print(header_row)
    print("-" * len(header_row))

    for r in results:
        row = "  ".join(str(r[h]).ljust(col_widths[h]) for h in headers)
        print(row)

    print("=" * len(header_row))
    print("\nBrier Score & Log Loss = lower is better | AUC, EVG, Shot Decision Acc = higher is better")
    print("EVG = avg win prob improvement if model's optimal shot was played instead")


def rank_models(results):
    def rank_list(vals, higher_is_better):
        indexed = sorted(enumerate(vals), key=lambda x: x[1], reverse=higher_is_better)
        ranks = [0] * len(vals)
        for rank, (idx, _) in enumerate(indexed):
            ranks[idx] = rank + 1
        return ranks

    metrics = [
        ("Brier Score ↓", False),
        ("Log Loss ↓", False),
        ("AUC ↑", True),
        ("EVG ↑", True),
        ("Shot Decision Acc ↑", True),
    ]

    total_ranks = [0] * len(results)
    for metric, higher_is_better in metrics:
        vals = [r[metric] for r in results]
        ranks = rank_list(vals, higher_is_better)
        for i, rank in enumerate(ranks):
            total_ranks[i] += rank

    ranked = sorted(zip(total_ranks, results), key=lambda x: x[0])
    best = ranked[0][1]["Model"]
    ranking = [(r["Model"], total_rank) for total_rank, r in ranked]

    print(f"\n Best overall model: {best}")
    print("Full ranking (lower score = better):")
    for name, score in ranking:
        print(f"  {name}: {score}")

    return best, ranking