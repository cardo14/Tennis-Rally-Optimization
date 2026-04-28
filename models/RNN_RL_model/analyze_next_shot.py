import argparse
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from next_shot_model import NextShotDataset, NextShotModel, decode_shot, collate_fn


def analyze(model_path, data_csv, window=5, batch_size=256, out_prefix='analysis'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = NextShotDataset(data_csv, window=window)

    # use same split logic as training: 80/20
    n = len(ds)
    idx = list(range(n))
    split = int(0.8 * n)
    val_idx = idx[split:]
    val_ds = torch.utils.data.Subset(ds, val_idx)

    loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)

    model = NextShotModel(window=window)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device).eval()

    records = []
    y_true = []
    y_pred = []
    probs_by_true = {i: [] for i in range(9)}

    with torch.no_grad():
        for states, targets in loader:
            states = states.to(device)
            logits = model(states)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            targets = targets.numpy()

            top1 = probs.argmax(axis=1)
            for i in range(len(targets)):
                t = int(targets[i])
                p = probs[i].tolist()
                pred = int(top1[i])
                records.append({
                    'state': states[i].cpu().tolist(),
                    'true': t,
                    'pred': pred,
                    'probs': p,
                })
                y_true.append(t)
                y_pred.append(pred)
                probs_by_true[t].append(p)

    df = pd.DataFrame(records)
    df.to_csv(f'{out_prefix}_predictions.csv', index=False)

    # confusion matrix
    K = 9
    cm = np.zeros((K, K), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1

    cm_df = pd.DataFrame(cm, index=[decode_shot(i) for i in range(K)], columns=[decode_shot(i) for i in range(K)])
    cm_df.to_csv(f'{out_prefix}_confusion.csv')

    # Top-k rates already computed in training script; compute top1/top3 here
    top1 = np.mean([1 if r['true'] == r['pred'] else 0 for r in records])
    top3_count = 0
    for r in records:
        top3 = np.argsort(r['probs'])[-3:][::-1]
        if r['true'] in top3:
            top3_count += 1
    top3 = top3_count / len(records)

    summary = {
        'n_examples': len(records),
        'top1': float(top1),
        'top3': float(top3),
    }
    with open(f'{out_prefix}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Per-class distribution of predicted probabilities (avg)
    avg_probs = {i: np.mean(probs_by_true[i], axis=0).tolist() if len(probs_by_true[i]) > 0 else [0.0]*K for i in range(K)}
    pd.DataFrame(avg_probs).T.to_csv(f'{out_prefix}_avg_probs_by_true.csv', header=[f'p_{i}' for i in range(K)])

    # Simple heatmap: split FH/BH by direction
    # FH codes 0-3, BH codes 4-7, Other 8
    def heat_from_probs(avg_probs_dict):
        fh = [avg_probs_dict[i][i] if i < len(avg_probs_dict[i]) else 0.0 for i in range(4)]
        bh = [avg_probs_dict[4 + i][4 + i] if 4 + i < len(avg_probs_dict) else 0.0 for i in range(4)]
        other = avg_probs_dict[8][8] if 8 in avg_probs_dict else 0.0
        return fh, bh, other

    fh, bh, other = heat_from_probs(avg_probs)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].bar(range(4), fh)
    ax[0].set_title('Avg prob correct FH→dir')
    ax[1].bar(range(4), bh)
    ax[1].set_title('Avg prob correct BH→dir')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_heatmap.png', dpi=150)

    print('Done. Outputs:')
    print(f' - {out_prefix}_predictions.csv (row per example)')
    print(f' - {out_prefix}_confusion.csv')
    print(f' - {out_prefix}_summary.json')
    print(f' - {out_prefix}_avg_probs_by_true.csv')
    print(f' - {out_prefix}_heatmap.png')


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--out', default='analysis')
    args = parser.parse_args()
    analyze(args.model, args.data, window=args.window, batch_size=args.batch_size, out_prefix=args.out)


if __name__ == '__main__':
    cli()
