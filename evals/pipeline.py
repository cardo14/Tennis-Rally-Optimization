import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss

from entropy_rl_model import EntropyRegRL, RallyDataBuilder
from bidirectional_model import RallyNet, RallyTensorDataset, create_batches

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DATA_PATH = "data/raw/charting-m-points-2020s.csv"

# ── load & split ──────────────────────────────────────────────────────────────
import pandas as pd
samples = RallyDataBuilder(pd.read_csv(DATA_PATH)).samples
train_s, test_s = train_test_split(samples, test_size=0.2, random_state=42)

# ── train entropy RL ──────────────────────────────────────────────────────────
rl = EntropyRegRL(episodes=500)
rl.fit(DATA_PATH)   # trains internally

rl_preds = [rl.predict_proba(s['shots'])[-1] for s in test_s]
rl_true  = [s['outcome'] for s in test_s]

# ── train bidirectional LSTM ──────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net    = RallyNet().to(device)
optim  = torch.optim.Adam(net.parameters(), lr=0.001)
crit   = nn.BCELoss()

train_loader = DataLoader(RallyTensorDataset(train_s), batch_size=32,
                          shuffle=True, collate_fn=create_batches)

for ep in range(5):
    for shots, lengths, labels in train_loader:
        shots, lengths, labels = shots.to(device), lengths.to(device), labels.to(device)
        optim.zero_grad()
        crit(net(shots, lengths), labels).backward()
        optim.step()
    print(f"Epoch {ep+1}/5 done")

test_loader = DataLoader(RallyTensorDataset(test_s), batch_size=32,
                         collate_fn=create_batches)
net.eval()
bi_preds, bi_true = [], []
with torch.no_grad():
    for shots, lengths, labels in test_loader:
        bi_preds.extend(net(shots.to(device), lengths.to(device)).cpu().numpy())
        bi_true.extend(labels.numpy())

# ── metrics ───────────────────────────────────────────────────────────────────
def metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"  Accuracy : {accuracy_score(y_true, [p > 0.5 for p in y_pred]):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_true, y_pred):.4f}")
    print(f"  Log Loss : {log_loss(y_true, y_pred):.4f}")
    print(f"  Brier    : {brier_score_loss(y_true, y_pred):.4f}")

metrics("Entropy-Reg RL",    rl_true,  rl_preds)
metrics("Bidirectional LSTM", bi_true, bi_preds)