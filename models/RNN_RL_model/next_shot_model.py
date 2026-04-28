import argparse
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def parse_rally(rally_str: str):
    if not rally_str or not isinstance(rally_str, str):
        return []

    serve_digits = {'0', '4', '5', '6'}
    serve_letters = {'n', 'w', 'd', 'x', 'g', 'e', '!', 'c', 'V', 'P', 'Q', 'S', 'R'}
    terminal_symbols = {'*', '@', '#', 'C'}
    fh_letters = {'f', 'r', 'v', 'o', 'u', 'l', 'h', 'j'}
    bh_letters = {'b', 's', 'z', 'p', 'y', 'm', 'i', 'k'}
    other_letters = {'t', 'q'}
    shot_letters = fh_letters | bh_letters | other_letters
    error_types = {'n', 'w', 'd', 'x', '!', 'e'}

    shots = []
    i = 0
    n = len(rally_str)

    while i < n:
        if rally_str[i] in serve_digits or rally_str[i] in serve_letters or rally_str[i] == '+':
            i += 1
        else:
            break

    while i < n:
        ch = rally_str[i]
        if ch in terminal_symbols:
            break
        if ch not in shot_letters:
            i += 1
            continue

        shot = {
            "shot": ("FH" if ch in fh_letters else "BH" if ch in bh_letters else "OTHER"),
            "dir": None,
        }
        i += 1

        while i < n:
            c = rally_str[i]
            if c in {'+', ';', '^', '-', '='}:
                i += 1
            else:
                break

        if i < n and rally_str[i] in {'0', '1', '2', '3'}:
            shot["dir"] = int(rally_str[i])
            i += 1

        if i < n and rally_str[i] in {'7', '8', '9', '0'}:
            i += 1

        if i < n and rally_str[i] in error_types:
            i += 1

        if i < n and rally_str[i] in {'*', '@', '#'}:
            i += 1
            shots.append(shot)
            break

        shots.append(shot)

    return shots


def simplify_rally(parsed_rally: List[dict]) -> List[Tuple[str, int]]:
    simplified = []
    for shot_info in parsed_rally:
        shot_type = shot_info.get('shot', '').upper()
        dir = shot_info.get('dir', 0)
        simplified.append((shot_type, dir))
    return simplified


def encode_shot(shot: Tuple[str, int]) -> int:
    shot_type, direction = shot
    if direction is None:
        direction = 0

    if shot_type == 'FH':
        return direction
    elif shot_type == 'BH':
        return 4 + direction
    else:
        return 8


def decode_shot(shot_code: int) -> str:
    if shot_code < 4:
        return f"FH→{shot_code}"
    elif shot_code < 8:
        return f"BH→{shot_code - 4}"
    else:
        return "Other"


class NextShotDataset(Dataset):
    """Creates (state, next_shot) supervised examples from rally strings.

    Each state is the last `window` encoded shots (padded with 0).
    The label is the integer code of the next shot.
    """

    def __init__(self, csv_path: str, rally_col_candidates=None, window: int = 5, max_examples: int = None):
        df = pd.read_csv(csv_path)

        if rally_col_candidates is None:
            rally_col_candidates = ['rally', 'rally_str', 'rally_string', 'sequence']

        rally_col = None
        for c in rally_col_candidates:
            if c in df.columns:
                rally_col = c
                break

        if rally_col is None:
            # pick first string column
            for c in df.columns:
                if df[c].dtype == object:
                    rally_col = c
                    break

        if rally_col is None:
            raise ValueError('No rally column detected in CSV')

        examples = []
        for idx, r in df[rally_col].dropna().items():
            parsed = parse_rally(str(r))
            simplified = simplify_rally(parsed)
            encoded = [encode_shot(s) for s in simplified]

            # sliding windows: for i in [0 .. len-2], state = last `window` before i+1, target = encoded[i+1]
            for i in range(len(encoded) - 1):
                state = encoded[max(0, i - window + 1): i + 1]
                target = encoded[i + 1]
                # pad to left
                pad_len = window - len(state)
                state_padded = [0] * pad_len + state
                examples.append((state_padded, target))

                if max_examples and len(examples) >= max_examples:
                    break
            if max_examples and len(examples) >= max_examples:
                break

        self.examples = examples
        self.window = window

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        state, target = self.examples[idx]
        return torch.LongTensor(state), torch.tensor(target, dtype=torch.long)


class NextShotModel(nn.Module):
    def __init__(self, num_tokens=10, d_emb=16, d_hidden=32, window=5, num_actions=9):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, d_emb)
        self.lstm = nn.LSTM(d_emb, d_hidden, batch_first=True)
        self.fc = nn.Linear(d_hidden, num_actions)

    def forward(self, x):
        # x: (batch, seq_len) long
        x = self.embed(x)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits


def train(model, dataloader, optimizer, device, epoch, log_every=100):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for i, (states, targets) in enumerate(dataloader):
        states = states.to(device)
        targets = targets.to(device)

        logits = model(states)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_every == 0:
            print(f"Epoch {epoch} | Step {i+1}/{len(dataloader)} | loss={total_loss/(i+1):.4f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    n = 0
    top1 = 0
    top3 = 0
    with torch.no_grad():
        for states, targets in dataloader:
            states = states.to(device)
            targets = targets.to(device)
            logits = model(states)
            total_loss += criterion(logits, targets).item()
            n += targets.size(0)

            probs = F.softmax(logits, dim=-1)
            topk = probs.topk(3, dim=-1).indices
            top1 += (topk[:, 0] == targets).sum().item()
            top3 += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()

    return {
        'cross_entropy': total_loss / n,
        'top1': top1 / n,
        'top3': top3 / n,
        'n': n,
    }


def collate_fn(batch):
    states = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    return states, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_out', default='models/rnn_model/next_shot.pth')
    parser.add_argument('--max_examples', type=int, default=200000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset = NextShotDataset(args.data, window=args.window, max_examples=args.max_examples)
    # split
    n = len(dataset)
    idx = list(range(n))
    random.shuffle(idx)
    split = int(0.8 * n)
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = NextShotModel(window=args.window).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, device, epoch)
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={metrics['cross_entropy']:.4f} | top1={metrics['top1']:.3f} | top3={metrics['top3']:.3f}")

    torch.save(model.state_dict(), args.model_out)
    print('Saved model to', args.model_out)


if __name__ == '__main__':
    main()
