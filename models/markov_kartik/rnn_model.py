import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

def parse_rally(rally_str):
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
            "shot": (
                "FH" if ch in fh_letters else
                "BH" if ch in bh_letters else
                "OTHER"
            ),
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

def simplify_rally(parsed_rally):
    simplified = []
    for shot_info in parsed_rally:
        shot_type = shot_info.get('shot', '').upper()
        dir = shot_info.get('dir', 0)
        simplified.append((shot_type, dir))
    return simplified

def encode_shot(shot):
    shot_type, direction = shot
    if direction is None:
        direction = 0
    if shot_type == 'FH':
        return direction
    elif shot_type == 'BH':
        return 4 + direction
    else:
        return 8

class TennisDataset(Dataset):
    def __init__(self, df):
        self.sequences = []
        self.labels = []
        
        for _, row in df.iterrows():
            rally_str = row['2nd'] if pd.notna(row['2nd']) and row['2nd'] != '' else row['1st']
            if not rally_str or rally_str in ['S','R']:
                continue
            
            parsed = parse_rally(rally_str)
            simplified = simplify_rally(parsed)
            
            if len(simplified) > 0:
                encoded = [encode_shot(s) for s in simplified]
                self.sequences.append(encoded)
                
                winner = 1 if row['PtWinner'] == row['Svr'] else 0
                self.labels.append(winner)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def pad_batch(batch):
    sequences, labels = zip(*batch)
    
    max_len = max(len(s) for s in sequences)
    
    padded = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded.append(seq + [0] * pad_len)
    
    return torch.tensor(padded, dtype=torch.long), torch.tensor(labels, dtype=torch.float)

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embed = nn.Embedding(10, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.embed(x)
        output, _ = self.lstm(x)  # ← use all hidden states, not just last
        out = self.fc(output[:, -1, :])      # ← predict at every timestep
        return torch.sigmoid(out)

df = pd.read_csv("data/processed/points_hard_2022_2024.csv")

dataset = TennisDataset(df)
print(f"rallies: {len(dataset)}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=pad_batch)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=pad_batch)

model = RNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

for epoch in range(10):
    model.train()
    total_loss = 0
    
    for seq, labels in train_loader:
        optimizer.zero_grad()
        
        pred = model(seq).squeeze()
        loss = loss_fn(pred, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"epoch {epoch+1}, loss: {total_loss/len(train_loader):.4f}")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for seq, labels in val_loader:
        pred = model(seq).squeeze()
        pred_class = (pred > 0.5).float()
        correct += (pred_class == labels).sum().item()
        total += labels.size(0)

print(f"accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), 'tennis_rnn.pth')