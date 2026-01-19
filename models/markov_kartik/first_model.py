import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Parse rally string into shots
def parse_rally(rally_str):
    # check if rally string is valid
    if not rally_str or not isinstance(rally_str, str):
        return []

    # define what characters mean in the rally notation
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

    # skip the serve part at the beginning
    while i < n:
        if rally_str[i] in serve_digits or rally_str[i] in serve_letters or rally_str[i] == '+':
            i += 1
        else:
            break

    # go through each character and extract shots
    while i < n:
        ch = rally_str[i]

        # if we hit an ending symbol, stop
        if ch in terminal_symbols:
            break

        # if not a shot letter, skip it
        if ch not in shot_letters:
            i += 1
            continue

        # figure out if forehand or backhand
        shot = {
            "shot": (
                "FH" if ch in fh_letters else
                "BH" if ch in bh_letters else
                "OTHER"
            ),
            "dir": None,
        }

        i += 1

        # skip modifier symbols like +, ;, etc
        while i < n:
            c = rally_str[i]
            if c in {'+', ';', '^', '-', '='}:
                i += 1
            else:
                break

        # get direction if it exists
        if i < n and rally_str[i] in {'0', '1', '2', '3'}:
            shot["dir"] = int(rally_str[i])
            i += 1

        # skip depth number if it exists
        if i < n and rally_str[i] in {'7', '8', '9', '0'}:
            i += 1

        # skip error type if it exists
        if i < n and rally_str[i] in error_types:
            i += 1

        # check if point ended on this shot
        if i < n and rally_str[i] in {'*', '@', '#'}:
            i += 1
            shots.append(shot)
            break

        shots.append(shot)

    return shots

# convert shot dict to simple tuple
def simplify_rally(parsed_rally):
    simplified = []
    for shot_info in parsed_rally:
        shot_type = shot_info.get('shot', '').upper()
        dir = shot_info.get('dir', 0)
        simplified.append((shot_type, dir))
    return simplified

# convert shot tuple to a number
def encode_shot(shot):
    shot_type, direction = shot
    
    # if no direction, default to 0
    if direction is None:
        direction = 0
    
    # FH gets 0-3, BH gets 4-7, OTHER gets 8
    if shot_type == 'FH':
        return direction
    elif shot_type == 'BH':
        return 4 + direction
    else:
        return 8

# dataset class to organize our data
class TennisDataset(Dataset):
    def __init__(self, df):
        self.sequences = []  # store rally sequences
        self.labels = []     # store who won
        
        # go through each row in dataframe
        for _, row in df.iterrows():
            # get rally string
            rally_str = row['2nd'] if pd.notna(row['2nd']) and row['2nd'] != '' else row['1st']
            if not rally_str or rally_str in ['S','R']:
                continue
            
            # parse and encode the rally
            parsed = parse_rally(rally_str)
            simplified = simplify_rally(parsed)
            
            if len(simplified) > 0:
                # convert each shot to a number
                encoded = []
                for s in simplified:
                    encoded.append(encode_shot(s))
                
                self.sequences.append(encoded)
                
                # did server win? 1 = yes, 0 = no
                winner = 1 if row['PtWinner'] == row['Svr'] else 0
                
                # create label for EACH shot
                # if server won, all shots get label 1
                # if returner won, all shots get label 0
                labels_per_shot = []
                for _ in range(len(encoded)):
                    labels_per_shot.append(winner)
                
                self.labels.append(labels_per_shot)
    
    # how many rallies do we have
    def __len__(self):
        return len(self.sequences)
    
    # get rally number idx
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# pad sequences to same length for batching
def pad_batch(batch):
    sequences = []
    labels = []
    
    # unpack batch
    for seq, label in batch:
        sequences.append(seq)
        labels.append(label)
    
    # find longest rally
    max_len = 0
    for s in sequences:
        if len(s) > max_len:
            max_len = len(s)
    
    # pad all sequences to max length
    padded_seq = []
    padded_labels = []
    
    for seq, label in zip(sequences, labels):
        pad_len = max_len - len(seq)
        
        # add zeros to the end
        padded_seq.append(seq + [0] * pad_len)
        padded_labels.append(label + [0] * pad_len)
    
    # convert to tensors
    seq_tensor = torch.tensor(padded_seq, dtype=torch.long)
    label_tensor = torch.tensor(padded_labels, dtype=torch.float)
    
    return seq_tensor, label_tensor

# the neural network model
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # embedding: convert shot numbers to vectors
        # 10 = vocab size (shots 0-9)
        # 16 = embedding dimension
        self.embed = nn.Embedding(10, 16)
        
        # LSTM: the memory part
        # 16 = input size (from embedding)
        # 32 = hidden size (memory capacity)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        
        # fully connected layer: makes final prediction
        # 32 = input (from LSTM)
        # 1 = output (probability)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        # step 1: convert shot numbers to embeddings
        x = self.embed(x)
        
        # step 2: pass through LSTM
        # output = hidden state at EVERY shot
        output, _ = self.lstm(x)
        
        # step 3: predict probability at every shot
        out = self.fc(output)
        
        # step 4: squeeze removes extra dimension
        # sigmoid converts to probability 0-1
        return torch.sigmoid(out.squeeze(-1))

# load data
print("Loading data...")
df = pd.read_csv("data/processed/points_hard_2022_2024.csv")

# create dataset
dataset = TennisDataset(df)
print(f"Total rallies: {len(dataset)}")

# split into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

# create data loaders
# batch_size = process 32 rallies at once
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=pad_batch)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=pad_batch)

# create model
model = RNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# training loop
print("\nTraining...")
for epoch in range(10):
    model.train()
    total_loss = 0
    
    # go through each batch
    for seq, labels in train_loader:
        # reset gradients
        optimizer.zero_grad()
        
        # make predictions
        pred = model(seq)
        
        # calculate loss only on real shots (not padding)
        mask = (seq != 0).float()
        loss = (loss_fn(pred, labels) * mask).sum() / mask.sum()
        
        # backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.4f}")

# save model
torch.save(model.state_dict(), 'tennis_rnn.pth')
print("\nModel saved to tennis_rnn.pth")

# test on example rallies
model.eval()
print("\n" + "="*60)
print("EXAMPLE RALLY ANALYSIS")
print("="*60)

# get 5 test rallies
test_rallies = []
for i in range(5):
    if i < len(val_data):
        seq, label = val_data[i]
        test_rallies.append((seq, label))

with torch.no_grad():
    for rally_num in range(len(test_rallies)):
        seq, label = test_rallies[rally_num]
        
        # prepare input
        seq_tensor = torch.tensor([seq], dtype=torch.long)
        
        # get predictions
        predictions = model(seq_tensor).squeeze().numpy()
        
        # find actual rally length (ignore padding)
        rally_len = 0
        for s in seq:
            if s != 0:
                rally_len += 1
        
        # print rally info
        print(f"\nRally {rally_num + 1}")
        print("-" * 40)
        print(f"Shots: {seq[:rally_len]}")
        
        outcome = 'Server won' if label[0] == 1 else 'Returner won'
        print(f"Actual outcome: {outcome}")
        print("\nWin probability after each shot:")
        
        # track best shot
        best_shot_num = 1
        best_advantage = -999
        
        # go through each shot
        for shot_idx in range(rally_len):
            prob = predictions[shot_idx]
            
            # calculate advantage (change in win probability)
            if shot_idx == 0:
                # first shot compared to 50%
                advantage = prob - 0.5
            else:
                # compare to previous shot
                advantage = prob - predictions[shot_idx - 1]
            
            # decode shot
            shot_code = seq[shot_idx]
            if shot_code < 4:
                shot_name = f"FH dir {shot_code}"
            elif shot_code < 8:
                shot_name = f"BH dir {shot_code - 4}"
            else:
                shot_name = "OTHER"
            
            # print shot info
            sign = '+' if advantage >= 0 else ''
            print(f"  Shot {shot_idx + 1} ({shot_name}): {prob:.3f} ({sign}{advantage:.3f})")
            
            # track best shot
            # if advantage > best_advantage:
            #     best_advantage = advantage
            #     best_shot_num = shot_idx + 1
            if advantage > best_advantage:
                best_advantage = advantage
                best_shot_num = shot_idx + 1
        
        # print best shot
        sign = '+' if best_advantage >= 0 else ''
        print(f"\n  → Most advantageous shot in this rally: Shot {best_shot_num} (advantage: {sign}{best_advantage:.3f})")

print("\n" + "="*60)
print("Done!")
