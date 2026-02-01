import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import visualizations as viz

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
        if predictions.ndim == 0:
            predictions = np.array([predictions])
        
        
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
            
            # decode shots
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
            if advantage > best_advantage:
                best_advantage = advantage
                best_shot_num = shot_idx + 1
        
        # print best shot
        sign = '+' if best_advantage >= 0 else ''
        print(f"\n  → Most advantageous shot in this rally: Shot {best_shot_num} (advantage: {sign}{best_advantage:.3f})")

print("\n" + "="*60)



# Breakeven analysis


print("\n" + "="*60)
print("BREAKEVEN POINT ANALYSIS")
print("="*60)
print("Finding shots where advantage turned to disadvantage...")
print()

def find_breakeven_and_alternatives(rally_seq, model):
    #Where advantageous shots become disadvantageous
    #This is all possible shots: FH 0-3, BH 4-7, OTHER 8
    all_shots = []
    for d in range(4):
        all_shots.append(('FH', d, d))
    for d in range(4):
        all_shots.append(('BH', d, 4+d))
    all_shots.append(('OTHER', None, 8))
    
    results = []
    
    with torch.no_grad():
        for shot_idx in range(len(rally_seq)):
            if rally_seq[shot_idx] == 0:  # padding
                break
            
            # Get state BEFORE this shot
            rally_before = rally_seq[:shot_idx]
            
            # Baseline probability (where we were before this shot)
            if len(rally_before) == 0:
                baseline_prob = 0.5
            else:
                seq_tensor = torch.tensor([rally_before], dtype=torch.long)
                preds = model(seq_tensor).squeeze().numpy()
                baseline_prob = preds[-1] if preds.ndim > 0 else preds.item()
            
            # Current shot analysis
            current_shot = rally_seq[shot_idx]
            current_seq = rally_before + [current_shot]
            
            seq_tensor = torch.tensor([current_seq], dtype=torch.long)
            preds = model(seq_tensor).squeeze().numpy()
            current_prob = preds[-1] if preds.ndim > 0 else preds.item()
            current_advantage = current_prob - baseline_prob
            
            # Test ALL alternative shots from this position
            alternative_results = []
            for shot_name, direction, encoding in all_shots:
                alt_seq = rally_before + [encoding]
                seq_tensor = torch.tensor([alt_seq], dtype=torch.long)
                preds = model(seq_tensor).squeeze().numpy()
                alt_prob = preds[-1] if preds.ndim > 0 else preds.item()
                alt_advantage = alt_prob - baseline_prob
                
                alternative_results.append({
                    'name': f"{shot_name} dir {direction}" if direction is not None else shot_name,
                    'encoding': encoding,
                    'prob': alt_prob,
                    'advantage': alt_advantage,
                    'is_current': encoding == current_shot
                })
            
            # Sort alternatives by advantage (best first)
            alternative_results.sort(key=lambda x: x['advantage'], reverse=True)
            
            # Decode current shot name
            if current_shot < 4:
                current_name = f"FH dir {current_shot}"
            elif current_shot < 8:
                current_name = f"BH dir {current_shot - 4}"
            else:
                current_name = "OTHER"
            
            # Find best alternative (that's not the current shot)
            best_alternative = None
            for alt in alternative_results:
                if not alt['is_current']:
                    best_alternative = alt
                    break
            
            # Here, we can detect breakeven
            is_breakeven = current_advantage < 0
            
            results.append({
                'shot_num': shot_idx + 1,
                'shot_name': current_name,
                'baseline_prob': baseline_prob,
                'current_prob': current_prob,
                'current_advantage': current_advantage,
                'is_breakeven': is_breakeven,
                'best_alternative': best_alternative,
                'all_alternatives': alternative_results
            })
    
    return results

# Analyze test rallies for breakeven points
for rally_num in range(len(test_rallies)):
    seq, label = test_rallies[rally_num]
    
    print(f"\n{'='*60}")
    print(f"RALLY {rally_num + 1} - BREAKEVEN ANALYSIS")
    print('='*60)
    
    # Find actual rally length
    rally_len = 0
    for s in seq:
        if s != 0:
            rally_len += 1
    
    outcome = 'Server won' if label[0] == 1 else 'Returner won'
    print(f"Outcome: {outcome}")
    print(f"Rally length: {rally_len} shots\n")
    
    # Get breakeven analysis
    breakeven_results = find_breakeven_and_alternatives(seq[:rally_len], model)
    
    # Track if we found any breakeven points
    found_breakeven = False
    breakeven_shots = []
    
    for result in breakeven_results:
        if result['is_breakeven']:
            found_breakeven = True
            breakeven_shots.append(result['shot_num'])
    
    if found_breakeven:
        print(f"BREAKEVEN POINTS DETECTED at shot(s): {breakeven_shots}")
        print(f"   (These shots DECREASED win probability)\n")
    else:
        print("✓ No breakeven points - all shots increased win probability\n")
    
    # Detailed shot-by-shot analysis
    for result in breakeven_results:
        shot_num = result['shot_num']
        shot_name = result['shot_name']
        current_adv = result['current_advantage']
        
        # Format advantage display
        sign = '+' if current_adv >= 0 else ''
        status = "✗ DISADVANTAGEOUS" if result['is_breakeven'] else "✓ Advantageous"
        
        print(f"Shot {shot_num}: {shot_name}")
        print(f"  Win prob: {result['baseline_prob']:.3f} → {result['current_prob']:.3f} ({sign}{current_adv:.3f})")
        print(f"  Status: {status}")
        
        # If breakeven, show the better alternative
        if result['is_breakeven']:
            best_alt = result['best_alternative']
            print(f"  💡 BETTER OPTION: {best_alt['name']}")
            print(f"     Would give: {result['baseline_prob']:.3f} → {best_alt['prob']:.3f} ({best_alt['advantage']:+.3f})")
            print(f"     Improvement over actual: {best_alt['advantage'] - current_adv:+.3f}")
            
            # Show top 3 alternatives
            print(f"  📊 Top 3 alternatives:")
            for i, alt in enumerate(result['all_alternatives'][:3]):
                if alt['is_current']:
                    print(f"     {i+1}. {alt['name']}: {alt['advantage']:+.3f} ← ACTUAL SHOT")
                else:
                    print(f"     {i+1}. {alt['name']}: {alt['advantage']:+.3f}")
        
        print()
    
    # Summary statistics
    print(f"\n{'─'*60}")
    print("RALLY SUMMARY:")
    print(f"  Total shots: {len(breakeven_results)}")
    print(f"  Advantageous shots: {sum(1 for r in breakeven_results if not r['is_breakeven'])}")
    print(f"  Disadvantageous shots (breakeven): {sum(1 for r in breakeven_results if r['is_breakeven'])}")
    
    if found_breakeven:
        total_lost = sum(r['current_advantage'] for r in breakeven_results if r['is_breakeven'])
        print(f"  Total probability lost at breakeven points: {total_lost:.3f}")
        
        # Calculate potential gain if best alternatives were played
        total_potential_gain = 0
        for r in breakeven_results:
            if r['is_breakeven']:
                potential_gain = r['best_alternative']['advantage'] - r['current_advantage']
                total_potential_gain += potential_gain
        print(f"  Potential gain if alternatives played: {total_potential_gain:+.3f}")

print("\n" + "="*60)


# Graphics from separate file
viz.generate_all_visualizations(dataset, test_rallies, model)