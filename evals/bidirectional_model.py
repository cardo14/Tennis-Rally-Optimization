"""
Enhanced Tennis Rally Prediction Model
- Bidirectional LSTM
- Rally length as feature
- Shot direction parsing
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RallyEncoder:
    
    """
    x: 0 = Forehand, 1 = Backhand
    y: 0 = Down the line, 1 = Crosscourt, 2 = Middle/Body, 3 = Other/Unknown direction
    """

    SHOT_MAP = {
        'f': (0, 0), 'r': (0, 1), 'v': (0, 2),  # Forehand variants
        'b': (1, 0), 's': (1, 1), 'z': (1, 2),  # Backhand variants
        'o': (0, 3), 'u': (0, 3), 'l': (0, 3), 'h': (0, 3), 'j': (0, 3),
        'p': (1, 3), 'y': (1, 3), 'm': (1, 3), 'i': (1, 3), 'k': (1, 3)
    }
    
    @staticmethod
    def process(rally_text):
        if not rally_text or rally_text in ['S', 'R']:
            return None
        
        shot_sequence = []
        for char in rally_text.lower():
            if char in RallyEncoder.SHOT_MAP:
                hand, angle = RallyEncoder.SHOT_MAP[char]
                combined_encoding = hand * 4 + angle
                shot_sequence.append(combined_encoding)
        
        return shot_sequence if shot_sequence else None


class RallyDataBuilder:
    
    def __init__(self, dataframe):
        self.raw_data = dataframe
        self.encoder = RallyEncoder()
        self.samples = self._build()
    
    def _build(self):
        valid_samples = []
        
        for idx, game_point in self.raw_data.iterrows():
            rally_code = game_point['2nd'] if pd.notna(game_point.get('2nd')) else game_point.get('1st')
            sequence = self.encoder.process(rally_code)
            
            if sequence:
                server_won = 1.0 if game_point['PtWinner'] == game_point['Svr'] else 0.0
                valid_samples.append({
                    'shots': sequence,
                    'length': len(sequence),
                    'outcome': server_won
                })
        
        return valid_samples
    
    def get_pytorch_dataset(self):
        return RallyTensorDataset(self.samples)


class RallyTensorDataset(Dataset):
    def __init__(self, sample_list):
        self.data = sample_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        sample = self.data[i]
        return sample['shots'], sample['length'], sample['outcome']


def create_batches(samples):

    shot_seqs, seq_lens, outcomes = zip(*samples)
    
    longest = max(len(s) for s in shot_seqs)
    padded = [s + [0] * (longest - len(s)) for s in shot_seqs]
    
    return (
        torch.LongTensor(padded),
        torch.FloatTensor(seq_lens),
        torch.FloatTensor(outcomes)
    )


class RallyNet(nn.Module):
    
    def __init__(self, vocab_size=8, embedding_size=16, context_size=32):
        super().__init__()
        
        self.shot_embedding = nn.Embedding(vocab_size, embedding_size)
        self.bidirectional_processor = nn.LSTM(
            embedding_size, 
            context_size, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.metadata_layer = nn.Linear(1, 8)
        self.fusion_layer = nn.Linear(context_size * 2 + 8, 32)
        self.predictor = nn.Linear(32, 1)
        self.activation = nn.ReLU()
    
    def forward(self, shot_codes, metadata):
        embedded = self.shot_embedding(shot_codes)
        
        lstm_out, _ = self.bidirectional_processor(embedded)
        final_context = lstm_out[:, -1, :]
        
        meta_features = self.activation(self.metadata_layer(metadata.unsqueeze(1)))
        
        merged = torch.cat([final_context, meta_features], dim=1)
        hidden = self.activation(self.fusion_layer(merged))
        
        return torch.sigmoid(self.predictor(hidden)).squeeze()


def train_model(data_path, epochs=5, batch_sz=32, learn_rate=0.001):
    """Main training pipeline"""
    
    # Data preparation
    match_data = pd.read_csv(data_path)
    builder = RallyDataBuilder(match_data)
    full_dataset = builder.get_pytorch_dataset()
    
    print(f"Loaded {len(full_dataset)} valid rallies")
    
    # Split
    split_point = int(0.8 * len(full_dataset))
    train_set, test_set = torch.utils.data.random_split(
        full_dataset, 
        [split_point, len(full_dataset) - split_point]
    )
    
    train_batches = DataLoader(train_set, batch_size=batch_sz, shuffle=True, collate_fn=create_batches)
    test_batches = DataLoader(test_set, batch_size=batch_sz, collate_fn=create_batches)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = BidirectionalRallyNet().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=learn_rate)
    criterion = nn.BCELoss()
    
    # Training loop
    for ep in range(epochs):
        net.train()
        epoch_loss = []
        
        for shots, lengths, labels in train_batches:
            shots, lengths, labels = shots.to(device), lengths.to(device), labels.to(device)
            
            optim.zero_grad()
            predictions = net(shots, lengths)
            loss = criterion(predictions, labels)
            loss.backward()
            optim.step()
            
            epoch_loss.append(loss.item())
        
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {ep+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    # Evaluation
    net.eval()
    hits, attempts = 0, 0
    
    with torch.no_grad():
        for shots, lengths, labels in test_batches:
            shots, lengths, labels = shots.to(device), lengths.to(device), labels.to(device)
            predictions = net(shots, lengths)
            decisions = (predictions > 0.5).float()
            hits += (decisions == labels).sum().item()
            attempts += labels.size(0)
    
    accuracy = 100 * hits / attempts
    print(f"\nTest Set Accuracy: {accuracy:.2f}%")
    
    return net


if __name__ == "__main__":
    model = train_model("data/raw/charting-m-points-2020s.csv", epochs=5)