import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from base_model import WinProbabilityModel


class RallyDataset(Dataset):
    def __init__(self, rallies, outcomes, max_len=30):
        self.max_len = max_len
        self.X = []
        self.y = []

        for rally, outcome in zip(rallies, outcomes):
            padded = rally[:max_len] + [0] * max(0, max_len - len(rally))
            self.X.append(padded)
            self.y.append(float(outcome))

        self.X = torch.tensor(self.X, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embed(x)
        output, _ = self.lstm(x)
        out = self.fc(output[:, -1, :])
        return torch.sigmoid(out).squeeze(-1)

    def forward_sequence(self, x):
        """Returns win prob at every timestep, not just the last."""
        x = self.embed(x)
        output, _ = self.lstm(x)
        out = self.fc(output)
        return torch.sigmoid(out).squeeze(-1)


class RNNModel(WinProbabilityModel):
    def __init__(self, vocab_size=50, embed_dim=16, hidden_dim=32, max_len=30,
                 epochs=20, batch_size=32, lr=0.001):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.net = None
        self.trained = False

    def get_name(self):
        return "RNN (LSTM)"

    def fit(self, rallies, outcomes):
        self.net = RNNNet(self.vocab_size, self.embed_dim, self.hidden_dim)
        dataset = RallyDataset(rallies, outcomes, self.max_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCELoss()

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.net(X_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  RNN epoch {epoch+1}/{self.epochs} — loss: {total_loss/len(loader):.4f}")

        self.net.eval()
        self.trained = True

    def predict_proba(self, shot_sequence):
        if not self.trained:
            return [0.5] * len(shot_sequence)

        padded = shot_sequence[:self.max_len] + [0] * max(0, self.max_len - len(shot_sequence))
        x = torch.tensor([padded], dtype=torch.long)

        with torch.no_grad():
            probs = self.net.forward_sequence(x)[0].tolist()

        return probs[:len(shot_sequence)]

    def load_existing(self, path):
        """Load your existing tennis_rnn.pth weights."""
        self.net = RNNNet(self.vocab_size, self.embed_dim, self.hidden_dim)
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        self.trained = True
        print(f"Loaded existing RNN weights from {path}")