import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data pipeline (same as bidirectional model)
# ---------------------------------------------------------------------------

class RallyEncoder:
    """
    x: 0 = Forehand, 1 = Backhand
    y: 0 = Down the line, 1 = Crosscourt, 2 = Middle/Body, 3 = Other/Unknown
    Combined encoding: hand * 4 + angle  →  values 0-7
    """

    SHOT_MAP = {
        'f': (0, 0), 'r': (0, 1), 'v': (0, 2),
        'b': (1, 0), 's': (1, 1), 'z': (1, 2),
        'o': (0, 3), 'u': (0, 3), 'l': (0, 3), 'h': (0, 3), 'j': (0, 3),
        'p': (1, 3), 'y': (1, 3), 'm': (1, 3), 'i': (1, 3), 'k': (1, 3),
    }

    @staticmethod
    def process(rally_text):
        if not rally_text or rally_text in ['S', 'R']:
            return None
        shot_sequence = []
        for char in rally_text.lower():
            if char in RallyEncoder.SHOT_MAP:
                hand, angle = RallyEncoder.SHOT_MAP[char]
                shot_sequence.append(hand * 4 + angle)
        return shot_sequence if shot_sequence else None


class RallyDataBuilder:
    def __init__(self, dataframe):
        self.raw_data = dataframe
        self.samples = self._build()

    def _build(self):
        valid_samples = []
        for _, game_point in self.raw_data.iterrows():
            rally_code = (
                game_point['2nd']
                if pd.notna(game_point.get('2nd'))
                else game_point.get('1st')
            )
            sequence = RallyEncoder.process(rally_code)
            if sequence:
                server_won = 1.0 if game_point['PtWinner'] == game_point['Svr'] else 0.0
                valid_samples.append({'shots': sequence, 'outcome': server_won})
        return valid_samples


# ---------------------------------------------------------------------------
# Networks (unchanged from original)
# ---------------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


def encode_state(shot_sequence, action_dim, max_history=5):
    state = np.zeros(action_dim * max_history + 1)
    history = shot_sequence[-max_history:]
    for i, shot in enumerate(history):
        if 0 <= shot < action_dim:
            state[i * action_dim + shot] = 1.0
    state[-1] = len(shot_sequence) / 30.0
    return state


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class EntropyRegRL(WinProbabilityModel):
    """
    Entropy-Regularized Reinforcement Learning model.
    Same as original but now uses RallyEncoder / RallyDataBuilder
    for data loading (consistent with the bidirectional model).
    action_dim is 8 to match the hand*4+angle encoding.
    """

    def __init__(self, action_dim=8, alpha=0.1, lr=0.001, episodes=500,
                 gamma=0.95, max_history=5):
        self.action_dim = action_dim
        self.alpha = alpha
        self.lr = lr
        self.episodes = episodes
        self.gamma = gamma
        self.max_history = max_history

        state_dim = action_dim * max_history + 1

        self.policy = PolicyNet(state_dim, action_dim)
        self.value_net = ValueNet(state_dim)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value_net.parameters(), lr=lr)

        self.trained = False
        self.opponent_shot_counts = defaultdict(lambda: np.zeros(action_dim))

    def get_name(self):
        return "Entropy-Reg RL"

    def _get_state_tensor(self, shot_sequence):
        state = encode_state(shot_sequence, self.action_dim, self.max_history)
        return torch.tensor(state, dtype=torch.float32)

    def fit(self, data_path):
        """Load data via the shared pipeline, then run RL training."""
        print(f"  Loading data from {data_path} ...")
        df = pd.read_csv(data_path)
        builder = RallyDataBuilder(df)
        rallies = [s['shots'] for s in builder.samples]
        outcomes = [s['outcome'] for s in builder.samples]
        print(f"  {len(rallies)} valid rallies loaded.")

        print(f"  Training Entropy-Reg RL with alpha={self.alpha}...")

        for episode in range(self.episodes):
            idx = np.random.randint(len(rallies))
            rally = [min(s, self.action_dim - 1) for s in rallies[idx]]
            outcome = outcomes[idx]

            if not rally:
                continue

            states, log_probs, entropies, values, rewards = [], [], [], [], []
            prev_prob = 0.5

            for i in range(len(rally)):
                state_tensor = self._get_state_tensor(rally[:i])
                states.append(state_tensor)

                with torch.no_grad():
                    v = self.value_net(state_tensor).item()
                values.append(v)

                action_probs = self.policy(state_tensor)
                dist = torch.distributions.Categorical(action_probs)

                action = torch.tensor(rally[i])
                log_p = dist.log_prob(action)
                entropy = dist.entropy()

                log_probs.append(log_p)
                entropies.append(entropy)

                reward = (v - prev_prob) + self.alpha * entropy.item()
                rewards.append(reward)
                prev_prob = v

                if i % 2 == 1:
                    context = tuple(rally[max(0, i - 2):i])
                    self.opponent_shot_counts[context][rally[i]] += 1

            rewards[-1] += float(outcome) - prev_prob

            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            if returns.std() > 1e-6:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            self.value_opt.zero_grad()
            state_tensors = torch.stack(states)
            predicted_values = self.value_net(state_tensors)
            target_values = torch.tensor([float(outcome)] * len(states), dtype=torch.float32)
            value_loss = F.mse_loss(predicted_values, target_values)
            value_loss.backward()
            self.value_opt.step()

            self.policy_opt.zero_grad()
            policy_loss = sum(
                -(log_p * G.detach() + self.alpha * ent)
                for log_p, ent, G in zip(log_probs, entropies, returns)
            )
            policy_loss.backward()
            self.policy_opt.step()

            if (episode + 1) % 100 == 0:
                print(f"  RL episode {episode + 1}/{self.episodes}")

        self.trained = True

    def predict_proba(self, shot_sequence):
        if not self.trained:
            return [0.5] * len(shot_sequence)
        probs = []
        for i in range(len(shot_sequence)):
            state = self._get_state_tensor(shot_sequence[:i])
            with torch.no_grad():
                p = self.value_net(state).item()
            probs.append(p)
        return probs

    def get_opponent_distribution(self, context_shots):
        context = tuple(context_shots[-2:])
        counts = self.opponent_shot_counts[context]
        total = counts.sum()
        return counts / total if total > 0 else np.ones(self.action_dim) / self.action_dim

    def get_optimal_action(self, shot_sequence):
        state = self._get_state_tensor(shot_sequence)
        with torch.no_grad():
            probs = self.policy(state).numpy()
        return probs


if __name__ == "__main__":
    model = EntropyRegRL(episodes=500, alpha=0.1)
    model.fit("data/raw/charting-m-points-2020s.csv")