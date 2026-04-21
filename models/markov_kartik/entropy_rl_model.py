import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from base_model import WinProbabilityModel


class PolicyNet(nn.Module):
    """
    Policy network: takes current rally state and outputs
    a distribution over next shots.
    """
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
    """Estimates win probability from state."""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1)


def encode_state(shot_sequence, action_dim, max_history=5):
    """
    Encodes current state as a fixed-size vector.
    One-hot of last few shots + normalized rally length.
    """
    state = np.zeros(action_dim * max_history + 1)
    history = shot_sequence[-max_history:]
    for i, shot in enumerate(history):
        if 0 <= shot < action_dim:
            state[i * action_dim + shot] = 1.0
    state[-1] = len(shot_sequence) / 30.0
    return state


class EntropyRegRL(WinProbabilityModel):
    """
    Entropy-Regularized Reinforcement Learning model.

    Models both player and opponent as stochastic agents.
    Reward at each step = change in win probability + alpha * entropy(policy)

    The entropy bonus prevents the policy from collapsing to always
    playing one shot — naturally models how real players mix strategies.

    alpha controls how stochastic the learned policy is:
        high alpha = more mixing, low alpha = more decisive
    """

    def __init__(self, action_dim=9, alpha=0.1, lr=0.001, episodes=500,
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

    def fit(self, rallies, outcomes):
        print(f"  Training Entropy-Reg RL with alpha={self.alpha}...")

        for episode in range(self.episodes):
            idx = np.random.randint(len(rallies))
            rally = rallies[idx]
            outcome = outcomes[idx]

            if len(rally) == 0:
                continue

            rally = [min(s, self.action_dim - 1) for s in rally]

            states = []
            log_probs = []
            entropies = []
            values = []
            rewards = []

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

                curr_prob = v
                delta = curr_prob - prev_prob
                reward = delta + self.alpha * entropy.item()
                rewards.append(reward)
                prev_prob = curr_prob

                # track opponent shots (even positions = opponent)
                if i % 2 == 1:
                    context = tuple(rally[max(0, i-2):i])
                    self.opponent_shot_counts[context][rally[i]] += 1

            rewards[-1] += float(outcome) - prev_prob

            # discounted returns
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)

            if returns.std() > 1e-6:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # update value net
            self.value_opt.zero_grad()
            state_tensors = torch.stack(states)
            predicted_values = self.value_net(state_tensors)
            target_values = torch.tensor(
                [float(outcome)] * len(states), dtype=torch.float32
            )
            value_loss = F.mse_loss(predicted_values, target_values)
            value_loss.backward()
            self.value_opt.step()

            # update policy with entropy-regularized REINFORCE
            self.policy_opt.zero_grad()
            policy_loss = 0
            for log_p, entropy, G in zip(log_probs, entropies, returns):
                policy_loss += -(log_p * G.detach() + self.alpha * entropy)
            policy_loss.backward()
            self.policy_opt.step()

            if (episode + 1) % 100 == 0:
                print(f"  RL episode {episode+1}/{self.episodes}")

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
        """
        Returns learned stochastic distribution over opponent shots.
        This is the key output — opponent is modeled as mixing strategies,
        not always playing the same shot.
        """
        context = tuple(context_shots[-2:])
        counts = self.opponent_shot_counts[context]
        total = counts.sum()
        if total == 0:
            return np.ones(self.action_dim) / self.action_dim
        return counts / total

    def get_optimal_action(self, shot_sequence):
        """
        Returns recommended next shot distribution.
        High entropy = multiple shots are good (mixed strategy).
        Low entropy = model is confident in one shot.
        """
        state = self._get_state_tensor(shot_sequence)
        with torch.no_grad():
            probs = self.policy(state).numpy()
        return probs