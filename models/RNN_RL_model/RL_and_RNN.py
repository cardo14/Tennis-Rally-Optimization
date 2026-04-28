import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

def parse_rally(rally_str):
    """Parse rally string into shots"""
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

def decode_shot(shot_code):
    """Convert shot number to readable text"""
    if shot_code < 4:
        return f"FH→{shot_code}"
    elif shot_code < 8:
        return f"BH→{shot_code - 4}"
    else:
        return "Other"

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 16)
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.embed(x)
        output, _ = self.lstm(x)
        out = self.fc(output)
        return torch.sigmoid(out.squeeze(-1))

#Q-learning agent  

class QNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)   # 5 shots → 64 neurons
        self.fc2 = nn.Linear(64, 32)  # 64 → 32
        self.fc3 = nn.Linear(32, 8)   # 32 → 8 actions, excluding 'other' sho
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action


class RLAgent:

    def __init__(self):
        self.q_network = QNetwork()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Hyperparameters (simple version)
        self.epsilon = 1.0      # Start by exploring a lot
        self.epsilon_min = 0.1  # Always explore at least 10%
        self.epsilon_decay = 0.995
        self.gamma = 0.95       # Discount factor
        
        # Memory
        self.memory = []
        self.max_memory = 5000
    
    def get_state_tensor(self, shot_history):

        # Get last 5 shots, pad if fewer
        recent = shot_history[-5:] if len(shot_history) > 0 else []
        padded = recent + [0] * (5 - len(recent))
        return torch.FloatTensor(padded)
    
    def choose_action(self, shot_history, training=True):  #Choose which shot to play 

        if training and random.random() < self.epsilon:
            return random.randint(0, 7)  # Random shot
        
        # Use Q-network to pick best shot
        state = self.get_state_tensor(shot_history).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def remember(self, shot_history, action, reward, next_shot_history, done):  #store experience in memory
        self.memory.append((shot_history.copy(), action, reward, 
                          next_shot_history.copy(), done))
        
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
    
    def learn(self, batch_size=32):  # Learn from past experiences

        if len(self.memory) < batch_size:
            return 0
        
        batch = random.sample(self.memory, batch_size)
        
        total_loss = 0
        for shot_history, action, reward, next_shot_history, done in batch:

            state = self.get_state_tensor(shot_history).unsqueeze(0) # convert to tensors 
            next_state = self.get_state_tensor(next_shot_history).unsqueeze(0)
            
            q_values = self.q_network(state)
            current_q = q_values[0, action]
            
            if done:
                target_q = reward  # if rally ended just use reward
            else:
                with torch.no_grad():
                    next_q_values = self.q_network(next_state)
                    max_next_q = next_q_values.max()
                target_q = reward + self.gamma * max_next_q
            
            loss = F.mse_loss(current_q, torch.tensor(target_q))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / batch_size


def train_simple_rl(rnn_model, num_episodes=1000):  # Training loop

    print("=" * 60)
    print("SIMPLE RL TRAINING")
    print("=" * 60)
    print(f"Training for {num_episodes} rallies\n")
    
    agent = RLAgent()
    rnn_model.eval()
    
    # Track progress
    rewards_history = []
    win_rates = []
    
    for episode in range(num_episodes):
        # Start new rally
        rally = []
        total_reward = 0
        current_player = 0  # 0=server, 1=returner
        
        # Play one rally 
        for step in range(20):
            # Agent chooses action
            action = agent.choose_action(rally, training=True)
            rally.append(action)
            
            # Use RNN to see what happens
            with torch.no_grad():
                seq = torch.tensor([rally], dtype=torch.long)
                predictions = rnn_model(seq).squeeze()
                
                if len(rally) == 1:
                    win_prob = predictions.item()
                else:
                    win_prob = predictions[-1].item()
            
            # Decide if rally ends
            end_prob = min(0.7, len(rally) * 0.1)
            if win_prob > 0.85 or win_prob < 0.15:
                end_prob += 0.3
            
            rally_ends = random.random() < end_prob or len(rally) >= 20
            
            if rally_ends:
                # Rally over - determine winner
                server_wins = random.random() < win_prob
                
                if current_player == 0:  # Server's turn
                    reward = 1.0 if server_wins else -1.0
                else:  # Returner's turn
                    reward = -1.0 if server_wins else 1.0
                
                total_reward += reward
                done = True
                
                # Remember this experience
                agent.remember(rally[:-1], action, reward, rally, done)
                break
            else:
                # Rally continues
                reward = 0.0
                done = False
                
                # Remember this experience
                agent.remember(rally[:-1], action, reward, rally, done)
                
                # Switch player
                current_player = 1 - current_player
        
        loss = agent.learn(batch_size=32)

        rewards_history.append(total_reward)
        
        recent = rewards_history[-100:]           # Calculate win rate (last 100 episodes)
        win_rate = sum(1 for r in recent if r > 0) / len(recent)
        win_rates.append(win_rate)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards_history[-100:]) / 100
            print(f"Episode {episode+1:4d} | "
                  f"Win Rate: {win_rate:.1%} | "
                  f"Avg Reward: {avg_reward:+.2f} | "
                  f"Epsilon: {agent.epsilon:.2f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Final Win Rate: {win_rates[-1]:.1%}\n")
    
    return agent, rewards_history, win_rates


def get_recommendations(agent, current_rally, top_k=3):

    print("\n" + "=" * 60)
    print("SHOT RECOMMENDATIONS")
    print("=" * 60)
    print(f"Current rally: {[decode_shot(s) for s in current_rally[-5:]]}")
    print(f"Rally length: {len(current_rally)}\n")
    
    # Get Q-values for all actions
    state = agent.get_state_tensor(current_rally).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.q_network(state).squeeze()
    
    # Get top actions
    top_values, top_actions = q_values.topk(top_k)
    
    print("Best shots to play:")
    for i, (value, action) in enumerate(zip(top_values, top_actions)):
        shot_name = decode_shot(action.item())
        print(f"  {i+1}. {shot_name:15s} (Q-value: {value.item():+.3f})")
    print()


def test_agent(agent, rnn_model, num_tests=5):
    """
    Test the agent by playing a few rallies
    """
    print("\n" + "=" * 60)
    print("TESTING AGENT")
    print("=" * 60)
    
    agent.q_network.eval()
    rnn_model.eval()
    
    wins = 0
    
    for test_num in range(num_tests):
        rally = []
        current_player = 0
        
        print(f"\n--- Test Rally {test_num + 1} ---")
        
        for step in range(20):
            action = agent.choose_action(rally, training=False)
            rally.append(action)
            
            shot_name = decode_shot(action)
            print(f"  Shot {len(rally)}: {shot_name}")
            
            # Check if rally ends
            with torch.no_grad():
                seq = torch.tensor([rally], dtype=torch.long)
                predictions = rnn_model(seq).squeeze()
                win_prob = predictions[-1].item() if len(rally) > 1 else predictions.item()
            
            end_prob = min(0.7, len(rally) * 0.1)
            if win_prob > 0.85 or win_prob < 0.15:
                end_prob += 0.3
            
            if random.random() < end_prob or len(rally) >= 20:
                server_wins = random.random() < win_prob
                
                if current_player == 0:
                    won = server_wins
                else:
                    won = not server_wins
                
                outcome = "WON ✓" if won else "LOST ✗"
                print(f"  → Rally ended: {outcome}")
                
                if won:
                    wins += 1
                break
            
            current_player = 1 - current_player
    
    print(f"\n{'=' * 60}")
    print(f"Test Results: {wins}/{num_tests} wins ({wins/num_tests:.1%})")
    print("=" * 60 + "\n")



if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SIMPLE TENNIS RL SYSTEM")
    print("=" * 60)
    
    # Load RNN
    print("\nLoading RNN model...")
    rnn_model = RNN()
    
    try:
        rnn_model.load_state_dict(torch.load('tennis_rnn.pth'))
        print("✓ RNN loaded from tennis_rnn.pth")
    except:
        print("⚠ Could not load tennis_rnn.pth")
        print("  Using untrained RNN for demonstration")
    
    rnn_model.eval()
    
    # Train RL agent
    print("\nTraining RL agent...")
    agent, rewards, win_rates = train_simple_rl(rnn_model, num_episodes=1000)
    
    # Save agent
    torch.save(agent.q_network.state_dict(), 'simple_rl_agent.pth')
    print("✓ Agent saved to simple_rl_agent.pth")

    # Get recommendations for example rally
    example_rally = [2, 5, 1, 7, 3]
    get_recommendations(agent, example_rally)
    
    # Test the agent
    test_agent(agent, rnn_model, num_tests=5)
    
    print("\n" + "=" * 60)
    print("DONE! Your RL agent is ready to use.")
    print("=" * 60)
    print("\nTo get shot recommendations:")
    print("  get_recommendations(agent, current_rally)")


# The reinforcement model should, help with preidicting actual shots 