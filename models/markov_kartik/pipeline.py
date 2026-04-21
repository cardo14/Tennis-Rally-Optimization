import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from base_model import WinProbabilityModel
from hmm_model import HMMModel, MarkovChainModel
from rnn_model import RNNModel
from entropy_rl_model import EntropyRegRL
from evaluation.evaluator import evaluate_models, print_table, rank_models


SHOT_TYPE_MAP = {
    'f': 0, 'b': 1, 'r': 2, 's': 3,
    'v': 4, 'z': 5, 'o': 6, 'u': 7,
    'other': 8
}
NUM_SHOT_TYPES = len(SHOT_TYPE_MAP)


def parse_shot(shot_char):
    return SHOT_TYPE_MAP.get(shot_char.lower(), SHOT_TYPE_MAP['other'])


def parse_rally_notes(notes_str):
    if not isinstance(notes_str, str) or len(notes_str) == 0:
        return []
    shots = []
    for char in notes_str:
        if char.isalpha():
            shots.append(parse_shot(char))
    return shots if len(shots) >= 2 else []


def load_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  {len(df)} points loaded")

    rallies = []
    outcomes = []

    for _, row in df.iterrows():
        shots = parse_rally_notes(str(row.get("Notes", "")))
        if not shots:
            continue
        pt_winner = row.get("PtWinner", None)
        svr = row.get("Svr", 1)
        if pd.isna(pt_winner):
            continue
        outcome = 1 if int(pt_winner) == int(svr) else 0
        rallies.append(shots)
        outcomes.append(outcome)

    print(f"  {len(rallies)} valid rallies parsed")
    return rallies, outcomes


def run_pipeline():
    data_path = "../../data/processed/points_hard_2022_2024.csv"

    rallies, outcomes = load_data(data_path)

    train_r, test_r, train_o, test_o = train_test_split(
        rallies, outcomes, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(train_r)} rallies | Test: {len(test_r)} rallies")

    all_shots = list(range(NUM_SHOT_TYPES))

    # initialize models
    hmm = HMMModel()
    markov = MarkovChainModel()
    rnn = RNNModel(vocab_size=NUM_SHOT_TYPES + 1, embed_dim=16, hidden_dim=32,
                   max_len=30, epochs=20, batch_size=32)
    rl = EntropyRegRL(action_dim=NUM_SHOT_TYPES, alpha=0.1, lr=0.001,
                      episodes=500, gamma=0.95)

    models = [hmm, markov, rnn, rl]

    # train
    print("\n--- Training ---")
    for model in models:
        print(f"\nTraining {model.get_name()}...")
        model.fit(train_r, train_o)

    # optionally load your existing RNN weights instead of retraining
    # rnn.load_existing("tennis_rnn.pth")

    # evaluate
    print("\n--- Evaluating ---")
    results = evaluate_models(models, test_r, test_o, all_shots)

    print_table(results)
    rank_models(results)

    # RL opponent analysis on first test rally
    print("\n--- Entropy-Reg RL: Opponent Behavior ---")
    sample_rally = test_r[0] if test_r else [0, 1, 2]
    opp_dist = rl.get_opponent_distribution(sample_rally)
    opt_action = rl.get_optimal_action(sample_rally)

    shot_names = {v: k for k, v in SHOT_TYPE_MAP.items()}

    print("\nOpponent shot distribution (learned):")
    for i, prob in enumerate(opp_dist):
        print(f"  {shot_names.get(i, i)}: {prob:.3f}")

    print("\nRecommended next shot distribution:")
    for i, prob in enumerate(opt_action):
        bar = "█" * int(prob * 30)
        print(f"  {shot_names.get(i, i)}: {prob:.3f}  {bar}")

    entropy = -np.sum(opt_action * np.log(opt_action + 1e-9))
    print(f"\nPolicy entropy: {entropy:.3f}  (higher = more mixed strategy)")


if __name__ == "__main__":
    run_pipeline()