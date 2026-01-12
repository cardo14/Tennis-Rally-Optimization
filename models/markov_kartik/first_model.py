import numpy as np
from collections import defaultdict
import pandas as pd

def build_markov_matrix(parsed_rallies):
    """
    parsed_rallies: list of dicts per point, e.g.
    [
        {'shots': [('FH',3), ('BH',2), ...], 'PtWinner': 1 or 2},
        ...
    ]
    """
    # Step 1: build all possible states
    states_set = set()
    for point in parsed_rallies:
        for shot in point['shots']:
            states_set.add(shot)  # shot is already a tuple (shot_type, dir)

    # Add absorbing states
    states_set.add('ServerWin')
    states_set.add('ReturnerWin')

    # Map state -> index
    state2idx = {s:i for i,s in enumerate(states_set)}
    idx2state = {i:s for s,i in state2idx.items()}
    n = len(state2idx)

    # Step 2: count transitions
    trans_counts = np.zeros((n,n), dtype=int)

    for point in parsed_rallies:
        shots = point['shots']
        if len(shots) == 0:
            continue

        for i in range(len(shots)-1):
            s_from = state2idx[shots[i]]
            s_to   = state2idx[shots[i+1]]
            trans_counts[s_from, s_to] += 1

        # last shot -> absorbing state
        last_shot = shots[-1]
        s_from = state2idx[last_shot]
        s_to = state2idx['ServerWin'] if point['PtWinner']==1 else state2idx['ReturnerWin']
        trans_counts[s_from, s_to] += 1

    # Step 3: convert counts -> probabilities
    trans_probs = np.zeros_like(trans_counts, dtype=float)
    for i in range(n):
        row_sum = trans_counts[i].sum()
        if row_sum > 0:
            trans_probs[i] = trans_counts[i] / row_sum
        else:
            # if no outgoing transitions, make absorbing
            if idx2state[i] not in ['ServerWin','ReturnerWin']:
                trans_probs[i, i] = 1.0

    return trans_probs, state2idx, idx2state


def compute_absorbing_probs(trans_probs, idx2state):
    """
    trans_probs: n x n matrix of transition probabilities
    idx2state: dict mapping index -> state
    Returns: dict of state -> (prob_server_wins, prob_returner_wins)
    """
    n = trans_probs.shape[0]

    # Identify absorbing vs transient states
    absorbing_idx = [i for i,s in idx2state.items() if s in ['ServerWin','ReturnerWin']]
    transient_idx = [i for i in range(n) if i not in absorbing_idx]

    # Q = transient -> transient, R = transient -> absorbing
    Q = trans_probs[np.ix_(transient_idx, transient_idx)]
    R = trans_probs[np.ix_(transient_idx, absorbing_idx)]

    # Fundamental matrix: N = (I-Q)^-1
    I = np.eye(len(Q))
    N = np.linalg.inv(I - Q)

    # Absorbing probabilities: B = N * R
    B = N @ R

    # Map results back to state names
    absorbing_states = [idx2state[i] for i in absorbing_idx]
    results = {}
    for idx, s_idx in enumerate(transient_idx):
        s_name = idx2state[s_idx]
        results[s_name] = {
            'prob_server_wins': B[idx, absorbing_states.index('ServerWin')],
            'prob_returner_wins': B[idx, absorbing_states.index('ReturnerWin')]
        }

    # Absorbing states themselves
    results['ServerWin'] = {'prob_server_wins': 1.0, 'prob_returner_wins': 0.0}
    results['ReturnerWin'] = {'prob_server_wins': 0.0, 'prob_returner_wins': 1.0}

    return results

def parse_rally(rally_str):
    if not rally_str or not isinstance(rally_str, str):
        return []

    # --- constants ---
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

    # --- skip serve encoding ---
    while i < n:
        if rally_str[i] in serve_digits or rally_str[i] in serve_letters or rally_str[i] == '+':
            i += 1
        else:
            break

    # --- parse rally ---
    while i < n:
        ch = rally_str[i]

        # terminal without explicit final shot (rare)
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
            "depth": None,
            "approach": False,
            "net_cord": False,
            "stop_volley": False,
            "position": None,
            "ending": None,
            "error_type": None,
        }

        i += 1

        # optional modifiers (order-independent, spec-legal)
        while i < n:
            c = rally_str[i]

            if c == '+':
                shot["approach"] = True
            elif c == ';':
                shot["net_cord"] = True
            elif c == '^':
                shot["stop_volley"] = True
            elif c == '-':
                shot["position"] = "net"
            elif c == '=':
                shot["position"] = "baseline"
            else:
                break
            i += 1

        # direction
        if i < n and rally_str[i] in {'0', '1', '2', '3'}:
            shot["dir"] = int(rally_str[i])
            i += 1

        # depth (returns only, optional)
        if i < n and rally_str[i] in {'7', '8', '9', '0'}:
            shot["depth"] = int(rally_str[i])
            i += 1

        # error type (optional)
        if i < n and rally_str[i] in error_types:
            shot["error_type"] = rally_str[i]
            i += 1

        # forced / unforced / winner
        if i < n and rally_str[i] in {'*', '@', '#'}:
            shot["ending"] = (
                "winner" if rally_str[i] == '*' else
                "unforced_error" if rally_str[i] == '@' else
                "forced_error"
            )
            i += 1
            shots.append(shot)
            break

        shots.append(shot)

    return shots

def simplify_rally(parsed_rally):
    """
    Take output from your existing parser and reduce it to a list of (shot, dir) tuples.
    Ignore depth, approach, error symbols, etc.
    Example: [{'shot':'f','dir':2, ...}, ...] -> [('F',2), ...]
    """
    simplified = []
    for shot_info in parsed_rally:
        shot_type = shot_info.get('shot', '').upper()
        dir = shot_info.get('dir', 0)
        simplified.append((shot_type, dir))
    return simplified

df = pd.read_csv("data/processed/points_hard_2022_2024.csv")

processed_rallies = []

for _, row in df.iterrows():
    # Use 2nd if it has data, else 1st
    rally_str = row['2nd'] if pd.notna(row['2nd']) and row['2nd'] != '' else row['1st']
    if not rally_str or rally_str in ['S','R']:
        continue  # skip points with missing or placeholder data

    # parse rally using your existing parser
    parsed_rally = parse_rally(rally_str)  # your parser function
    simplified_rally = simplify_rally(parsed_rally)
    
    processed_rallies.append({
        'shots': simplified_rally,
        'PtWinner': row['PtWinner']
    })

trans_probs, state2idx, idx2state = build_markov_matrix(processed_rallies)

# Compute absorbing probabilities
absorbing_probs = compute_absorbing_probs(trans_probs, idx2state)

# Example: probability server wins from a forehand crosscourt
state = ('F', 2)
print(absorbing_probs)
if state in absorbing_probs:
    print("Prob server wins from state", state, ":", absorbing_probs[state]['prob_server_wins'])
