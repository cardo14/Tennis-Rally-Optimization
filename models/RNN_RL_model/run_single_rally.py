import argparse
import torch
import torch.nn.functional as F

from next_shot_model import (
    parse_rally,
    simplify_rally,
    encode_shot,
    decode_shot,
    NextShotModel,
)


def run_rally_from_string(model_path, rally_str, window=5):
    parsed = parse_rally(rally_str)
    simplified = simplify_rally(parsed)
    encoded = [encode_shot(s) for s in simplified]

    if len(encoded) < 2:
        print('Rally too short for next-shot prediction.')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NextShotModel(window=window)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device).eval()

    correct = 0
    total = 0

    print('Rally shots:', [decode_shot(s) for s in encoded])
    for i in range(len(encoded) - 1):
        state = encoded[max(0, i - window + 1): i + 1]
        pad_len = window - len(state)
        state_padded = [0] * pad_len + state

        x = torch.LongTensor([state_padded]).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(probs.argmax())

        true_next = encoded[i + 1]
        match = (pred == true_next)
        print(f"Step {i+1}: state={[decode_shot(s) for s in state]} -> True: {decode_shot(true_next):10s} | Pred: {decode_shot(pred):10s} | Match: {match} | Prob: {probs[pred]:.3f}")

        total += 1
        if match:
            correct += 1

    print(f"\nAccuracy: {correct}/{total} = {correct/total:.3f}")


def run_rally_from_csv(model_path, csv_path, index, window=5):
    import pandas as pd

    df = pd.read_csv(csv_path)
    # detect rally column: prefer known names, else choose string column with long average length
    rally_col_candidates = ['rally', 'rally_str', 'rally_string', 'sequence', 'rallySequence']
    rally_col = None
    for c in rally_col_candidates:
        if c in df.columns:
            rally_col = c
            break

    if rally_col is None:
        # choose object dtype columns and pick the one with largest mean length and containing rally tokens
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols:
            raise ValueError('No string column found in CSV for rally data')

        def col_score(col):
            sample = df[col].dropna().astype(str).sample(min(200, max(1, df[col].dropna().shape[0])), random_state=1)
            lens = sample.map(len)
            # score: mean length times fraction containing shot letters
            shot_chars = set('frvoulhjbszpymikqt0123456789+*-@#')
            frac = (sample.map(lambda s: any((ch in shot_chars) for ch in s))).mean()
            return lens.mean() * frac

        scores = [(c, col_score(c)) for c in obj_cols]
        scores.sort(key=lambda x: x[1], reverse=True)
        rally_col = scores[0][0]

    rally_str = str(df[rally_col].iloc[index])
    print(f"Using column '{rally_col}' row {index}")
    run_rally_from_string(model_path, rally_str, window=window)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rally', help='Rally string to parse')
    group.add_argument('--csv', help='CSV file with rally strings')
    parser.add_argument('--index', type=int, default=0, help='Row index when using --csv')
    parser.add_argument('--window', type=int, default=5)
    args = parser.parse_args()

    if args.rally:
        run_rally_from_string(args.model, args.rally, window=args.window)
    else:
        run_rally_from_csv(args.model, args.csv, args.index, window=args.window)


if __name__ == '__main__':
    cli()
