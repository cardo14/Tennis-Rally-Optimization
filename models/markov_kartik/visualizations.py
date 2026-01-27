# visualizations.py
import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_shot_name(shot_code):
    """just converts the number back to a readable shot name"""
    if shot_code < 4:
        return f"FH-{shot_code}"
    elif shot_code < 8:
        return f"BH-{shot_code - 4}"
    else:
        return "OTHER"

def visualize_single_rally(rally_num, seq, label, predictions):
    """makes a nice graph for one rally showing probabilities"""
    # figure out how long the rally actually is (ignore padding zeros)
    rally_len = len([s for s in seq if s != 0])
    
    # get the shot numbers for x-axis
    shot_numbers = list(range(1, rally_len + 1))
    probs = predictions[:rally_len]
    
    # calculate how much each shot changed the win probability
    advantages = []
    for i in range(rally_len):
        if i == 0:
            # first shot compared to 50-50
            advantages.append(probs[i] - 0.5)
        else:
            # compare to previous shot
            advantages.append(probs[i] - probs[i-1])
    
    # make the figure with 2 subplots stacked on top of each other
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    outcome = 'Server Won' if label[0] == 1 else 'Returner Won'
    fig.suptitle(f'Rally {rally_num} - {outcome}', fontsize=14, fontweight='bold')
    
    # top plot: win probability over time
    ax1.plot(shot_numbers, probs, marker='o', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% line')
    ax1.fill_between(shot_numbers, 0.5, probs, alpha=0.3, color='lightblue')
    ax1.set_ylabel('Server Win Probability', fontsize=11)
    ax1.set_title('Win Probability After Each Shot', fontsize=12)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # bottom plot: momentum shifts (green = good for server, red = bad)
    colors = ['green' if adv >= 0 else 'red' for adv in advantages]
    bars = ax2.bar(shot_numbers, advantages, color=colors, alpha=0.6, edgecolor='black')
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('Shot Number', fontsize=11)
    ax2.set_ylabel('Momentum Shift', fontsize=11)
    ax2.set_title('How Much Each Shot Changed Win Probability', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # highlight the best shot with a gold border
    best_idx = np.argmax(np.abs(advantages))
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    return fig

def compare_multiple_rallies(test_rallies, model):
    """puts a bunch of rallies in one image to compare them"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Comparing Different Rallies', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    with torch.no_grad():
        for idx in range(min(len(test_rallies), 6)):
            seq, label = test_rallies[idx]
            seq_tensor = torch.tensor([seq], dtype=torch.long)
            predictions = model(seq_tensor).squeeze().numpy()
            
            # fix for single-shot rallies
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            
            # get actual rally length
            rally_len = len([s for s in seq if s != 0])
            shot_numbers = list(range(1, rally_len + 1))
            probs = predictions[:rally_len]
            
            # plot it
            ax = axes[idx]
            ax.plot(shot_numbers, probs, marker='o', linewidth=2, markersize=6, color='purple')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.fill_between(shot_numbers, 0.5, probs, alpha=0.2, color='purple')
            
            outcome = 'Server Won' if label[0] == 1 else 'Returner Won'
            ax.set_title(f'Rally {idx+1}: {outcome}', fontsize=10)
            ax.set_xlabel('Shot #')
            ax.set_ylabel('Win Prob')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
    
    # turn off extra subplots if we have less than 6 rallies
    for idx in range(len(test_rallies), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def plot_dataset_stats(dataset):
    """shows what kinds of shots appear in the dataset"""
    # count each shot type
    shot_counts = {i: 0 for i in range(9)}
    
    for seq, _ in dataset:
        for shot in seq:
            if shot != 0:  # ignore padding
                shot_counts[shot] = shot_counts.get(shot, 0) + 1
    
    shot_labels = [decode_shot_name(i) for i in range(9)]
    counts = [shot_counts[i] for i in range(9)]
    
    # make a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red']*4 + ['blue']*4 + ['green']  # red for FH, blue for BH, green for other
    ax.bar(shot_labels, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Shot Type', fontsize=12)
    ax.set_ylabel('How Many Times This Shot Appears', fontsize=12)
    ax.set_title('Distribution of Shots in Dataset', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig

def plot_rally_lengths(dataset):
    """shows how long rallies typically are"""
    rally_lengths = []
    
    for seq, _ in dataset:
        length = len([s for s in seq if s != 0])
        rally_lengths.append(length)
    
    # histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rally_lengths, bins=range(1, max(rally_lengths)+2), 
            color='teal', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Shots in Rally', fontsize=12)
    ax.set_ylabel('Number of Rallies', fontsize=12)
    ax.set_title('How Long Are The Rallies?', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # add a line showing the average
    avg_length = np.mean(rally_lengths)
    ax.axvline(avg_length, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_length:.1f} shots')
    ax.legend()
    
    plt.tight_layout()
    return fig

def generate_all_visualizations(dataset, test_rallies, model):
    """main function to create all the graphs"""
    print("\n" + "="*60)
    print("MAKING VISUALIZATIONS")
    print("="*60)
    
    # 1. show stats about the whole dataset
    print("\n1. Creating dataset overview plots...")
    fig1 = plot_dataset_stats(dataset)
    fig1.savefig('shot_distribution.png', dpi=150, bbox_inches='tight')
    print("   ✓ saved shot_distribution.png")
    
    fig2 = plot_rally_lengths(dataset)
    fig2.savefig('rally_lengths.png', dpi=150, bbox_inches='tight')
    print("   ✓ saved rally_lengths.png")
    
    # 2. compare a bunch of rallies side by side
    print("\n2. Creating rally comparison...")
    fig3 = compare_multiple_rallies(test_rallies, model)
    fig3.savefig('rally_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ saved rally_comparison.png")
    
    # 3. detailed analysis of first 3 rallies
    print("\n3. Creating detailed rally analyses...")
    with torch.no_grad():
        for i in range(min(3, len(test_rallies))):
            seq, label = test_rallies[i]
            seq_tensor = torch.tensor([seq], dtype=torch.long)
            predictions = model(seq_tensor).squeeze().numpy()
            
            # fix for single-shot rallies
            if predictions.ndim == 0:
                predictions = np.array([predictions])
            
            fig = visualize_single_rally(i + 1, seq, label, predictions)
            fig.savefig(f'rally_{i+1}_detailed.png', dpi=150, bbox_inches='tight')
            print(f"   ✓ saved rally_{i+1}_detailed.png")
    
    print("\n" + "="*60)
    print("done! check the folder for your graphs")
    print("="*60)