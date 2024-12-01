import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
import pickle
from argparse import ArgumentParser

def calculate_reward(verdicts, normalize=True):
    reward_map = {None: -2.0, False: -0.5, True: 1.0}
    rewards = sum(reward_map[v] for v in verdicts)
    if normalize:
        rewards /= len(verdicts)
    return rewards

def collect_task_rewards(root_dir, files):
    """Collect rewards for each task across all runs."""
    task_rewards = defaultdict(list)
    
    for file in tqdm(files, desc="Processing files"):
        with open(os.path.join(root_dir, file)) as f:
            for line in f:
                d = json.loads(line)
                task_id = d['uid']
                # Calculate rewards for each example in this task
                for verdicts in d['verdicts_per_examples']:
                    reward = calculate_reward(verdicts, normalize=True)
                    task_rewards[task_id].append(reward)
    
    return task_rewards

def collect_rewards_stats(rewards):
    # Add statistical annotations
    mean_reward = np.mean(rewards)
    median_reward = np.median(rewards)
    std_reward = np.std(rewards)
    
    return {
        'n': len(rewards),
        'mean': mean_reward,
        'median': median_reward,
        'std': std_reward,
        'min': min(rewards),
        'max': max(rewards)
    }
    

def plot_task_reward_distribution(rewards, stats, task_id, output_path):
    """Create a detailed distribution plot for a single task."""
    plt.figure(figsize=(8, 4))
    plt.xlim(-2, 1)
    
    # Create histogram with KDE
    sns.histplot(rewards, stat='density', bins=30, alpha=0.6, color='skyblue')
    sns.kdeplot(rewards, color='darkblue', linewidth=2)
    
    # Add vertical lines for mean and median
    n, mean_reward, median_reward, std_reward, min_reward, max_reward = stats['n'], stats['mean'], stats['median'], stats['std'], stats['min'], stats['max']
    plt.axvline(mean_reward, color='red', linestyle='--', label=f'Mean: {mean_reward:.3f}')
    plt.axvline(median_reward, color='green', linestyle='--', label=f'Median: {median_reward:.3f}')
    
    # Add text box with statistics
    stats_text = (f'n: {n}\n'
                 f'Mean: {mean_reward:.3f}\n'
                 f'Median: {median_reward:.3f}\n'
                 f'Std: {std_reward:.3f}\n'
                 f'Min: {min_reward:.3f}\n'
                 f'Max: {max_reward:.3f}')
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=8)
    
    plt.title(f'Reward Distribution for Task: {task_id}')
    plt.xlabel('Normalized Reward')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    

def main(args):
    files = os.listdir(args.results_root)
    
    # Create output directory for individual task plots
    output_root = args.output_root
    output_plot_dir = os.path.join(output_root, "task_reward_distribution_plots")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)
    
    # Collect all rewards for each task
    task_rewards_file = os.path.join(output_root, "task_rewards.pkl")
    if os.path.exists(task_rewards_file):
        with open(task_rewards_file, "rb") as f:
            task_rewards = pickle.load(f)
        if len(task_rewards) < 400:
            task_rewards = collect_task_rewards(args.results_root, files)
            with open(task_rewards_file, "wb") as f:
                pickle.dump(task_rewards, f)
    else:
        task_rewards = collect_task_rewards(args.results_root, files)
        with open(task_rewards_file, "wb") as f:
            pickle.dump(task_rewards, f)
            
    # Collect stats
    task_stats_file = os.path.join(output_root, "task_stats.json")
    if os.path.exists(task_stats_file):
        with open(task_stats_file) as f:
            task_stats = json.load(f)
        if len(task_stats) < 400:
            task_stats = {task_id: collect_rewards_stats(rewards) for task_id, rewards in task_rewards.items()}
            with open(task_stats_file, "w") as f:
                json.dump(task_stats, f, indent=4)
    else:
        task_stats = {task_id: collect_rewards_stats(rewards) for task_id, rewards in task_rewards.items()}
        with open(task_stats_file, "w") as f:
            json.dump(task_stats, f, indent=4)
            
    # Create individual plots for each task
    for (task_id, rewards), stats in tqdm(zip(task_rewards.items(), task_stats.values()), desc="Creating plots"):
        output_path = os.path.join(output_plot_dir, f"task_{task_id}_distribution.png")
        plot_task_reward_distribution(rewards, stats, task_id, output_path)
    
    print(f"Individual task rewards, stats, plots saved in {output_root}/")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_root", type=str, default="/mnt/nas/suehyun/BARC/induction_samples/ARC-Potpourri")
    parser.add_argument("--output_root", type=str, default="/mnt/nas/suehyun/BARC/reward_distributions/ARC_Potpourri_Induction_8B_80runs")
    args = parser.parse_args()
    main(args)