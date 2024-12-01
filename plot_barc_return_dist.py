import matplotlib.pyplot as plt
import numpy as np
import json
import os
from tqdm import tqdm
from scipy.stats import norm
import random
random.seed(42)
from random import sample

def calculate_reward(verdicts, normalize=True):
    reward_map = {None: -2, False: -0.5, True: 1}
    rewards = sum(reward_map[v] for v in verdicts) 
    if normalize:
        rewards /= len(verdicts)
    return rewards

def calculate_task_stats(all_verdicts):
    norm_rewards = [calculate_reward(verdicts, normalize=True) for verdicts in all_verdicts]
    mean = np.mean(norm_rewards)
    std = np.std(norm_rewards)
    return mean, std

def categorize_distribution(mean, std, mean_threshold=0.0, std_threshold=0.5):
    if mean > mean_threshold:
        if std > std_threshold:
            return 0  # Positive mean, high std
        else:
            return 1  # Positive mean, low std
    else:
        if std > std_threshold:
            return 2  # Negative mean, high std
        else:
            return 3  # Other (negative mean, low std)

def plot_task_distributions(file_stats, output_path):
    plt.figure(figsize=(15, 8))
    
    # Define colors for each category
    colors = ['#1f77b4',  # Blue: positive mean, high std
             '#2ca02c',   # Green: positive mean, low std
             '#d62728',   # Red: negative mean, high std
             '#7f7f7f']  # Gray: other
    
    x = np.linspace(-2, 1, 300)
    
    # Categorize each distribution
    categorized_stats = []
    for stat in file_stats:
        category = categorize_distribution(stat['mean'], stat['std'])
        categorized_stats.append((stat, category))
    
    # Calculate counts for each category
    counts = [sum(1 for _, cat in categorized_stats if cat == i) for i in range(4)]
    
    # Calculate alpha values inversely proportional to count
    max_count = max(counts)
    alpha_values = [min(0.8, max_count/(count*3)) if count > 0 else 0.8 for count in counts]
    
    category_names = ['Positive μ, High σ',
                     'Positive μ, Low σ',
                     'Negative μ, High σ',
                     'Other']
    
    # Plot individual distributions by category with adjusted alpha
    for cat in range(3, -1, -1):
        first_in_cat = True
        for stat, category in categorized_stats:
            if category == cat:
                label = category_names[cat] if first_in_cat else None
                plt.plot(x, norm.pdf(x, stat['mean'], stat['std']), 
                        alpha=alpha_values[category], color=colors[category], 
                        label=label)
                first_in_cat = False
    
    # Calculate and plot overall distribution
    all_means = [stat['mean'] for stat, _ in categorized_stats]
    overall_mean = np.mean(all_means)
    overall_std = np.std(all_means)
    plt.plot(x, norm.pdf(x, overall_mean, overall_std), 
            color='black', linewidth=2, 
            label=f'Overall (μ={overall_mean:.2f}, σ={overall_std:.2f})')
    
    # Add category counts to legend
    legend_labels = [f'{name} (n={count}, α={alpha_values[i]:.2f})' 
                    for i, (name, count) in enumerate(zip(category_names, counts))]
    legend_labels += ['Overall']
    legend_handles = [plt.Line2D([0], [0], color=colors[i], lw=2) for i in range(4)]
    legend_handles += [plt.Line2D([0], [0], color='black', lw=2)]
    
    plt.title(f'Task Reward Distributions by Category\n{os.path.basename(output_path)}')
    plt.xlabel('Normalized Reward')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_handles, labels=legend_labels, loc='best')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Process each file
root = "/mnt/nas/suehyun/BARC/induction_samples"
samples = sample(os.listdir(root), 10)
# root = "/mnt/nas/suehyun/BARC/results"
# Create output directory
output_dir = f"{root}/reward_distributions"
os.makedirs(output_dir, exist_ok=True)
# samples = ["arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769_exec_results_v4.jsonl"]
for file in tqdm(samples):
    file_stats = []
    
    # Process each task
    with open(os.path.join(root, file)) as f:
        for line in f:
            d = json.loads(line)
            mean, std = calculate_task_stats(d["verdicts_per_examples"])
            file_stats.append({
                'uid': d['uid'],
                'mean': mean,
                'std': std
            })
    
    # Create plot
    output_path = os.path.join(output_dir, f"{file.replace('.jsonl', '_categorized.png')}")
    plot_task_distributions(file_stats, output_path)

print(f"Plots saved in {output_dir}/")