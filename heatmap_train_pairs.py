import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from argparse import ArgumentParser
import random
from arc.types import ArcIOPair, ArcProblem
from arc import train_problems, validation_problems

def get_arc_problem(uid: str) -> ArcProblem:
    """Get ARC problem by uid"""
    for problem in train_problems + validation_problems:
        if problem.uid == uid:
            return problem
    assert False, f"Problem {uid} not found"

def count_predictions(hypothesis_verdicts: List[bool]) -> Tuple[int, int]:
    """Count predictions with True > False > None"""
    true_count = sum(1 for v in hypothesis_verdicts if v is True)
    false_count = sum(1 for v in hypothesis_verdicts if v is False)
    return (true_count, false_count)

def is_strictly_better(h1_verdicts: List[bool], h2_verdicts: List[bool]) -> bool:
    """Check if h2 is strictly better than h1 (h1 < h2)"""
    if len(h1_verdicts) != len(h2_verdicts):
        return False
    
    has_extra = False
    for v1, v2 in zip(h1_verdicts, h2_verdicts):
        # Check True predictions
        if v1 is True and v2 is not True:
            return False
        if v1 is not True and v2 is True:
            has_extra = True
            continue
            
        # If neither has True, check False vs None
        if v1 is False and v2 is None:
            return False
        if v1 is None and v2 is False:
            has_extra = True
            
    return has_extra

def create_verdict_heatmap(
    data: Dict[str, Any],
    plot_dir: str,
    colors: List[str] = ['lightgray', 'lightsalmon', 'dodgerblue']
) -> None:
    """
    Create and save a heatmap visualization of verdicts for train problems only.
    
    Args:
        data: Dictionary containing 'uid' and 'verdicts_per_examples'
        plot_dir: Directory to save the plots
        colors: List of colors for None, False, True verdicts
    """
    verdict_map = {None: 0, False: 1, True: 2}
    
    uid = data['uid']
    arc_problem = get_arc_problem(uid)
    num_train_problems = len(arc_problem.train_pairs)
    
    # Get only training verdicts
    verdicts = data['verdicts_per_examples']  # shape: (num_hypotheses, num_all_problems)
    train_verdicts = [v[:num_train_problems] for v in verdicts]  # shape: (num_hypotheses, num_train_problems)
    num_hypotheses = len(train_verdicts)
    
    # Sort hypotheses by partial ordering
    hypothesis_indices = list(range(num_hypotheses))
    hypothesis_indices.sort(key=lambda i: (
        count_predictions(train_verdicts[i]),
        sum(is_strictly_better(train_verdicts[j], train_verdicts[i]) 
            for j in range(num_hypotheses))
    ))
    
    # Create and fill the verdicts matrix
    verdicts_matrix = np.zeros((num_train_problems, num_hypotheses))
    for i, hypothesis_idx in enumerate(hypothesis_indices):
        for j in range(num_train_problems):
            verdict = train_verdicts[hypothesis_idx][j]
            verdicts_matrix[j, i] = verdict_map[verdict]

    # Create heatmap with explicit vmin and vmax
    plt.figure(figsize=(20, 10))
    cmap = sns.color_palette(colors, as_cmap=True)
    heatmap = sns.heatmap(verdicts_matrix,
                         cmap=cmap,
                         vmin=0,
                         vmax=2,
                         xticklabels=range(1, num_hypotheses + 1),
                         yticklabels=range(1, num_train_problems + 1))

    # Adjust colorbar ticks to be centered on each color
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([0.33, 1, 1.67])  # Centers of each color segment
    colorbar.set_ticklabels(['None', 'False', 'True'])
    colorbar.set_label('Verdict')

    plt.title(f'Train Verdicts Heatmap - {uid}\n({num_train_problems} training problems)')
    plt.xlabel('Hypotheses (ordered by partial ordering)')
    plt.ylabel('Training IO pairs')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{uid}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def process_results_file(
    results_path: str,
    plot_dir: str,
    num_tasks: int = None
) -> None:
    """
    Process a results file and create heatmaps for each task.
    
    Args:
        results_path: Path to the results JSONL file
        plot_dir: Directory to save the plots
        num_tasks: Number of samples to process (optional)
    """
    random.seed(0)
    os.makedirs(plot_dir, exist_ok=True)
    
    all_data = [json.loads(line) for line in open(results_path)]
    if num_tasks:
        all_data = random.sample(all_data, num_tasks)
    
    for data in tqdm(all_data):
        create_verdict_heatmap(
            data=data,
            plot_dir=plot_dir
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_path", 
                       default="results/arc_train_240/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769_exec_results_v4.jsonl",
                       help="Path to the results JSONL file")
    parser.add_argument("--plot_dir",
                       default="results/arc_train_240/train_verdict_heatmaps",
                       help="Directory to save the heatmap plots")
    parser.add_argument("--num_tasks",
                       type=int,
                       help="Number of samples to process")
    args = parser.parse_args()
    
    process_results_file(
        results_path=args.results_path,
        plot_dir=args.plot_dir,
        num_tasks=args.num_tasks
    )