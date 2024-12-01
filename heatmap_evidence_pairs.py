import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from argparse import ArgumentParser
import random

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
    metadata: Dict[str, Any],
    num_problems: int,
    num_hypotheses: int,
    plot_dir: str,
    colors: List[str] = ['lightgray', 'lightsalmon', 'dodgerblue']
) -> None:
    """
    Create and save a heatmap visualization of verdicts.
    
    Args:
        data: Dictionary containing 'uid', 'verdicts', and 're-arc_io_pairs_indices'
        metadata: Dictionary containing PSO difficulties
        num_problems: Number of problems/evidences
        num_hypotheses: Number of hypotheses
        plot_dir: Directory to save the plots
        colors: List of colors for None, False, True verdicts
    """
    verdict_map = {None: 0, False: 1, True: 2}
    verdicts_matrix = np.zeros((num_problems, num_hypotheses))
    
    uid = data['uid']
    pso_difficulties = metadata[uid]["pso_difficulties"]
    pair_indices = data['re-arc_io_pairs_indices']
    difficulty_indices = sorted(range(len(pair_indices)), 
                             key=lambda i: pso_difficulties[pair_indices[i]])
    
    # Sort problems by difficulty
    verdicts = data['verdicts']
    sorted_verdicts = []
    for hypothesis_verdicts in verdicts:
        sorted_verdicts.append([hypothesis_verdicts[i] for i in difficulty_indices])
    
    # Sort hypotheses by partial ordering
    hypothesis_indices = list(range(len(sorted_verdicts)))
    hypothesis_indices.sort(key=lambda i: (
        count_predictions(sorted_verdicts[i]),
        sum(is_strictly_better(sorted_verdicts[j], sorted_verdicts[i]) 
            for j in range(len(sorted_verdicts)))
    ))
    
    # Fill the verdicts matrix
    for i, hypothesis_idx in enumerate(hypothesis_indices):
        for j in range(len(difficulty_indices)):
            verdicts_matrix[j, i] = verdict_map[sorted_verdicts[hypothesis_idx][j]]

    # Create heatmap with explicit vmin and vmax
    plt.figure(figsize=(20, 10))
    cmap = sns.color_palette(colors, as_cmap=True)
    heatmap = sns.heatmap(verdicts_matrix,
                         cmap=cmap,
                         vmin=0,
                         vmax=2,
                         xticklabels=range(1, num_hypotheses + 1),
                         yticklabels=range(1, num_problems + 1))

    # Adjust colorbar ticks to be centered on each color
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_ticks([0.33, 1, 1.67])  # Centers of each color segment
    colorbar.set_ticklabels(['None', 'False', 'True'])
    colorbar.set_label('Verdict')

    plt.title(f'Verdicts Heatmap - {uid}')
    plt.xlabel('Hypotheses (ordered by partial ordering)')
    plt.ylabel('IO pairs (ordered by PSO difficulty)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{uid}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def process_results_file(
    results_path: str,
    metadata_path: str,
    plot_dir: str,
    num_problems: int,
    num_hypotheses: int,
) -> None:
    """
    Process a results file and create heatmaps for each task.
    
    Args:
        results_path: Path to the results JSONL file
        metadata_path: Path to the metadata JSON file
        plot_dir: Directory to save the plots
        num_problems: Number of problems/evidences
        num_hypotheses: Number of hypotheses
    """
    random.seed(0)
    
    os.makedirs(plot_dir, exist_ok=True)
    metadata = json.load(open(metadata_path))
    
    with open(results_path) as f:
        for line in tqdm(f):
            data = json.loads(line)
            create_verdict_heatmap(
                data=data,
                metadata=metadata,
                num_problems=num_problems,
                num_hypotheses=num_hypotheses,
                plot_dir=plot_dir
            )

# Example usage:
if __name__ == "__main__":
    parser = ArgumentParser()
    # Configuration
    parser.add_argument("--results_path", 
                       default="results/arc_train_160/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333_tasks_50_evidences_30_exec_results_v4.jsonl",
                       help="Path to the results JSONL file")
    parser.add_argument("--metadata_path",
                       default="/mnt/nas/suehyun/ARC/dataset/RE-ARC/metadata.json",
                       help="Path to the metadata JSON file")
    parser.add_argument("--plot_dir",
                       default="results/arc_train_160/tasks_50_evidences_30_h_64_heatmaps",
                       help="Directory to save the heatmap plots")
    parser.add_argument("--num_problems",
                       default=30,
                       type=int,
                       help="Number of problems/evidences")
    parser.add_argument("--num_hypotheses",
                       default=64,
                       type=int,
                       help="Number of hypotheses")
    args = parser.parse_args()
    
    process_results_file(
        results_path=args.results_path,
        metadata_path=args.metadata_path,
        plot_dir=args.plot_dir,
        num_problems=args.num_problems,
        num_hypotheses=args.num_hypotheses,
    )