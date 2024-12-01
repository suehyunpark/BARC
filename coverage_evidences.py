import json
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
from scipy.stats import entropy
from heatmap_evidence_pairs import is_strictly_better

def calculate_coverage_metrics(verdicts: List[List[bool]], num_problems: int) -> Dict[str, float]:
    """
    Calculate various coverage metrics for a set of hypotheses.
    
    Args:
        verdicts: List of verdict lists [num_hypotheses][num_problems]
        num_problems: Number of problems/evidences
    
    Returns:
        Dictionary containing different coverage metrics
    """
    # Convert verdicts to binary patterns
    patterns = []
    for hypothesis_verdicts in verdicts:
        # Convert None/False/True to 0/1/2
        pattern = tuple(0 if v is None else (1 if v is False else 2) for v in hypothesis_verdicts)
        patterns.append(pattern)
    
    # 1. Pattern Diversity: How many unique verdict patterns exist
    unique_patterns = len(set(patterns))
    pattern_diversity = unique_patterns / len(patterns)  # Normalized by total hypotheses
    
    # 2. Pattern Distribution Entropy: How evenly distributed are the patterns
    pattern_counts = Counter(patterns)
    pattern_probs = np.array(list(pattern_counts.values())) / len(patterns)
    pattern_entropy = entropy(pattern_probs, base=2)  # Higher means more uniform distribution
    max_entropy = np.log2(len(patterns))  # Maximum possible entropy
    normalized_entropy = pattern_entropy / max_entropy if max_entropy > 0 else 0
    
    # 3. Problem Coverage: For each problem, what percentage of possible verdicts (None/False/True) are used
    verdict_coverage = []
    for prob_idx in range(num_problems):
        prob_verdicts = set(v[prob_idx] for v in verdicts)
        coverage = len(prob_verdicts) / 3  # Divided by 3 possible verdicts
        verdict_coverage.append(coverage)
    avg_problem_coverage = np.mean(verdict_coverage)
    
    # 4. Partial Order Depth: How many distinct levels exist in the partial ordering
    def count_better_hypotheses(idx):
        return sum(1 for j in range(len(verdicts)) if is_strictly_better(patterns[idx], patterns[j]))
    
    levels = set(count_better_hypotheses(i) for i in range(len(verdicts)))
    ordering_depth = len(levels) / len(verdicts)  # Normalized by total hypotheses
    
    return {
        "pattern_diversity": pattern_diversity,
        "pattern_entropy": normalized_entropy,
        "problem_coverage": avg_problem_coverage,
        "ordering_depth": ordering_depth,
        "composite_score": np.mean([pattern_diversity, normalized_entropy, 
                                  avg_problem_coverage, ordering_depth])
    }

def analyze_task_coverage(
    results_path: str,
    num_problems: int,
    num_hypotheses: int,
) -> Dict[str, Dict[str, float]]:
    """
    Analyze coverage metrics for each task in the results file.
    
    Args:
        results_path: Path to the results JSONL file
        num_problems: Number of problems/evidences
        num_hypotheses: Number of hypotheses
    
    Returns:
        Dictionary mapping task UIDs to their coverage metrics
    """
    coverage_results = {}
    
    with open(results_path) as f:
        for line in f:
            data = json.loads(line)
            uid = data['uid']
            verdicts = data['verdicts']
            
            metrics = calculate_coverage_metrics(verdicts, num_problems)
            coverage_results[uid] = metrics
    
    return coverage_results

# Example usage:
if __name__ == "__main__":
    results_path = "/mnt/nas/suehyun/BARC/results/arc_train_160/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333_tasks_50_evidences_30_exec_results_v4.jsonl"
    
    coverage_results = analyze_task_coverage(
        results_path=results_path,
        num_problems=30,
        num_hypotheses=64
    )
    
    # Print summary statistics
    metrics = list(next(iter(coverage_results.values())).keys())
    print("\nOverall Coverage Metrics:")
    for metric in metrics:
        values = [results[metric] for results in coverage_results.values()]
        print(f"{metric}:")
        print(f"  Mean: {np.mean(values):.3f}")
        print(f"  Std:  {np.std(values):.3f}")
        print(f"  Min:  {np.min(values):.3f}")
        print(f"  Max:  {np.max(values):.3f}")