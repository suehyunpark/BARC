import os
import re
import json
import random
from argparse import ArgumentParser
from typing import List, Dict, Tuple, Optional
from utils import parse_code
from arc import train_problems
from tqdm import tqdm

def load_execution_results(results_dir: str) -> Dict:
    """Load and aggregate execution results from multiple files."""
    pattern = re.compile(r'.*_\d{16}_exec_results_v4\.jsonl$')
    results = {}
    
    for dir in os.listdir(results_dir):
        dir_path = os.path.join(results_dir, dir)
        if os.path.isdir(dir_path) and "train" in dir:
            for filename in os.listdir(dir_path):
                if pattern.match(filename) and "temp_0.8" in filename:
                    with open(os.path.join(dir_path, filename), 'r') as f:
                        for line in f:
                            d = json.loads(line)
                            uid = d["uid"]
                            if uid not in results:
                                results[uid] = d
                            else:
                                # Merge lists from multiple results
                                result = results[uid]
                                for (k1, v1), (k2, v2) in zip(result.items(), d.items()):
                                    if k1 == k2 and isinstance(v1, list) and isinstance(v2, list):
                                        result[k1] = v1 + v2
                                results[uid] = result
    return results

def get_problem_by_uid(uid: str) -> any:
    """Find ARC problem by UID."""
    for problem in train_problems:
        if problem.uid == uid:
            return problem
    return None

def format_problem_data(uid: str, train_examples: List[List[List[int]]], test_examples: List[List[List[int]]], trajectory: List[Tuple[int, str, List[bool]]]) -> Dict:
    """Format problem data according to the required structure."""
    return {
        "uid": uid,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "trajectory": trajectory
    }

def count_predictions(hypothesis_verdicts: List[bool]) -> Tuple[int, int]:
    """Count predictions with True > False > None"""
    true_count = sum(1 for v in hypothesis_verdicts if v is True)
    false_count = sum(1 for v in hypothesis_verdicts if v is False)
    return (true_count, false_count)

def count_verdicts(verdicts: List[bool]) -> Tuple[int, int, int]:
    """Count the number of True, False, and None verdicts."""
    true_count = sum(1 for v in verdicts if v is True)
    false_count = sum(1 for v in verdicts if v is False)
    none_count = sum(1 for v in verdicts if v is None)
    return (true_count, false_count, none_count)

def is_strictly_better(h1_verdicts: List[bool], h2_verdicts: List[bool]) -> bool:
    """Check if h2 is strictly better than h1 (h1 < h2)"""
    if len(h1_verdicts) != len(h2_verdicts):
        return False
    
    h1_true, h1_false, h1_none = count_verdicts(h1_verdicts)
    h2_true, h2_false, h2_none = count_verdicts(h2_verdicts)
    
    # If h2 has fewer True, it's not better
    if h2_true < h1_true:
        return False
        
    # If same True but h2 has fewer False, it's not better
    if h2_true == h1_true and h2_false < h1_false:
        return False
        
    # h2 is better if:
    # 1. More True, or
    # 2. Same True but more False, or
    # 3. Same True and False but fewer None
    return (h2_true > h1_true or 
            (h2_true == h1_true and h2_false > h1_false) or
            (h2_true == h1_true and h2_false == h1_false and h2_none < h1_none))

def get_correct_indices(verdicts: List[bool]) -> set:
    """Get indices where predictions are correct (True)."""
    return {i for i, v in enumerate(verdicts) if v is True}


def is_strictly_better(h1_verdicts: List[bool], h2_verdicts: List[bool]) -> int:
    """Compare h1 and h2 verdicts, returning:
    1 if h2 is strictly better than h1
    -1 if h1 is strictly better than h2 
    0 if they are incomparable or equal"""
    if len(h1_verdicts) != len(h2_verdicts):
        return 0
    
    h1_true, h1_false, h1_none = count_verdicts(h1_verdicts)
    h2_true, h2_false, h2_none = count_verdicts(h2_verdicts)
    
    # Check if h2 is strictly better than h1
    if (h2_true > h1_true or 
        (h2_true == h1_true and h2_false < h1_false) or
        (h2_true == h1_true and h2_false == h1_false and h2_none < h1_none)):
        return 1
        
    # Check if h1 is strictly better than h2
    if (h1_true > h2_true or
        (h1_true == h2_true and h1_false < h2_false) or
        (h1_true == h2_true and h1_false == h2_false and h1_none < h2_none)):
        return -1
        
    # Otherwise they are incomparable or equal
    return 0

def get_correct_indices(verdicts: List[bool]) -> set:
    """Get indices where predictions are correct (True)."""
    return {i for i, v in enumerate(verdicts) if v is True}

def sort_hypotheses(responses: List[str], verdicts_per_examples: List[List[bool]]) -> List[int]:
    """Sort hypotheses indices by their performance using strictly better ordering."""
    valid_indices = [i for i, resp in enumerate(responses) if parse_code(resp)]
    if not valid_indices:
        return []
    
    def index_comparator(idx1: int, idx2: int) -> int:
        return is_strictly_better(
            verdicts_per_examples[idx1],
            verdicts_per_examples[idx2]
        )
    
    return sorted(valid_indices, key=cmp_to_key(index_comparator))

def find_maximal_groups(verdicts_per_examples: List[List[bool]]) -> List[List[int]]:
    """Find maximal groups of hypotheses that form valid subset relationships."""
    n = len(verdicts_per_examples)
    if n < 2:
        return []
    
    def find_chains_from_start(start_idx):
        chains = []
        stack = [(start_idx, [start_idx])]  # (current_idx, current_chain)
        
        while stack:
            # print(f"Stack size: {len(stack)}")
            current_idx, current_chain = stack.pop()
            
            # Try to extend the chain
            extended = False
            for next_idx in range(n):
                if next_idx not in current_chain and is_strictly_better(
                    verdicts_per_examples[current_idx], 
                    verdicts_per_examples[next_idx]
                ):
                    extended = True
                    new_chain = current_chain + [next_idx]
                    stack.append((next_idx, new_chain))
            
            # If we couldn't extend this chain, it's complete
            if not extended and len(current_chain) > 1:
                chains.append(current_chain)
        
        return chains
    
    # Find all chains starting from each index
    all_chains = []
    for start in tqdm(range(n), desc="Finding chains"):
        chains = find_chains_from_start(start)
        all_chains.extend(chains)
    print(f"Found {len(all_chains)} chains")
    # Remove duplicates and subsumed chains
    maximal_chains = []
    for chain in sorted(all_chains, key=len, reverse=True):
        chain_set = set(chain)
        if not any(chain_set < set(existing) for existing in maximal_chains):
            if chain not in maximal_chains:
                maximal_chains.append(chain)
    
    return maximal_chains

def sort_by_partial_ordering(responses: List[str], verdicts_per_examples: List[List[bool]]) -> List[List[int]]:
    """Sort indices by partial ordering, returning valid subset groups."""
    
    # Get all valid code implementations
    valid_indices = [i for i, resp in enumerate(responses) if parse_code(resp)]
    if not valid_indices:
        return []
    
    # Filter verdicts to only include valid implementations
    filtered_verdicts = [verdicts_per_examples[i] for i in valid_indices]
    
    # Check if there exists a fully correct hypothesis
    if not any(all(v is True for v in verdicts) for verdicts in filtered_verdicts):
        return []
    
    # Find maximal groups
    groups = find_maximal_groups(filtered_verdicts)
    
    # Map back to original indices
    return [[valid_indices[i] for i in group] for group in groups]

def create_learning_trajectory(responses: List[str], verdicts_per_examples: List[List[bool]], 
                             groups: List[List[int]], trajectory_length: int) -> List[Tuple[int, str, List[bool]]]:
    """Create a learning trajectory from a group, ensuring proper subset progression."""
    
    if trajectory_length < 2:
        return []
    
    # Select the largest group that can accommodate the trajectory length
    suitable_groups = [g for g in groups if len(g) >= trajectory_length]
    if not suitable_groups:
        return []
    
    # Choose the group with the strongest final hypothesis
    chosen_group = max(suitable_groups, 
                      key=lambda g: (sum(1 for v in verdicts_per_examples[g[-1]] if v is True),
                                   -sum(1 for v in verdicts_per_examples[g[-1]] if v is False)))
    
    # Select evenly spaced indices from the group
    step_size = (len(chosen_group) - 1) / (trajectory_length - 1)
    selected_indices = []
    for i in range(trajectory_length):
        idx = chosen_group[min(int(i * step_size), len(chosen_group) - 1)]
        selected_indices.append(idx)
    
    return [(idx, parse_code(responses[idx])[0], verdicts_per_examples[idx]) 
            for idx in selected_indices]

def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="../results", help="Directory containing execution results")
    parser.add_argument("--trajectories_per_task", type=int, default=10, help="Number of trajectories per task")
    parser.add_argument("--max_trajectory_length", type=int, default=6, help="Maximum length of learning trajectories")
    parser.add_argument("--output_file", type=str, default="formatted_dataset.jsonl", help="Output JSONL file path")
    args = parser.parse_args()
    
    # Load and process results
    results = load_execution_results(args.results_dir)
    formatted_data = []
    task_cnt = 0
    
    for uid, data in tqdm(results.items(), total=len(results), desc="Processing results"):
        problem_trajectories = []
        responses = data["responses"]
        verdicts_per_examples = data["verdicts_per_examples"]
        grouped_indices = sort_by_partial_ordering(responses, verdicts_per_examples)
        
        # Try to generate multiple trajectories of different lengths
        for _ in range(args.trajectories_per_task):
            trajectory_length = random.randint(2, args.max_trajectory_length)
            trajectory = create_learning_trajectory(responses, verdicts_per_examples, grouped_indices, trajectory_length)
            
            if trajectory:
                problem_trajectories.append(trajectory)
        
        if not problem_trajectories:
            continue
            
        # Get corresponding ARC problem
        problem = get_problem_by_uid(uid)
        if not problem:
            continue
        task_cnt += 1
        
        train_examples = []
        for train_pair in problem.train_pairs:
            input_grid = train_pair.x.tolist()
            output_grid = train_pair.y.tolist()
            train_examples.append([input_grid, output_grid])
            
        test_examples = []
        for test_pair in problem.test_pairs:
            input_grid = test_pair.x.tolist()
            output_grid = test_pair.y.tolist()
            test_examples.append([input_grid, output_grid])
            
        # Format data for each implementation in the trajectory
        trajectory = []
        for i, (idx, code, verdicts) in enumerate(trajectory):
            # Find the output grids for this code implementation
            output_grids = data["output_grids"][idx]
            assert len(train_examples) + len(test_examples) == len(output_grids), f"Number of output grids does not match: {len(train_examples) + len(test_examples)} != {len(output_grids)}"
            uid_i = f"{uid}_{i}"
            trajectory.append({
                "uid": uid_i,
                "verdicts": verdicts,
                "code": code
            })
            
        formatted_problem = format_problem_data(uid_i, train_examples, test_examples, trajectory)
        formatted_data.append(formatted_problem)
    
    # Save formatted data
    output_file = args.output_file
    output_file = output_file.replace(".jsonl", f"_ntasks-{task_cnt}_ninstances-{len(formatted_data)}.jsonl")
    with open(output_file, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Processed {len(formatted_data)} examples")

if __name__ == "__main__":
    main()