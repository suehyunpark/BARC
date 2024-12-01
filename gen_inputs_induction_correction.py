from functools import cmp_to_key
import os
import re
import json
import random
from argparse import ArgumentParser
from typing import List, Dict, Tuple, Optional
from utils import parse_code
from arc import train_problems
from tqdm import tqdm

random.seed(0)

def load_execution_results(results_dir: str) -> Dict:
    """Load and aggregate execution results from multiple files."""
    pattern = re.compile(r'.*Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_\d{16}_timeout-8_exec_results_v4\.jsonl$')
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


def format_problem_data(uid: str, train_examples: List[List[List[int]]], test_examples: List[List[List[int]]], trajectory: List[Tuple[int, str, List[bool]]], candidates_cache: Dict[Tuple[int, Tuple[int]], List[Tuple[int, set]]]) -> Dict:
    """Format problem data according to the required structure."""
    candidates = []
    if candidates_cache is not None:
        candidates = [
            {
                "num_true_verdicts": k,
            "previous_correctness_superset": list(previous_S)
        }
        for k, previous_S in candidates_cache.keys()
    ]
    return {
        "uid": uid,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "candidates_cache": candidates,
        "trajectory": trajectory
    }


def count_verdicts(verdicts: List[bool]) -> Tuple[int, int, int]:
    """Count the number of True, False, and None verdicts."""
    true_count = 0
    false_count = 0
    none_count = 0
    
    for v in verdicts:
        if v is True:
            true_count += 1
        elif v is False:
            false_count += 1
        else:
            none_count += 1
            
    return (true_count, false_count, none_count)


def create_learning_trajectories(responses: List[str], verdicts_per_examples: List[List[bool]], test_indices: List[int],
                               num_trajectories: int) -> List[List[Tuple[int, str, List[bool]]]]:
    """Create multiple learning trajectories with adaptive length adjustment."""
    # Compute correctness sets
    correctness_sets = []
    for h_idx, verdicts in enumerate(verdicts_per_examples):
        correct_indices = set(idx for idx, v in enumerate(verdicts) if v is True)
        correctness_sets.append((h_idx, correct_indices))

    # Identify the best hypothesis (last hypothesis in the chain)
    max_size = max(len(cs[1]) for cs in correctness_sets)
    
    # Filter best hypotheses to only include unique code implementations
    best_hypotheses = []
    seen_code = set()
    for cs in correctness_sets:
        if len(cs[1]) == max_size:
            code = parse_code(responses[cs[0]])[0]
            if code not in seen_code:
                seen_code.add(code)
                best_hypotheses.append(cs)
    
    def construct_hypothesis_chain(h_max_idx: int, S_max: set, candidates_cache: Dict[Tuple[int, Tuple[int]], List[Tuple[int, set]]]={}) -> List[Tuple[int, set]]:
        # Initialize the chain
        chain = [(h_max_idx, S_max)]

        # Construct the chain backwards
        previous_S = S_max
        
        for k in range(max_size - 1, -1, -1):  # possible sizes
            # Find hypotheses with correctness set size k and subset of previous
            candidates = candidates_cache.get((k, tuple(previous_S)))  # cache candidates for subsequent trajectory generation
            if not candidates:  # if no cache, then generate candidates
                candidates = [
                    (h_idx, S_h)
                    for h_idx, S_h in correctness_sets
                    if len(S_h) == k and S_h < previous_S
                ]
                
            if not candidates:  # even after creating a cache, no candidates of this size
                continue
            else:
                candidates_cache[(k, tuple(previous_S))] = candidates
            
            # Randomly select one candidate
            h_k_idx, S_k = random.choice(candidates)
            chain.append((h_k_idx, S_k))
            previous_S = S_k
        
        # if chain[-1][1] == set() and len(chain) > 2:  # if the last chain is empty (incorrect hypothesis) then can start from the partially correct hypothesis instead of a completely incorrect one
        if random.random() < 0.5:  # 50% chance to start from (partially) correct hypothesis to prevent unnecessary long chains
            chain = chain[:-1]
        
        # Reverse the chain to have increasing correctness
        chain.reverse()
        return [h_idx for h_idx, _ in chain], candidates_cache
    
    trajectories = []
    excluded_chains = set()  # Track used chains to avoid duplicates
    candidates_cache = {}
    patience = 3
    
    while len(trajectories) < num_trajectories and patience > 0:
        h_max_idx, S_max = random.choice(best_hypotheses)  # There can be multiple unique best hypotheses
        chain, candidates_cache = construct_hypothesis_chain(h_max_idx, S_max, candidates_cache)
        
        # Convert chain to tuple to make it hashable and preserve order
        chain_tuple = tuple(chain)
        if chain:
            if chain_tuple not in excluded_chains:
                excluded_chains.add(chain_tuple)
                # Convert chain to trajectory format
                trajectory = [
                    (idx, parse_code(responses[idx])[0], verdicts_per_examples[idx])
                    for idx in chain
                ]
                trajectories.append(trajectory)
                print(' -> '.join([f"#{len(trajectories)}: H{idx} {verdicts}" for idx, _, verdicts in trajectory]))
            else:
                patience -= 1
    
    return trajectories, candidates_cache


def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="../results", help="Directory containing execution results")
    parser.add_argument("--trajectories_per_task", type=int, default=1, help="Number of trajectories per task")
    parser.add_argument("--output_file", type=str, default="formatted_dataset.jsonl", help="Output JSONL file path")
    # parser.add_argument("--warm_start", action="store_true", help="Warm start from partially successful hypothesis")
    args = parser.parse_args()
    
    # Load and process results
    results = load_execution_results(args.results_dir)
    formatted_data = []
    task_cnt = 0
    
    for uid, data in tqdm(results.items(), total=len(results), desc="Processing results"):
        responses = data["responses"]
        train_verdicts = data["train_verdicts"]
        train_test_verdicts = data["train_test_verdicts"]
        test_indices = list(set(range(len(train_test_verdicts[0]))) - set(range(len(train_verdicts[0]))))
        verdicts_per_examples = data["verdicts_per_examples"]
        if not any(verdict for verdict_per_example in verdicts_per_examples for verdict in verdict_per_example):  # Need at least one True verdict
            continue
        
        # Get corresponding ARC problem
        problem = get_problem_by_uid(uid)
        if not problem:
            continue
        
        problem_trajectories, candidates_cache = create_learning_trajectories(
            responses,
            verdicts_per_examples,
            test_indices,
            args.trajectories_per_task,
        )
        if candidates_cache is not None:
            print(f"Candidates cache size: {len(candidates_cache)}, keys: {list(candidates_cache.keys())}")
        
        if not problem_trajectories:
            continue
        task_cnt += 1
        
        print(f"Generated {len(problem_trajectories)} trajectories for task {uid}\n")
        
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
        for i, trajectory in enumerate(problem_trajectories):
            uid_i = f"{uid}_{i}"
            trajectory_data = []
            for idx, code, verdicts in trajectory:
                # Find the output grids for this code implementation
                output_grids = data["output_grids"][idx]
                assert len(train_examples) + len(test_examples) == len(output_grids), f"Number of output grids does not match: {len(train_examples) + len(test_examples)} != {len(output_grids)}"
                trajectory_data.append({
                    "verdicts": verdicts,
                    "source": code
                })
                # print(f"#{idx+1}: H{idx} {verdicts}")
                # print(f"\t{code}")
            formatted_problem = format_problem_data(uid_i, train_examples, test_examples, trajectory_data, candidates_cache)
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