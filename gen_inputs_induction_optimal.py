import os
import re
import json
import random
from argparse import ArgumentParser
from typing import List, Dict, Tuple
from utils import parse_code
from arc import train_problems

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

def get_unique_successful_codes(data: Dict) -> Tuple[List[int], List[str]]:
    """Extract unique successful code implementations."""
    successful_pairs = [(i, data["responses"][i]) for i, v in enumerate(data["train_test_verdicts"]) if v]
    if not successful_pairs:
        return [], []
    successful_indices, successful_responses = zip(*successful_pairs)
    unique_codes = set()
    unique_indices = set()
    for idx, response in zip(successful_indices, successful_responses):
        parsed_codes = parse_code(response)
        if parsed_codes and idx not in unique_indices:
            unique_codes.add(parsed_codes[0])
            unique_indices.add(idx)
    return list(unique_indices), list(unique_codes)

def get_problem_by_uid(uid: str) -> any:
    """Find ARC problem by UID."""
    for problem in train_problems:
        if problem.uid == uid:
            return problem
    return None

def format_problem_data(uid: str, train_examples: List[List[List[int]]], test_examples: List[List[List[int]]], code: str) -> Dict:
    """Format problem data according to the required structure."""
    return {
        "uid": uid,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "source": code
    }

def main():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="../results", help="Directory containing execution results")
    parser.add_argument("--samples_per_task", type=int, default=4, help="Number of samples to take per task")
    parser.add_argument("--output_file", type=str, default="formatted_dataset.jsonl", help="Output JSONL file path")
    args = parser.parse_args()
    
    # Load and process results
    results = load_execution_results(args.results_dir)
    formatted_data = []
    task_cnt = 0
    for uid, data in results.items():
        # Get unique successful implementations
        unique_indices, unique_codes = get_unique_successful_codes(data)
        
        if not unique_codes:
            continue
            
        # Sample n unique implementations
        sampled_indices, sampled_codes = zip(*random.sample(list(zip(unique_indices, unique_codes)), min(args.samples_per_task, len(unique_codes))))
        
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
            
        # Format data for each sampled implementation
        for i, (idx, code) in enumerate(zip(sampled_indices, sampled_codes)):
            # Find the output grids for this code implementation
            output_grids = data["output_grids"][idx]
            assert len(train_examples) + len(test_examples) == len(output_grids), f"Number of output grids does not match: {len(train_examples) + len(test_examples)} != {len(output_grids)}"
            newline = "\n"
            for j, example in enumerate(train_examples + test_examples):
                assert example[1] == output_grids[j], f"Output grids do not match {uid}, {j}th, {data['verdicts_per_examples'][idx]}: {newline.join(map(str, example[1]))} != {newline.join(map(str, output_grids[j]))}"
            uid_i = f"{uid}_{i}"
            
            formatted_problem = format_problem_data(uid_i, train_examples, test_examples, code)
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