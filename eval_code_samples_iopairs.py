import os
import pathlib
import traceback
import sys
from typing import List
from tqdm import tqdm

def trace_calls(frame, event, arg):
    if event != 'call':
        return
    co = frame.f_code
    func_name = co.co_name
    if func_name == 'execve':
        filename = co.co_filename
        line_no = frame.f_lineno
        if 'lscpu' in str(arg):
            print(f"lscpu called from {filename}:{line_no}")
            traceback.print_stack(frame)
    return trace_calls

sys.settrace(trace_calls)

# Rest of your imports and code below this line


import json
from enum import Enum
# extract markdown code blocks
from utils import parse_code
from execution import multi_execute_transformation
from seeds.common import *
import argparse
import os

from arc.types import ArcIOPair



TRANSPOSE = False

MULTI_EXECUTE = True

class GridComparisonResult(Enum):
    EQUAL = 0
    SHAPE_MISMATCH = 1
    CONTENT_MISMATCH = 2
    TYPE_MISMATCH = 3
    ERROR = 4
    NON_2D_ARRAY = 5

def compare_grids(output_grid, expected_output_grid):
    if isinstance(output_grid, str):
        return GridComparisonResult.ERROR, 0.0
    
    if not isinstance(output_grid, np.ndarray):
        return GridComparisonResult.TYPE_MISMATCH, 0.0
    
    if len(output_grid.shape) != 2:
        return GridComparisonResult.NON_2D_ARRAY, 0.0
    
    if output_grid.shape != expected_output_grid.shape:
        return GridComparisonResult.SHAPE_MISMATCH, 0.0
    
    if np.array_equal(output_grid, expected_output_grid):
        return GridComparisonResult.EQUAL, 1.0
    
    # If shapes match but content doesn't, calculate the ratio of matching elements
    ratio = np.sum(output_grid == expected_output_grid) / np.prod(expected_output_grid.shape)
    return GridComparisonResult.CONTENT_MISMATCH, ratio


def validate(io_pair, code, timeout=2):
    """Validate a single ArcIOPair against a code implementation."""
    failure = False
    return_output_grids = []

    if TRANSPOSE:
        input_grid = io_pair.x.T
        expected_output_grid = io_pair.y.T
    else:
        input_grid = io_pair.x
        expected_output_grid = io_pair.y

    try:
        output_grids = multi_execute_transformation([code], [input_grid], random_seeds=[0], timeout=timeout, 
                                                    function_name="transform", num_workers=32)
        output_grid = output_grids[0]
    except KeyboardInterrupt:
        exit()
    except Exception as e:
        output_grid = "error"
        print(e)

    comparison_result, ratio = compare_grids(output_grid, expected_output_grid)
    
    if isinstance(output_grid, np.ndarray):
        return_output_grids.append(output_grid.astype(int).tolist())
    else:
        return_output_grids.append(output_grid)

    if comparison_result != GridComparisonResult.EQUAL:
        failure = True
        if comparison_result == GridComparisonResult.ERROR:
            print(f"\t\t[-] Error occurred: {output_grid}")
        elif comparison_result == GridComparisonResult.TYPE_MISMATCH:
            print("\t\t[-] output is not a numpy array")
        elif comparison_result == GridComparisonResult.SHAPE_MISMATCH:
            print(f"\t\t[-] output shape does not match expected shape: {output_grid.shape} vs {expected_output_grid.shape}")
        elif comparison_result == GridComparisonResult.CONTENT_MISMATCH:
            print(f"\t\t[-] comparison failed, ratio of correct elements: {ratio}")

    if not failure: 
        print(f"\t[+] passed")

    return (not failure, ratio, return_output_grids)


def multi_validate(io_pairs, codes, timeout=2):
    """Validate multiple codes against multiple IO pairs."""
    results = [list() for _ in range(len(codes))]
    return_output_grids = [[] for _ in range(len(codes))]
    
    for pair_idx, io_pair in enumerate(io_pairs):
        input_grid = io_pair.x
        try:
            output_grids = multi_execute_transformation(codes, [input_grid]*len(codes), 
                                                      random_seeds=[0]*len(codes),
                                                      timeout=timeout, function_name="transform", 
                                                      num_workers=64)
        except KeyboardInterrupt:
            exit()

        assert len(output_grids) == len(codes)
        assert len(results) == len(codes)
        
        for code_idx, output_grid in enumerate(output_grids):
            try:
                comparison_result, ratio = compare_grids(output_grid, io_pair.y)
                if isinstance(output_grid, np.ndarray):
                    return_output_grids[code_idx].append(output_grid.astype(int).tolist())
                else:
                    return_output_grids[code_idx].append(output_grid)
            except:
                breakpoint()
            
            if comparison_result == GridComparisonResult.EQUAL:
                results[code_idx].append((True, ratio))
            elif comparison_result in [GridComparisonResult.SHAPE_MISMATCH, 
                                     GridComparisonResult.CONTENT_MISMATCH]:
                results[code_idx].append((False, ratio))
            else:
                results[code_idx].append((None, 0.0))
    
    return results, return_output_grids



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", help="Path to the answer file")
    parser.add_argument("--data_dir", default="/mnt/nas/suehyun/ARC/dataset/RE-ARC/tasks",
                       help="Path to RE-ARC tasks directory")
    parser.add_argument("--metadata_file", default="/mnt/nas/suehyun/ARC/dataset/RE-ARC/metadata.json")
    parser.add_argument("--results_dir", default="results", help="Directory to save results")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to run")
    parser.add_argument("--n_tasks", type=int, default=100, help="Number of tasks to run")
    parser.add_argument("--easy_instances", action="store_true", help="Run only easy instances")
    parser.add_argument("--timeout", type=int, default=5, help="Timeout for each execution")
    args = parser.parse_args()

    random.seed(0)
    
    # Load problem responses
    with open(args.answer_file) as f:
        problem_answers = [json.loads(line) for line in f]
    
    if args.n_tasks:
        problem_answers = random.sample(problem_answers, args.n_tasks)
        
    with open(args.metadata_file) as f:
        metadata = json.load(f)
        
    uids = set([p["uid"] for p in problem_answers])

    # Load all RE-ARC IO pairs
    all_io_pairs = {}
    files = os.listdir(args.data_dir)
    for file in tqdm(files, desc="Loading IO pairs"):
        if file.endswith('.json'):
            uid = os.path.splitext(file)[0]
            if uid not in uids:
                continue
            file_path = os.path.join(args.data_dir, file)
            with open(file_path) as f:
                data = json.load(f)
                indices = list(range(len(data)))
                if args.n_samples:
                    pso_difficulties = metadata[uid]["pso_difficulties"]
                    indices = sorted(indices, key=lambda i: pso_difficulties[i])  # Sort by difficulty
                    if args.easy_instances:
                        indices = sorted(indices[:args.n_samples])  # Take easy instances
                    else:
                        indices = sorted(random.sample(indices, args.n_samples))  # Randomly sample difficulty
                    data = [data[i] for i in indices] 
                pairs = [ArcIOPair(np.array(item["input"]), np.array(item["output"])) 
                        for item in data]
                all_io_pairs[uid] = (pairs, indices)

    os.makedirs("results", exist_ok=True)
    saving_file = pathlib.Path(args.results_dir) / pathlib.Path(args.answer_file).with_suffix("").name
    saving_file = saving_file.with_name(f"{saving_file.name}_tasks_{'easy' if args.easy_instances else ''}{args.n_tasks}_evidences_{args.n_samples}_exec_results_v4.jsonl")
    print(f"Saving to {saving_file}")

    accepted = 0

    for problem_idx, p in enumerate(tqdm(problem_answers)):
        uid = p["uid"]
        responses = p["responses"]
        print(f"Problem: {uid}")
        
        codes = [parse_code(response)[0] if parse_code(response) else "" 
                for response in responses]

        # Get corresponding IO pairs
        io_pairs, io_pairs_indices = all_io_pairs.get(uid, None)
        if not io_pairs:
            print(f"Warning: No IO pairs found for {uid}")
            continue
        
        verdicts = []
        all_output_grids = []

        if not MULTI_EXECUTE:
            for code in codes:
                code_verdicts = []
                code_ratios = []
                code_outputs = []
                
                for io_pair in io_pairs:
                    try:
                        success, ratio, outputs = validate(io_pair, code, args.timeout)
                        code_verdicts.append(success)
                        code_ratios.append(ratio)
                        code_outputs.extend(outputs)
                    except KeyboardInterrupt:
                        exit()
                    except Exception as e:
                        code_verdicts.append(None)
                        code_ratios.append(0.0)
                        code_outputs.append("error")

                verdicts.append(code_verdicts)
                all_output_grids.append(code_outputs)
        else:
            results, output_grids = multi_validate(io_pairs, codes, args.timeout)
            for idx, (result, outputs) in enumerate(zip(results, output_grids)):
                code_verdicts = [verdict for verdict, _ in result]
                code_ratios = [ratio for _, ratio in result]
                
                verdicts.append(code_verdicts)
                all_output_grids.append(outputs)
                
                num_successes = sum(1 for v in code_verdicts if v)
                avg_ratio = sum(code_ratios) / len(code_ratios)
                print(f"    {idx}: {num_successes}/{len(code_verdicts)} passed "
                      f"avg_ratio={avg_ratio:.2f}")

        # Update problem answers with results
        problem_answers[problem_idx]["re-arc_io_pairs_indices"] = io_pairs_indices
        problem_answers[problem_idx]["verdicts"] = verdicts
        problem_answers[problem_idx]["output_grids"] = all_output_grids

        # Count as accepted if any code passes all IO pairs
        if any(all(v) for v in verdicts):
            accepted += 1

        print(f"Accepted: {accepted}/{problem_idx+1}")
        
        print(f"Saving to {saving_file}")
        with open(saving_file, "w") as f:
            for p in problem_answers[:problem_idx+1]:
                f.write(json.dumps(p) + "\n")

    print(f"Accepted: {accepted}/{len(problem_answers)}")
    
    # print(f"Saving to {saving_file}")
    # with open(saving_file, "w") as f:
    #     f.write("\n".join(json.dumps(p) for p in problem_answers))



if __name__ == "__main__":
    main()