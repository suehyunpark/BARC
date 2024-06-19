import argparse
import random
from execution import execute_transformation, execute_input_generator
import numpy as np
import os
import re
import tqdm
import json

class Problem:
    def __init__(self, source_code):
        self.source = source_code
        self.examples = []

    def add_example(self, input_grid, output_grid):
        self.examples.append((input_grid, output_grid))
    def to_dict(self):
        return {
            "source": self.source,
            "examples": [(input_grid.tolist(), output_grid.tolist()) for input_grid, output_grid in self.examples]
        }


def check_grid(grid):
    # check the grid is well-formed, 2d numpy array of integers between 0-9
    try:
        assert isinstance(grid, np.ndarray)
        assert len(grid.shape) == 2
        assert np.all((0 <= grid) & (grid <= 9))
    except AssertionError:
        return False
    return True

def check_grids_all_equal(grids):
    """check a set of grids are all equal"""
    assert len(grids) > 0
    return all(np.array_equal(grids[0], grid) for grid in grids)

def check_diversity(grids, threshold):
    """check a set of grids is diverse, i.e. the grids should be sufficiently different from each other"""
    # TODO
    pass

def check_identity(input_grid, output_grid):
    try:
        assert check_grid(input_grid)
        assert check_grid(output_grid)
    except:
        breakpoint()
    return np.array_equal(input_grid, output_grid)

def generate_input_grids(problem_source, num_returns=3, timeout=1, function_name="generate_input", retries=50, deduplicate=True):
    """
    given a problem source code, generate an input grid
    """
    input_grids = []
    tries = 0
    while len(input_grids) < num_returns and tries < retries:
        input_grid = execute_input_generator(problem_source, timeout, function_name)
        if not check_grid(input_grid):
            tries += 1
            continue
        if deduplicate and any(np.array_equal(input_grid, existing_grid) for existing_grid in input_grids):
            tries += 1
            continue
        input_grids.append(input_grid)

    return input_grids

def get_random_color_mapping(only_non_black=True):
    """
    Get a random color mapping from 0-9 to 0-9 where 0 is black.
    If only_non_black is True, map 1-9 to 1-9.
    """
    if only_non_black:
        colors = list(range(1, 10))
        random.shuffle(colors)
        color_mapping = {i: colors[i-1] for i in range(1, 10)}
        color_mapping[0] = 0
    else:
        colors = list(range(10))
        random.shuffle(colors)
        color_mapping = {i: colors[i] for i in range(10)}
    
    return color_mapping

def apply_color_mapping(grid, color_mapping):
    """
    Apply a color mapping to the grid
    """
    return np.vectorize(color_mapping.get)(grid)

def run_transformation(source, input_grid, timeout=1, function_name="main", num_returns=50):
    """
    run the transformation on the input grid and return the output grid multiple times
    """
    output_grids = []
    for _ in range(num_returns):
        output_grid = execute_transformation(source, input_grid, timeout, function_name)
        output_grids.append(output_grid)
    return output_grids

def generate_problem(problem_source, num_examples=4, num_input_grids=10, num_deterministic_check=10, num_color_permute_check=10, timeout=1):
    """
    Generate a problem by generating input grids and running the transformation on them.
    Return None for the problem if:
    1. non-deterministic transformations
    2. non color-invariant transformations

    For the example input-output grid pair, remove for the example pair if:
    1. input grid is the same as the output grid
    """
    problem = Problem(problem_source)

    input_grids = generate_input_grids(problem_source, num_returns=num_input_grids, timeout=timeout, deduplicate=True)
    
    # Check for non-deterministic transformations
    for input_grid in input_grids:
        output_grids = run_transformation(problem_source, input_grid, timeout=timeout, num_returns=num_deterministic_check)
        if not all(check_grid(output_grid) for output_grid in output_grids):
            # if any of the output grids are not well-formed, skip this particular input grid
            print('Non well-formed output grid, skipping this input grid')
            continue
        if not check_grids_all_equal(output_grids):
            print("Non-deterministic transformation")
            return None

        # Check for non-color-invariant transformations
        for _ in range(num_color_permute_check):
            color_mapping = get_random_color_mapping(only_non_black=True)
            permuted_input_grid = apply_color_mapping(input_grid, color_mapping)
            original_output_grid = execute_transformation(problem_source, input_grid, timeout, function_name="main")
            permuted_output_grid = execute_transformation(problem_source, permuted_input_grid, timeout, function_name="main")
            if not check_grid(permuted_output_grid) or not check_grid(original_output_grid) or not check_grid(input_grid):
                print("Permute check failed due to non-well-formed grids")
                return None
            if not check_identity(apply_color_mapping(original_output_grid, color_mapping), permuted_output_grid):
                print("Permute check failed")
                return None

        output_grid = execute_transformation(problem_source, input_grid, timeout, function_name="main")
        if check_identity(input_grid, output_grid):
            print("Identity transformation, skipping this example")
            continue
        problem.add_example(input_grid, output_grid)

    return problem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, help="Path to jsonl file containing program generation")
    parser.add_argument("--problem_source_uid", type=str, help="Problem id of a seed problem to validate")
    parser.add_argument("--run_all_seed", action="store_true", help="Run all seed problems")
    args = parser.parse_args()

    # only one of problem_uid or run_all_seed or jsonl should be provided
    assert sum([args.problem_source_uid is not None, args.run_all_seed, args.jsonl is not None]) == 1, "Provide one of problem_uid, run_all_seed or jsonl"

    problems_source = []
    problem_source_uids = []
    if args.problem_source_uid:
        problem_source_uids = [args.problem_source_uid]
    elif args.run_all_seed:
        seeds = os.listdir("seeds")
        # filter files with .py extension and 8 hex value characters in the file name
        pattern = r"([0-9a-f]{8})\.py"
        problem_source_uids = [re.match(pattern, filename).group(1) for filename in seeds if re.match(pattern, filename)]
        # Now `matched_files` contains all the filenames that match the pattern
    elif args.jsonl:
        with open(args.jsonl) as f:
            data = f.readlines()
        for line in data:
            problem = json.loads(line)
            problems_source.append(problem["source"])
    else:
        raise ValueError("Provide one of problem_uid, run_all_seed or jsonl")
    
    if problem_source_uids:
        for problem_source_uid in problem_source_uids:
            with open(f"seeds/{problem_source_uid}.py") as f:
                source = f.read()
            problems_source.append(source)


    problems = []
    for i, problem_source in tqdm.tqdm(enumerate(problems_source)):
        problem = generate_problem(problem_source)
        if problem and len(problem.examples) >= 4:
            problems.append(problem)
        else:
            if problem_source_uids:
                print(f"Problem {problem_source_uids[i]} is not valid")

    # write list of Problem to jsonl file
    with open("generated_problems.jsonl", "w") as f:
        for problem in problems:
            f.write(json.dumps(problem.to_dict()) + "\n")
 

if __name__ == "__main__":
    main()