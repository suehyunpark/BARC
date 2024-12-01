from typing import List, Optional
from arc import train_problems, validation_problems
import os
import re
import json
import numpy as np
import tiktoken
from datasets import Dataset
from tqdm import tqdm
import random

VERSION = "0.3"
EXTRA_NEWLINE = "\n"
TRANSPOSE = False

COLOR_MAPPING = {
    0: "Black",
    1: "Blue",
    2: "Red",
    3: "Green",
    4: "Yellow",
    5: "Grey",
    6: "Pink",
    7: "Orange",
    8: "Teal",
    9: "Maroon"
}

COLOR_REPLACEMENTS = {
    "Grey": "Gray",
    "Teal": "Purple",
    "Maroon": "Brown",
}

# Fix Color Mapping
for k, v in COLOR_MAPPING.items():
    if v in COLOR_REPLACEMENTS:
        COLOR_MAPPING[k] = COLOR_REPLACEMENTS[v]


# Map a hard coded color to a deterministic some other color in source code, keeping cases same
def color_deterministic(problem_source_code, old_color, new_color):
    upper_template = f"(((?<=[^a-zA-Z])|^)({old_color.upper()})(?=[^a-zA-Z]|$))"
    capitalized_template = (
        f"(((?<=[^a-zA-Z])|^)({old_color.lower().capitalize()})(?=[^a-zA-Z]|$))"
    )
    lower_template = f"(((?<=[^a-zA-Z])|^)({old_color.lower()})(?=[^a-zA-Z]|$))"

    # Do findall operation with this regex
    upper_regex = re.compile(upper_template)
    capitalized_regex = re.compile(capitalized_template)
    lower_regex = re.compile(lower_template)

    replace_upper = re.sub(
        upper_regex, lambda x: new_color.upper(), problem_source_code
    )

    replace_capitalized = re.sub(
        capitalized_regex,
        lambda x: new_color.lower().capitalize(),
        replace_upper,
    )

    replace_lower = re.sub(
        lower_regex,
        lambda x: new_color.lower(),
        replace_capitalized,
    )

    return replace_lower


def test_color_deterministic():
    problem_source_code = "teal, Teal, TEAL"
    ret = color_deterministic(problem_source_code, "teal", "purple")
    print(ret)


def convert_color_name(text, mapping):
    for old_color, new_color in mapping.items():
        text = color_deterministic(text, old_color, new_color)
    return text

def test_convert_color_name():
    text = "teal, Teal, TEAL\nMaroon COLOR>MAROON, maroon"
    ret = convert_color_name(text, COLOR_REPLACEMENTS)
    print(ret)


class IOPair:
    x: np.ndarray
    y: np.ndarray
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # check type
        assert isinstance(self.x, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        # check shape
        assert len(self.x.shape) == 2
        assert len(self.y.shape) == 2

class Problem:
    # typing hint for the members
    filename: str
    seed_id: str
    code: str
    train_pairs: list
    test_pairs: list

    def __init__(self, filename=None, code=None, seed_id=None, train_pairs=None, test_pairs=None):
        self.filename = filename
        self.seed_id = None
        if filename:
            self.seed_id = filename.split(".")[0]
            if "_" in self.seed_id:
                self.seed_id= self.seed_id.split("_")[0]
        if seed_id:
            self.seed_id = seed_id
        if self.seed_id:
            pattern = r"[0-9a-f]{8}"
            assert re.match(pattern, self.seed_id)
            self.load_arc_problem(self.seed_id)

        self.code = code
        if train_pairs:
            self.train_pairs = train_pairs
        if test_pairs:
            self.test_pairs = test_pairs

        assert self.code, "Code is not provided"
        assert self.train_pairs, "Train pairs are not provided"
        assert self.test_pairs, "Test pairs are not provided"
        # check type
        assert isinstance(self.train_pairs, list)
        assert isinstance(self.test_pairs, list)
        assert all(isinstance(pair, IOPair) for pair in self.train_pairs)
        assert all(isinstance(pair, IOPair) for pair in self.test_pairs)


    def load_arc_problem(self, seed_id):
        # using train_problems
        arc_problem = None
        for problem in train_problems + validation_problems:
            if problem.uid == seed_id:
                arc_problem = problem
                break
        assert arc_problem is not None
        self.train_pairs = []
        for pair in arc_problem.train_pairs:
            self.train_pairs.append(IOPair(pair.x.T, pair.y.T))
        self.test_pairs = []
        for pair in arc_problem.test_pairs:
            self.test_pairs.append(IOPair(pair.x.T, pair.y.T))
            
            
def format_example(example: List[List[int]]) -> Optional[IOPair]:
    input_grid = np.array(example[0])
    output_grid = np.array(example[1])
    if (input_grid.shape[0] > 30 or input_grid.shape[1] > 30 
        or output_grid.shape[0] > 30 or output_grid.shape[1] > 30):
        return None
    return IOPair(input_grid, output_grid)

def grid_to_input(grid, transpose: bool):
    if transpose:
        transformed_grid = grid.T
    else:
        transformed_grid = grid
    return "\n".join(" ".join(COLOR_MAPPING[c] for c in row) for row in transformed_grid) + EXTRA_NEWLINE

def make_problem_input_str(problem: Problem, transpose: bool):
    prompt = "Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines."
    prompt += "\nHere are the input and output grids for the reference examples:\n"
    for i, pair in enumerate(problem.train_pairs):
        prompt += f"Example {i+1}\n"
        prompt += f"Input:\n{grid_to_input(pair.x, transpose)}\nOutput:\n{grid_to_input(pair.y, transpose)}\n\n" 
    prompt += "Here is (are) the input grid(s) for the test example(s):\n"
    prompt += "Input(s):\n" + "\n".join(grid_to_input(pair.x, transpose) for pair in problem.test_pairs)
    return prompt

def make_input_prompt_induction(problem: Problem, transpose: bool):
    question = make_problem_input_str(problem, transpose=transpose)
    question += "\nWrite a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples."
    return question

DEFAULT_SYSTEM_PROMPT = "You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."

def convert_chat_format(question, answer):
    messages = {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
    }
    if answer:
        messages["messages"].append({"role": "assistant", "content": answer})
    return messages

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_file", type=str)
    parser.add_argument("--load_huggingface_dataset", type=str)
    parser.add_argument("--output_huggingface_dataset", type=str, required=False, default=None)
    args = parser.parse_args()



    # common_lib, _ = get_common_lib_from_file("seeds/common.py")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    if args.load_file or args.load_huggingface_dataset:
        assert args.output_huggingface_dataset , "output_huggingface_dataset is required"
        output_huggingface_dataset = args.output_huggingface_dataset.strip("/")
    elif args.load_huggingface_dataset:
        output_huggingface_dataset = args.load_huggingface_dataset.strip("/") + "_messages_format" + "_" + VERSION
        print(f"output_huggingface_dataset: {output_huggingface_dataset}")
    loaded_problems = []
    if args.load_file or args.load_huggingface_dataset:
        if args.load_file:
            assert args.load_file.endswith(".jsonl"), "Expected a jsonl file"
            assert os.path.exists(args.load_file), "File does not exist"
            loaded_data = []
            with open(args.load_file) as f:
                for line in f:
                    loaded_data.append(json.loads(line))
        else:
            # load from huggingface dataset
            import datasets
            print(datasets.load_dataset(args.load_huggingface_dataset))
            loaded_data = datasets.load_dataset(args.load_huggingface_dataset)['train']

        print(f"get {len(loaded_data)} problems from file")
        print(loaded_data)

        for d in tqdm(loaded_data):
            code = d['source']
            train_pairs = []
            for example in d["train_examples"]:
                pair = format_example(example)
                if pair is None:
                    continue
                train_pairs.append(pair)

            test_pairs = []
            for example in d["test_examples"]:
                pair = format_example(example)
                if pair is None:
                    continue
                test_pairs.append(pair)
        
            problem = Problem(code=code, train_pairs=train_pairs, test_pairs=test_pairs)
            loaded_problems.append(problem)

    print(f"get {len(loaded_problems)} problems from file")
    
    train_data = []
    problems = loaded_problems
    for problem in problems:
        question = make_input_prompt_induction(problem, transpose=TRANSPOSE)
        answer = f"""Let's solve this puzzle using Python code with the common library functions. We'll first reason about the problem and then write the code to solve it. The `transform` function will take the input grid and return the output grid. Here is the Python code with the comments describing how to solve the problem:
```python
{problem.code}
```
""" 
        answer = convert_color_name(answer, COLOR_REPLACEMENTS)
        train_data.append(convert_chat_format(question, answer))

    # Filter by token count
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    filtered_train_data = []
    token_counts = []
    
    for data in train_data:
        token_count = sum(len(tokenizer.encode(msg["content"])) for msg in data["messages"])
        if token_count < 8000:
            filtered_train_data.append(data)
            token_counts.append(token_count)

    print(f"Total number of tokens: {sum(token_counts)}")
    print(f"Average number of tokens per example: {sum(token_counts) / len(token_counts)}")
    print(f"Max number of tokens per example: {max(token_counts)}")
    print(f"Original examples: {len(train_data)}, Filtered examples: {len(filtered_train_data)}")

    # Push to HuggingFace
    random.seed(0)
    random.shuffle(filtered_train_data)
    
    split_index = int(0.95 * len(filtered_train_data))
    train_split = filtered_train_data[:split_index]
    test_split = filtered_train_data[split_index:]
    
    sample_file = os.path.join(args.load_file.rsplit("/", 1)[0], f"sample_training_instance_{args.output_huggingface_dataset.split('/')[-1]}.json")
    with open(sample_file, 'w') as f:
        json.dump(train_split[0], f, indent=2)
        
    from datasets import DatasetDict
    dataset_dict = DatasetDict({
        "train_sft": Dataset.from_list(train_split),
        "test_sft": Dataset.from_list(test_split)
    })

    dataset_name = f'suehyunpark/induction_{args.output_huggingface_dataset.split("/")[-1]}'
    
    # Create README content with dataset information
    readme_content = f"""# ARC Induction Dataset

This dataset contains examples for training models on ARC (Abstraction and Reasoning Corpus) puzzle solving through code induction.
## Input data file
{args.load_file}

## Dataset Statistics
- Total examples: {len(filtered_train_data)}
- Train split: {len(train_split)} examples
- Test split: {len(test_split)} examples
- Average tokens per example: {sum(token_counts) / len(token_counts):.1f}
- Max tokens per example: {max(token_counts)}

## Format
Each example contains:
- Input: Description of an ARC puzzle with train/test examples
- Output: Python code solution with explanations
"""

    # Push dataset with README
    dataset_dict.push_to_hub(
        dataset_name,
        private=True,
        commit_description=readme_content
    )

if __name__ == "__main__":
    main()