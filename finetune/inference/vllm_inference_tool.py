import os
import re
import sys
import time
import json
import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from verification_tools import *
import torch
from arc import train_problems, validation_problems

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--problem_file", type=str, required=True, help="Problem file to use")
    parser.add_argument("--num_of_samples_per_problem", type=int, default=128, help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lora_dir", type=str, default=None, help="LoRA directory to use if applicable")
    parser.add_argument("--max_attempts", type=int, default=6, help="Maximum number of attempts")
    parser.add_argument("--use_tool_role", action="store_true")
    return parser.parse_args()

def load_data(problem_file):
    data = []
    with open(problem_file) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def initialize_tokenizer(model, lora_dir=None):
    if lora_dir:
        return AutoTokenizer.from_pretrained(lora_dir)
    return AutoTokenizer.from_pretrained(model)

def initialize_llm(model, lora_dir, tensor_parallel):
    if lora_dir:
        return LLM(model=model, enable_lora=True, max_lora_rank=256, max_model_len=12000,
                   enable_prefix_caching=True, tensor_parallel_size=tensor_parallel), LoRARequest("barc_adapter", 1, lora_dir)
    return LLM(model=model, enable_lora=False, max_model_len=12000,
               enable_prefix_caching=True, tensor_parallel_size=tensor_parallel), None
    
def parse_code(paragraph):
    """
    This function extracts all Markdown code blocks from a given paragraph.
    Args:
        paragraph (str): The input paragraph containing the Markdown code blocks.
    Returns:
        list: A list of extracted code blocks.
    """
    # Regular expression to match Markdown code blocks
    code_block_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    # Find all code blocks in the paragraph
    matches = code_block_pattern.findall(paragraph)

    # Strip any leading/trailing whitespace from each code block
    code_blocks = [match.strip() for match in matches]

    if code_blocks:
        return code_blocks
    
    # assume that it does not begin with python
    code_block_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    matches = code_block_pattern.findall(paragraph)
    code_blocks = [match.strip() for match in matches]
    return code_blocks


def get_arc_problem(uid):
    for problem in train_problems + validation_problems:
        if problem.uid == uid:
            return problem
    return None

def save_results(saving_file, all_responses):
    with open(saving_file, "w") as f:
        f.write("\n".join(json.dumps(p) for p in all_responses))

def main():
    args = parse_arguments()
    print(f"\nStarting inference with model: {args.model}")
    print(f"LoRA directory: {args.lora_dir if args.lora_dir else 'None'}\n")
    
    data = load_data(args.problem_file)
    print(f"Loaded {len(data)} problems from {args.problem_file}")
    
    tokenizer = initialize_tokenizer(args.model, args.lora_dir)
    tokenizer.model_max_length = 131072
    tensor_parallel = torch.cuda.device_count()
    print(f"Using {tensor_parallel} GPUs")
    llm, lora_request = initialize_llm(args.model, args.lora_dir, tensor_parallel)
    print(f"LLM: {llm}")
    datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S%f")
    saving_file = f"{args.problem_file.replace('.jsonl', '')}_{args.lora_dir.split('/')[-1] if args.lora_dir else args.model.split('/')[-1]}_temp_{args.temperature}_{datetime_str}.jsonl"
    print(f"Saving to {saving_file}")
    time.sleep(5)

    all_responses = []
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=1536, n=1)
    
    for d in tqdm(data, desc="Processing problems"):
        uid = d["uid"]
        problem = get_arc_problem(uid)
        messages = d["messages"]
        
        input_grids = [pair.x for pair in problem.train_pairs]
        output_grids = [pair.y for pair in problem.train_pairs]
        
        if args.use_tool_role:
            tool_code = d.get("tool")
            tools = load_tool(messages, tool_code)
            
        # Process each trajectory sequentially
        trajectories = []
        for idx in tqdm(range(args.num_of_samples_per_problem), desc="Processing trajectories"):
            chat_history = messages[:2].copy()
            trajectory_verdicts = []
            trajectory_outputs = []
            success = False
            
            num_attempts = 0
            for _ in range(args.max_attempts):
                num_attempts += 1
                # Generate initial hypothesis
                output = llm.chat(chat_history, sampling_params=sampling_params, lora_request=lora_request, use_tqdm=False)
                hypothesis = output[0].outputs[0].text.strip()
                print(f"Hypothesis: {hypothesis}")

                chat_history.append({"role": "assistant", "content": hypothesis})
                # if True or TOOL_TOKEN in hypothesis:  # TODO: remove True
                try:
                    code = parse_code(hypothesis)[0]
                    result_str, verdicts, return_output_grids = verify_transformation(code, input_grids, output_grids)
                    print(f"Result: {result_str}")
                    
                    chat_history.append({"role": "user", "content": result_str})
                    trajectory_verdicts.append(verdicts)
                    trajectory_outputs.append(return_output_grids)
                    
                    if all(verdicts):
                        success = True
                        break
                except Exception as e:
                    print(f"Error: {e}")
                    trajectory_verdicts.append(None)
                    trajectory_outputs.append(None)
                    result_str = None
                    break
                
            trajectories.append({
                "success": success,
                "num_attempts": num_attempts,
                "output_messages": chat_history[2:],
                "verdicts": trajectory_verdicts,
                "output_grids": trajectory_outputs,
            })
            print(f"Trajectory: {trajectories[-1]}")

            if success:
                print(f"✓ Success on attempt {num_attempts}")
            elif num_attempts >= args.max_attempts:
                print(f"✗ Failed after {num_attempts} attempts")

        # Save response data
        response_data = {
            "uid": uid,
            "input_messages": messages[:2],
            "base_model": args.model,
            "lora_dir": args.lora_dir,
            "trajectories": trajectories
        }
        all_responses.append(response_data)
        save_results(saving_file, all_responses)

    print(f"\nInference complete. Results saved to {saving_file}")

if __name__ == "__main__":
    main()
