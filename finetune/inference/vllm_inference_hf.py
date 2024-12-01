import os
import sys
import time
import json
import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--problem_file", type=str, required=True, help="Problem file to use")
    parser.add_argument("--split", type=str, default="train_sft", help="Split to use if loading from HuggingFace")
    parser.add_argument("--num_of_samples_per_problem", type=int, default=128, help="Number of samples per problem")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--batch_size", type=int, default=16, help="Base batch size")
    parser.add_argument("--lora_dir", type=str, default=None, help="LoRA directory to use if applicable")
    parser.add_argument("--max_tokens", type=int, default=1536, help="Maximum tokens for generation")
    return parser.parse_args()

def load_data(problem_file):
    data = []
    with open(problem_file) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_responses(saving_file, all_responses):
    with open(saving_file, "w") as f:
        f.write("\n".join(json.dumps(p) for p in all_responses))

def main():
    args = parse_arguments()
    
    # Initialize tensor parallel size
    tensor_parallel = torch.cuda.device_count()
    print(f"Using {tensor_parallel} GPUs")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir if args.lora_dir else args.model)

    # Load problem data
    if os.path.exists(args.problem_file):
        # Load from local file
        data = load_data(args.problem_file)
    else:
        # Load from HuggingFace datasets
        from datasets import load_dataset
        dataset = load_dataset(args.problem_file, split=args.split)
        data = dataset.to_pandas().to_dict('records')

    # Initialize LLM
    if args.lora_dir:
        llm = LLM(model=args.model, 
                  enable_lora=True, 
                  max_lora_rank=256, 
                  max_model_len=12000,
                  enable_prefix_caching=True, 
                  tensor_parallel_size=tensor_parallel)
        lora_request = LoRARequest("barc_adapter", 1, args.lora_dir)
    else:
        llm = LLM(model=args.model, 
                  enable_lora=False, 
                  max_model_len=12000,
                  enable_prefix_caching=True, 
                  tensor_parallel_size=tensor_parallel)
        lora_request = None

    # Setup saving file
    datetime_str = datetime.datetime.now().strftime("%m%d%H%M%S%f")
    model_identifier = args.lora_dir.split('/')[-1] if args.lora_dir else args.model.split('/')[-1]
    saving_file = f"{args.problem_file.split('/')[-1].replace('.jsonl', '')}_{model_identifier}_temp_{args.temperature}_{datetime_str}.jsonl"
    print(f"Saving to {saving_file}")
    time.sleep(5)

    # Process all problems
    all_responses = []
    for i, d in tqdm(enumerate(data), total=len(data), desc="Processing problems"):
        messages = d["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # Prepare inputs
        inputs = tokenizer.apply_chat_template([
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": messages[1]["content"]}
        ], tokenize=False, add_generation_prompt=True)
        
        input_tokens = tokenizer.apply_chat_template([
            {"role": "system", "content": messages[0]["content"]},
            {"role": "user", "content": messages[1]["content"]}
        ], tokenize=True, add_generation_prompt=True)
        print(f"Number of tokens: {len(input_tokens)}")

        # Determine batch size based on input length
        assert args.num_of_samples_per_problem % args.batch_size == 0
        if len(input_tokens) < 1750:
            tmp_batch_size = args.batch_size * 4
        elif len(input_tokens) < 4000:
            tmp_batch_size = args.batch_size * 4
        elif len(input_tokens) < 5000:
            tmp_batch_size = args.batch_size
        else:
            tmp_batch_size = args.batch_size

        print(f"batch size: {tmp_batch_size}")

        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n=tmp_batch_size
        )

        # Generate outputs
        aggregate_outputs = []
        if args.num_of_samples_per_problem >= tmp_batch_size:
            for i in range(args.num_of_samples_per_problem // tmp_batch_size):
                outputs = llm.generate(
                    inputs,
                    sampling_params,
                    lora_request=lora_request
                )
                aggregate_outputs.append(outputs)
        else:
            outputs = llm.generate(
                inputs,
                sampling_params,
                lora_request=lora_request
            )
            aggregate_outputs.append(outputs)

        if not aggregate_outputs:
            breakpoint()

        # Process outputs
        responses = []
        for outputs in aggregate_outputs:
            for output in outputs:
                for i in range(len(output.outputs)):
                    generated_text = output.outputs[i].text
                    responses.append(generated_text)
        print(responses[0])
        # Save responses
        dataset_name = args.problem_file.split("/")[-1].split("jsonl")[0]
        uid = d.get("uid", f"{dataset_name}_{i}")
        response_data = {
            "uid": uid,
            "prompt": inputs,
            "responses": responses,
            "base_model": args.model,
            "lora_dir": args.lora_dir
        }
        all_responses.append(response_data)
        save_responses(saving_file, all_responses)

    print(f"Saving to {saving_file}")
    time.sleep(15)

if __name__ == "__main__":
    main()