import os
import random
from typing import List
import tqdm
import json
from gen_dataset_both import IOPair, Problem, make_input_prompt_induction, convert_chat_format_induction
from arc.types import ArcProblem
import re

from gen_dataset_both import TRANSPOSE, EXTRA_NEWLINE

from arc.read import parse_group, parse_dir
import os

def parse_dir_jsonl(d) -> List[ArcProblem]:
    random.seed(0)
    cases = []
    for file in os.listdir(d):
        path = os.path.join(d, file)
        data = []
        with open(path) as f:
            for line in f:
                parsed = json.loads(line)
                # uid = parsed["name"].replace(".jsonl", "")
                id = parsed["id"]
                if "orig" in id:
                    continue
                data.append(parsed)
        
        samples = random.sample(data, 1) 
        for parsed in samples:
            id = parsed["id"]
            train = parsed["train"]
            demo_pairs = parse_group(train)

            test = parsed["test"]
            test_pairs = parse_group(test)
            cases.append(
                ArcProblem(
                    train_pairs=demo_pairs,
                    test_pairs=test_pairs,
                    uid=id,
                )
            )
    return cases

def get_re_arc_problems():
    problems = []
    problem_dir = "/mnt/nas/suehyun/ARC/dataset/ARC_RE-ARC_interleaved/seed-42_sample-10"
    problems.extend(parse_dir_jsonl(problem_dir))
    
    return problems


re_arc_problems = get_re_arc_problems()
# assert every uid is unique
uids = [p.uid for p in re_arc_problems]
assert len(uids) == len(set(uids)), f"{len(uids)} != {len(set(uids))}"

VERSION = "v2"

ALL_PROBLEMS = []

def main():

    # get problems under the seed directory
    seeds = os.listdir("../../seeds")
    # filter files with .py extension and 8 hex value characters in the file name
    pattern = r"[0-9a-f]{8}(_[a-zA-Z]+)?\.py"
    # get all files and its content
    seeds = [seed for seed in seeds if re.match(pattern, seed)]
    def extract_uid(seed):
        uid = ""
        if "." in seed:
            uid = seed.split(".")[0]
        if "_" in uid:
            uid = uid.split("_")[0]
        return uid
    seeds_uid = [extract_uid(seed) for seed in seeds]
    print(len(seeds))
    print(seeds_uid)


    import random
    random.seed(0)
    problems = re_arc_problems
            
    # save all problems

    seed_uid_hit = set()
    for arc_problem in tqdm.tqdm(problems):
        
        uid = arc_problem.uid
        uid_ = uid.split('_')[0]
        if uid_ in seeds_uid:
            seed_uid_hit.add(uid_)
            continue
        
        train_pairs = []
        for pair in arc_problem.train_pairs:
            train_pairs.append(IOPair(pair.x.T, pair.y.T))
        test_pairs = []
        for pair in arc_problem.test_pairs:
            test_pairs.append(IOPair(pair.x.T, pair.y.T))
            
        problem = Problem(train_pairs=train_pairs, test_pairs=test_pairs, code="# No code")
        question = make_input_prompt_induction(problem, transpose=TRANSPOSE)
    
#         answer = f"""Let's solve this puzzle using Python code with the common library functions. We'll first reason about the problem and then write the code to solve it. The `transform` function will take the input grid and return the output grid. Here is the Python code with the comments describing how to solve the problem:
# ```python
# """
        messages = convert_chat_format_induction(question, None)['messages']
        ALL_PROBLEMS.append({"uid": uid, "messages": messages})

    # breakpoint()
        

    # print([p['uid'] for p in ALL_PROBLEMS[0:50]])
    # print(f"The number of problems is {len(ALL_PROBLEMS)}")

    # breakpoint()
    
    split_filename = "re_arc"
    problem_file = f"arc_problems_{split_filename}_{len(ALL_PROBLEMS)}.jsonl"
    if TRANSPOSE:
        problem_file = f"arc_problems_{split_filename}_{len(ALL_PROBLEMS)}_transpose.jsonl"
    if EXTRA_NEWLINE:
        problem_file = f"arc_problems_{split_filename}_{len(ALL_PROBLEMS)}_extra_newline.jsonl"
    if TRANSPOSE and EXTRA_NEWLINE:
        problem_file = f"arc_problems_{split_filename}_{len(ALL_PROBLEMS)}_transpose_extra_newline.jsonl"

    if VERSION:
        problem_file = problem_file.replace(".jsonl", f"_{VERSION}.jsonl")
    
    print(f"Saving to {problem_file}")
    with open(problem_file, "w") as f:
        f.write("\n".join(json.dumps(p) for p in ALL_PROBLEMS))

    # with open(saving_file, "w") as f:
    #     f.write("\n".join(json.dumps(p) for p in all_problem_answers))

if __name__ == "__main__":
    main()
