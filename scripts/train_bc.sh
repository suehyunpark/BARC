python gen_inputs_induction_optimal.py \
--results_dir results \
--samples_per_task 4 \
--output_file finetune/alignment-handbook/arc_inputs_train_bc_optimal-actions_temp-0.8_nsampled-128_max4-per-task.jsonl

python gen_inputs_induction_optimal.py \
--results_dir results \
--samples_per_task 1 \
--output_file finetune/alignment-handbook/arc_inputs_train_bc_optimal-actions_temp-0.8_nsampled-128_max1-per-task.jsonl

# Loading from a JSONL file
python finetune/alignment-handbook/gen_dataset_induction.py \
--load_file finetune/alignment-handbook/arc_inputs_train_bc_optimal-actions_temp-0.8_nsampled-128_max4-per-task_ntasks-182_ninstances-574.jsonl \
--output_huggingface_dataset arc_inputs_train_bc_optimal-actions_max4-per-task

python finetune/alignment-handbook/gen_dataset_induction.py \
--load_file finetune/alignment-handbook/arc_inputs_train_bc_optimal-actions_temp-0.8_nsampled-128_max1-per-task_ntasks-182_ninstances-182.jsonl \
--output_huggingface_dataset arc_inputs_train_bc_optimal-actions_max1-per-task

cd BARC/finetune/alignment-handbook/
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
--num_processes=4 \
scripts/run_sft.py recipes/improve/config_fft_llama3_induction_bc_optimal-action.yaml \
--load_in_4bit=false

python gen_inputs_induction_correction.py \
--results_dir results \
--trajectories_per_task 1 \
--output_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max1-per-task_timeout-8.jsonl

python gen_inputs_induction_correction.py \
--results_dir results \
--trajectories_per_task 1 \
--output_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max1-per-task_warm-start_timeout-8.jsonl

python finetune/alignment-handbook/gen_dataset_induction_correction_tool.py \
--load_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max1-per-task_timeout-8_ntasks-178_ninstances-178.jsonl \
--output_huggingface_dataset arc_inputs_train_bc_trajectories_max1-per-task_tool_token

python gen_inputs_induction_correction.py \
--results_dir results \
--trajectories_per_task 10 \
--output_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max10-per-task_timeout-8.jsonl

python gen_inputs_induction_correction.py \
--results_dir results \
--trajectories_per_task 10 \
--output_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max10-per-task_warm-start_timeout-8.jsonl

python finetune/alignment-handbook/gen_dataset_induction_correction_tool.py \
--load_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max1-per-task_warm-start_timeout-8_ntasks-178_ninstances-178.jsonl \
--output_huggingface_dataset arc_inputs_train_bc_trajectories_max1-per-task_warm-start

python finetune/alignment-handbook/gen_dataset_induction_correction_tool.py \
--load_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max10-per-task_timeout-8_ntasks-178_ninstances-1664.jsonl \
--output_huggingface_dataset arc_inputs_train_bc_trajectories_max10-per-task_tool_token

python finetune/alignment-handbook/gen_dataset_induction_correction_tool.py \
--load_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max10-per-task_timeout-8_ntasks-178_ninstances-1664.jsonl \
--output_huggingface_dataset arc_inputs_train_bc_trajectories_max10-per-task_check_answer

python finetune/alignment-handbook/gen_dataset_induction_correction_tool.py \
--load_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max10-per-task_timeout-8_ntasks-178_ninstances-1664.jsonl \
--output_huggingface_dataset arc_inputs_train_bc_trajectories_max10-per-task

python finetune/alignment-handbook/gen_dataset_induction_correction_tool.py \
--load_file finetune/alignment-handbook/arc_inputs_train_bc_trajectories_temp-0.8_nsampled-128_max10-per-task_warm-start_timeout-8_ntasks-178_ninstances-1564.jsonl \
--output_huggingface_dataset arc_inputs_train_bc_trajectories_max10-per-task_warm-start

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
--num_processes=4 \
scripts/run_sft.py recipes/improve/config_fft_llama3_induction_bc_trajectory.yaml \
--load_in_4bit=false

CUDA_VISIBLE_DEVICES="0,1,2,3" python vllm_inference.py \  # 4 GPUs, 2 hours
--model suehyunpark/potpourri-8b-inst-fft-induction-bc-optimal-action-max1-per-task \
--problem_file problems/arc_problems_train_400_extra_newline_v2.jsonl

python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_400_extra_newline_v2_potpourri-8b-inst-fft-induction-bc-optimal-action-max1-per-task_temp_0.8_1115151531810143.jsonl



CUDA_VISIBLE_DEVICES="2,3" python vllm_inference.py \
--model barc0/Llama-3.1-ARC-Potpourri-Induction-8B \
--problem_file problems/arc_problems_train_400_extra_newline_v2.jsonl

CUDA_VISIBLE_DEVICES="0,1" python vllm_inference_hf.py \
--model barc0/Llama-3.1-ARC-Potpourri-Induction-8B \
--problem_file barc0/induction_heavy_100k_jsonl \
--split train_sft \
--num_of_samples_per_problem 16 \
--temperature 0.8

CUDA_VISIBLE_DEVICES="2,3" python vllm_inference_tool.py \
--model suehyunpark/potpourri-8b-inst-fft-induction-bc-trajectory-max1-per-task-tool-token-fix \
--problem_file problems/arc_problems_train_400_extra_newline_v2.jsonl \
--num_of_samples_per_problem 1 \
--temperature 0.8

CUDA_VISIBLE_DEVICES="0,1" python vllm_inference_tool.py \
--model suehyunpark/potpourri-8b-inst-fft-induction-bc-trajectory-max10-per-task-tool-token-fix \
--problem_file problems/arc_problems_train_400_extra_newline_v2.jsonl \
--num_of_samples_per_problem 1 \
--temperature 0.8

CUDA_VISIBLE_DEVICES="4,5,6,7" python vllm_inference_tool_batched.py \
--model suehyunpark/potpourri-8b-inst-fft-induction-bc-trajectory-max10-per-task-warm-start \
--problem_file problems/arc_problems_train_400_extra_newline_v2.jsonl \
--num_of_samples_per_problem 1 \
--temperature 0.8

CUDA_VISIBLE_DEVICES="0,1,2,3" python vllm_inference_tool_batched.py \
--model suehyunpark/potpourri-8b-inst-fft-induction-bc-trajectory-max1-per-task-tool-token-fix \
--problem_file problems/arc_problems_train_400_extra_newline_v2.jsonl \
--num_of_samples_per_problem 1 \
--temperature 0.8

CUDA_VISIBLE_DEVICES="4,5,6,7" python vllm_inference_tool_batched.py \
--model suehyunpark/potpourri-8b-inst-fft-induction-bc-trajectory-max1-per-task-warm-start \
--problem_file problems/arc_problems_train_400_extra_newline_v2.jsonl \
--num_of_samples_per_problem 1 \
--temperature 0.8



export LD_LIBRARY_PATH="${HOME}/.conda/envs/barc/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"
