CUDA_VISIBLE_DEVICES="0,1" python vllm_inference.py
CUDA_VISIBLE_DEVICES="2,3" python vllm_inference.py
python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769.jsonl
python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_re_arc_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1106144751388516.jsonl
python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333.jsonl

# temp 0.8
python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1111110349182985.jsonl \
--timeout 8 \
--num_workers 8

python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1111110600563620.jsonl \
--timeout 8 \
--num_workers 8

python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_240_3631a71a_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1111171530373334.jsonl \
--timeout 8 \
--num_workers 8

# temp 1.0
python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_1.0_1111182947795511.jsonl

python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_240_3631a71a_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_1.0_1111201234584929.jsonl

python eval_code_samples.py \
--answer_file finetune/inference/responses/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_1.0_1111182912115165.jsonl

python eval_code_samples_iopairs.py \
--answer_file finetune/inference/responses/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769.jsonl \
--results_dir results/arc_train_240 \
--n_samples 100 \
--n_tasks 50

python eval_code_samples_iopairs.py \
--answer_file finetune/inference/responses/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333.jsonl \
--results_dir results/arc_train_160 \
--n_samples 100 \
--n_tasks 50

python eval_code_samples_iopairs.py \
--answer_file finetune/inference/responses/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769.jsonl \
--results_dir results/arc_train_240 \
--n_samples 30 \
--n_tasks 50

python eval_code_samples_iopairs.py \
--answer_file finetune/inference/responses/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333.jsonl \
--results_dir results/arc_train_160 \
--n_samples 30 \
--n_tasks 50

# python eval_code_samples_iopairs.py \
# --answer_file finetune/inference/responses/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769.jsonl \
# --results_dir results/arc_train_240 \
# --n_samples 100 \
# --n_tasks 50 \
# --easy_instances

python eval_code_samples_iopairs.py \
--answer_file finetune/inference/responses/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333.jsonl \
--results_dir results/arc_train_160 \
--n_samples 100 \
--n_tasks 50 \
--easy_instances

python /mnt/nas/suehyun/ARC/plot_barc_reward_dist.py \
--results_root results/re_arc_240 \
--output_root reward_distributions/ARC_Potpourri_Induction_8B_re_arc_240_64

python /mnt/nas/suehyun/ARC/plot_barc_reward_dist.py \
--results_root results/arc_train_240 \
--output_root reward_distributions/ARC_Potpourri_Induction_8B_arc_train_240_64