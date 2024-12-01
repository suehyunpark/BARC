parallel --will-cite ::: \
'python heatmap_evidence_pairs.py \
    --results_path results/arc_train_160/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333_tasks_50_evidences_30_exec_results_v4.jsonl \
    --metadata_path /mnt/nas/suehyun/ARC/dataset/RE-ARC/metadata.json \
    --plot_dir results/arc_train_160/tasks_50_evidences_30_h_64_heatmaps \
    --num_problems 30 \
    --num_hypotheses 64' \
\
'python heatmap_evidence_pairs.py \
    --results_path results/arc_train_160/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333_tasks_50_evidences_100_exec_results_v4.jsonl \
    --metadata_path /mnt/nas/suehyun/ARC/dataset/RE-ARC/metadata.json \
    --plot_dir results/arc_train_160/tasks_50_evidences_100_h_64_heatmaps \
    --num_problems 100 \
    --num_hypotheses 64' \
\
'python heatmap_evidence_pairs.py \
    --results_path results/arc_train_240/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769_tasks_50_evidences_30_exec_results_v4.jsonl \
    --metadata_path /mnt/nas/suehyun/ARC/dataset/RE-ARC/metadata.json \
    --plot_dir results/arc_train_240/tasks_50_evidences_30_h_64_heatmaps \
    --num_problems 30 \
    --num_hypotheses 64' \
\
'python heatmap_evidence_pairs.py \
    --results_path results/arc_train_240/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769_tasks_50_evidences_100_exec_results_v4.jsonl \
    --metadata_path /mnt/nas/suehyun/ARC/dataset/RE-ARC/metadata.json \
    --plot_dir results/arc_train_240/tasks_50_evidences_100_h_64_heatmaps \
    --num_problems 100 \
    --num_hypotheses 64' \
\
'python heatmap_train_pairs.py \
    --results_path results/arc_train_160/arc_problems_train_160_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1107191100011333_exec_results_v4.jsonl \
    --plot_dir results/arc_train_160/tasks_50_train_pairs_h_64_heatmaps \
    --num_tasks 50' \
\
'python heatmap_train_pairs.py \
    --results_path results/arc_train_240/arc_problems_train_240_extra_newline_v2_Llama-3.1-ARC-Potpourri-Induction-8B_temp_0.8_1105182739940769_exec_results_v4.jsonl \
    --plot_dir results/arc_train_240/tasks_50_train_pairs_h_64_heatmaps \
    --num_tasks 50'