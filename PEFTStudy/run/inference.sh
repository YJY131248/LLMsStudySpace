mlx worker launch -- python3 ../src/inference.py \
    --llm_model_name Qwen \
    --llm_model_path ../../../model/Qwen2-7B-Instruct \
    --merge_save_path: ../out/merge_model/lora/ \
    --use_merge_model True \
    --log_path ../out/p-tuning_output.log \
