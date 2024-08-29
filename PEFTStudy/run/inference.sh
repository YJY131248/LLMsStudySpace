mlx worker launch -- python3 ../src/inference.py \
    --llm_model_name Qwen \
    --llm_model_path ../../../model/Qwen2-7B-Instruct \
    --peft_type p-tuning \
    --merge_save_path ../out/p-tuning_peft/p-tuning_model_end/ \
    --use_merge_model True \
    --log_path ../out/p-tuning_output.log \
