mlx worker launch -- python3 ../src/inference.py \
    --llm_model_name Qwen \
    --llm_model_path ../../../model/Qwen2-7B-Instruct \
    --peft_type lora \
    --merge_save_path ../out/p-tuning_peft/lora_model_end/ \
    --use_merge_model True \
    --log_path ../out/lora_output.log \
