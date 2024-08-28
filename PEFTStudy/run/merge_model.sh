mlx worker launch -- python3 ../src/merge_model.py \
    --peft_type lora \
    --llm_model_name Qwen \
    --llm_model_path ../../../model/Qwen2-7B-Instruct \
    --log_path ../out/p-tuning_output.log \
    --peft_checkpoint_path ../out/lora_peft/checkpoint-500/ \
    --merge_save_path: ../out/merge_model/lora/ \
    
