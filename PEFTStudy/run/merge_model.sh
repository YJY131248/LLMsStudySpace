mlx worker launch -- python3 ../src/merge_model.py \
    --peft_type p-tuning \
    --llm_model_name Qwen \
    --llm_model_path ../../../model/Qwen2-7B-Instruct \
    --peft_checkpoint_path ../out/p-tuning_peft/checkpoint-30000/ \
    --merge_save_path ../out/merge_model/p-tuning/ \
    --log_path ../out/p-tuning_output.log \
    
