python3 ../src/lora_merge.py \
    --peft_type lora \
    --llm_model_name Qwen \
    --llm_model_path ../model/qwen2.5-7b-instruct \
    --peft_checkpoint_path ../out/car_qwen_lora_model/checkpoint-645/ \
    --merge_save_path ../model/car_lora_model \
    --log_path ../log/car_qwen_lora_model_merge.log
    