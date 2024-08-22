mlx worker launch -- python3 finetune.py \
    --peft_type lora \
    --llm_model_name Qwen \
    --llm_model_path ../../../model/Qwen2-7B-Instruct \
    --dataset_path ../data/alpaca_gpt4_data_zh.json \
    --log_path ../out/lora_output.log \
    --max_length 256 \
    --lora_rank 8 \
    --output_dir ../out/lora_peft \
    --per_device_train_batch_size 1 \   
    --per_device_eval_batch_size 1 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --max_steps 2400 \
    --save_steps 240 \
    --save_total_limit 10 \
    --logging_steps 10 \
    --gradient_accumulation_steps 16 \