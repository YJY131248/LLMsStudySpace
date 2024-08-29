import torch
import logging
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq,
    Trainer, 
    TrainingArguments,
    HfArgumentParser
)
from peft import (
    LoraConfig, 
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit, 
    PromptEncoderReparameterizationType,
    TaskType,
    get_peft_model
)
from dataclasses import dataclass, field
from dataset_load import get_alpaca_dataset, get_tokenizer_dataset


# 设置模型微调的参数类
@dataclass
class FinetuneArguments:
    peft_type: str = field(default="lora")
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../../../model/Qwen2-7B-Instruct")
    dataset_path: str = field(default="../data/alpaca_gpt4_data_zh.json")
    log_path: str = field(default="../log/lora_output.log")
    max_length: int = field(default=256)
    lora_rank: int = field(default=8)


# 加载LLMs model/tokenizer
def get_base_llm_model_tokenizer(finetune_args):
    # 读取模型类型
    llm_model_name = finetune_args.llm_model_name
    llm_model_path = finetune_args.llm_model_path

    # 加载llm_model
    if llm_model_name == "Qwen" or llm_model_name == "BaiChuan":
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_path, 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.half
        )
    elif llm_model_name == "ChatGLM":
        model = AutoModel.from_pretrained(
            llm_model_path, 
            low_cpu_mem_usage=True, 
            torch_dtype=torch.half
        )
    # 模型不是为本项目支持的模型
    else:
        logger.error("错误参数：底座模型必须是Qwen/ChatGLM/BaiChuan")
        raise ValueError("错误参数：底座模型必须是Qwen/ChatGLM/BaiChuan")

    # 配置模型
    if finetune_args.peft_type != "prefix-tuning":
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    return model, tokenizer


# 根据peft类型返回相应的config
def get_peft_config(finetune_args, tokenizer):
    # 读取peft类型
    peft_type = finetune_args.peft_type
    if peft_type == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    elif peft_type == "p-tuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=10,
            encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
            encoder_hidden_size=1024
        )
    elif peft_type == "prefix-tuning":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=10,
            prefix_projection=True
        )
    elif peft_type == "prompt-tuning":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text = "你是百科全书智能问答机器人。",
            num_virtual_tokens = len(tokenizer("你是百科全书智能问答机器人。")["input_ids"]),
            tokenizer_name_or_path = finetune_args.llm_model_path
        )
    else:
        logger.error("错误参数：peft类型必须为lora/p-tuning/prefix-tuning/prompt-tuning")
        raise ValueError("错误参数：peft类型必须为lora/p-tuning/prefix-tuning/prompt-tuning")

    return peft_config


# 微调函数
def finetune_train(model, peft_config, tokenizer, dataset, train_args):
    model = get_peft_model(model=model, peft_config=peft_config)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()


def main():
    # 忽略警告
    warnings.filterwarnings("ignore")

    # 加载命令行参数
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # 设置logger
    logging.basicConfig(
        level=logging.DEBUG,
        filename=finetune_args.log_path,  #保存日志的本地目录
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    # 将logger声明为全局变量
    global logger
    logger = logging.getLogger(__name__)
    logger.debug("命令行参数")
    logger.debug("finetune_args:")
    logger.debug(finetune_args.__repr__())
    logger.debug("training_args:")
    logger.debug(training_args.__repr__())

    # 加载模型
    llm_model, llm_tokenizer = get_base_llm_model_tokenizer(finetune_args)
    logger.info('Base LLMs {} load successfully! LLM path::: {}'.format(finetune_args.llm_model_name, finetune_args.llm_model_path))

    # 获取peft_config参数
    peft_config = get_peft_config(finetune_args, llm_tokenizer)
    logger.info('Peft {} config load successfully!'.format(finetune_args.peft_type))

    # 加载数据
    dataset = get_alpaca_dataset(finetune_args.dataset_path, test_size=0.1)
    logger.info('dataset build successfully!')
    tokenizer_dataset = get_tokenizer_dataset(dataset, llm_tokenizer, max_length=finetune_args.max_length)
    logger.info('tokenizer dataset build successfully!')

    # 开始训练
    logger.info('Train start!')
    finetune_train(model=llm_model, peft_config=peft_config, tokenizer=llm_tokenizer, dataset=tokenizer_dataset, train_args=training_args)
    logger.info('Train end! LoRA model saves in the path:::{}'.format(training_args.output_dir))


if __name__ == "__main__":
    main()
