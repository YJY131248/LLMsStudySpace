import logging
import warnings
from peft import PeftModel
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from finetune import get_base_llm_model_tokenizer

# 设置模型微调的参数类
@dataclass
class MergeModelArguments:
    peft_type: str = field(default="lora")
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../../../model/Qwen2-7B-Instruct")
    peft_checkpoint_path: str = field(default="../out/lora_peft/checkpoint-500/")
    merge_save_path: str = field(default="../out/merge_model/lora/")
    log_path: str = field(default="../log/lora_output.log")

def main():
    # 忽略警告
    warnings.filterwarnings("ignore")

    # 加载命令行参数
    merge_model_args = HfArgumentParser(
        (MergeModelArguments)
    ).parse_args_into_dataclasses()[0]

    # 设置logger
    logging.basicConfig(
        level=logging.DEBUG,
        filename=merge_model_args.log_path,  #保存日志的本地目录
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )

    # 将logger声明为全局变量
    global logger
    logger = logging.getLogger(__name__)
    logger.debug("命令行参数")
    logger.debug("merge_model_args:")
    logger.debug(merge_model_args.__repr__())

    # 加载模型
    llm_model, llm_tokenizer = get_base_llm_model_tokenizer(merge_model_args)
    logger.info('Base LLMs {} load successfully! LLM path::: {}'.format(merge_model_args.llm_model_name, merge_model_args.llm_model_path))

    # 加载微调的参数
    llm_model = llm_model.cuda()
    peft_model = PeftModel.from_pretrained(llm_model, model_id=merge_model_args.peft_checkpoint_path)

    # 如果是lora微调，则进行参数合并
    if merge_model_args.peft_type == "lora":
        merge_model = peft_model.merge_and_unload()
    # 其他微调方式，保存微调参数
    else:
        merge_model = peft_model

    # 保存model和tokenizer
    merge_model.save_pretrained(merge_model_args.merge_save_path)
    llm_tokenizer.save_pretrained(merge_model_args.merge_save_path)

    # 打印日志
    logger.info('Merge Model {} saved successfully! PEFT checkout path:::{}! Merge Model path::: {}'.format(
        merge_model_args.llm_model_name, 
        merge_model_args.peft_checkpoint_path, 
        merge_model_args.merge_save_path)
    )


if __name__ == "__main__":
    main()
