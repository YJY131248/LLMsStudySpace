import logging
import warnings
from peft import PeftModel
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from finetune import get_llm_model_tokenizer

# set merge model arguments
@dataclass
class MergeModelArguments:
    peft_type: str = field(default="lora")
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../model/Qwen2-7B-Instruct")
    peft_checkpoint_path: str = field(default="../out/lora_peft/checkpoint-3000/")
    merge_save_path: str = field(default="../model/lora_qwen2.5_7b")
    log_path: str = field(default="../log/lora_merge.log")

def main():
    # ignore warnings
    warnings.filterwarnings("ignore")

    # load arguments
    merge_model_args = HfArgumentParser(
        (MergeModelArguments)
    ).parse_args_into_dataclasses()[0]

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        filename=merge_model_args.log_path,
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    logger.info("merge_model_args:")
    logger.info(merge_model_args.__repr__())

    # load the base LLM model and tokenizer
    llm_model, llm_tokenizer = get_llm_model_tokenizer(
        merge_model_args.llm_model_name, 
        merge_model_args.llm_model_path, 
        merge_model_args.peft_type
    )
    logger.info('Base LLMs {} load successfully! LLM path::: {}'.format(merge_model_args.llm_model_name, merge_model_args.llm_model_path))

    # load the PEFT model
    llm_model = llm_model.cuda()
    peft_model = PeftModel.from_pretrained(llm_model, model_id=merge_model_args.peft_checkpoint_path)

    # merge the model
    if merge_model_args.peft_type == "lora":
        merge_model = peft_model.merge_and_unload()
    else:
        merge_model = peft_model

    # save the model
    merge_model.save_pretrained(merge_model_args.merge_save_path)
    llm_tokenizer.save_pretrained(merge_model_args.merge_save_path)
    logger.info('Merge Model {} saved successfully! PEFT checkout path:::{}! Merge Model path::: {}'.format(
        merge_model_args.llm_model_name, 
        merge_model_args.peft_checkpoint_path, 
        merge_model_args.merge_save_path)
    )

if __name__ == "__main__":
    main()