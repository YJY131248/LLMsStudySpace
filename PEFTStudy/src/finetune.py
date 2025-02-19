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
from typing import Dict, Union
from dataclasses import dataclass, field
from data_utils import get_alpaca_dataset, get_tokenizer_dataset

# Define the fine-tuning argument class
@dataclass
class FinetuneArguments:
    peft_type: str = field(default="lora")
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../model/Qwen2-7B-Instruct")
    dataset_path: str = field(default="../data/alpaca_gpt4_data_zh.json")
    log_path: str = field(default="../log/lora_output.log")
    max_length: int = field(default=1024)
    lora_rank: int = field(default=16)
    lora_alpha: int = field(default=32)

# Load LLM model and tokenizer
def get_llm_model_tokenizer(llm_model_name, llm_model_path, peft_type):
    """
    Load base LLM model and tokenizer
    Args:
        llm_model_name: Name of the LLM model
        llm_model_path: Path to the LLM model
        peft_type: Type of PEFT method
    Returns:
        Tuple of (model, tokenizer)
    """

    try:
        if llm_model_name in ["Qwen", "Mistral", "Gemma"]:
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_path, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        elif llm_model_name == "Llama":
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
            )
            tokenizer = AutoTokenizer.from_pretrained(
                llm_model_path, 
                legacy=True,
                use_fast=False,
                padding_side="right"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif llm_model_name == "BaiChuan":
            model = AutoModelForCausalLM.from_pretrained(
                llm_model_path, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
        elif llm_model_name == "ChatGLM":
            model = AutoModel.from_pretrained(
                llm_model_path, 
                low_cpu_mem_usage=True, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                llm_model_path, 
                trust_remote_code=True
            )
        else:
            logger.error(f"Invalid model: Supported models are Qwen, ChatGLM, BaiChuan, Mistral, Llama, Gemma")
            raise ValueError(f"Invalid model: Supported models are Qwen, ChatGLM, BaiChuan, Mistral, Llama, Gemma")
        
        if peft_type != "prefix-tuning" and llm_model_name != "ChatGLM":
            model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.is_parallelizable = True
        model.model_parallel = True

        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise e

# Configure PEFT
def get_peft_config(peft_type, tokenizer, finetune_args):
    """
    Get PEFT configuration based on specified type
    Args:
        finetune_args: Fine-tuning arguments containing PEFT specifications
        tokenizer: Tokenizer for prompt tuning initialization
    Returns:
        PEFT configuration object
    """
    if peft_type == "lora":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=finetune_args.lora_alpha,
            lora_dropout=0.1,
        )
    elif peft_type == "p-tuning":
        return PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=10,
            encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
            encoder_hidden_size=1024
        )
    elif peft_type == "prefix-tuning":
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM, 
            num_virtual_tokens=10,
            prefix_projection=True
        )
    elif peft_type == "prompt-tuning":
        return PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text="You are an AI assistant.",
            num_virtual_tokens=len(tokenizer("You are an AI assistant.")["input_ids"]),
            tokenizer_name_or_path=finetune_args.llm_model_path
        )
    else:
        logger.error("Invalid PEFT type: Must be lora, p-tuning, prefix-tuning, or prompt-tuning")
        raise ValueError("Invalid PEFT type")

# Training function
def finetune_train(
    model: Union[AutoModelForCausalLM, AutoModel],
    peft_config: Union[LoraConfig, PromptEncoderConfig, PrefixTuningConfig, PromptTuningConfig],
    tokenizer: AutoTokenizer,
    dataset: Dict,
    train_args: TrainingArguments
):
    """
    Fine-tune the model using specified PEFT method
    Args:
        model: Base LLM model
        peft_config: PEFT configuration
        tokenizer: Tokenizer for text processing
        dataset: Training and evaluation datasets
        train_args: Training arguments
    """
    model = get_peft_model(model=model, peft_config=peft_config)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()

def main():
    # ignore warnings
    warnings.filterwarnings("ignore")

    # load arguments
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        filename=finetune_args.log_path,
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Arguments: ")
    logger.info("finetune_args:")
    logger.info(finetune_args.__repr__())
    logger.info("training_args:")
    logger.info(training_args.__repr__())

    # load the base LLM model and tokenizer
    llm_model, llm_tokenizer = get_llm_model_tokenizer(finetune_args.llm_model_name, finetune_args.llm_model_path, finetune_args.peft_type)
    logger.info('Base LLMs {} load successfully! LLM path::: {}'.format(finetune_args.llm_model_name, finetune_args.llm_model_path))

    # load the PEFT configuration
    peft_config = get_peft_config(finetune_args.peft_type, llm_tokenizer, finetune_args)
    logger.info('Peft {} config load successfully!'.format(finetune_args.peft_type))

    # load the dataset and tokenizer dataset
    dataset = get_alpaca_dataset(finetune_args.dataset_path, test_size=0.1)
    logger.info('dataset build successfully!')
    tokenizer_dataset = get_tokenizer_dataset(dataset, llm_tokenizer, max_length=finetune_args.max_length)
    logger.info('tokenizer dataset build successfully!')

    # start training
    logger.info('Train start!')
    finetune_train(
        model=llm_model, 
        peft_config=peft_config, 
        tokenizer=llm_tokenizer, 
        dataset=tokenizer_dataset, 
        train_args=training_args
    )
    logger.info('Train end! LoRA model saves in the path:::{}'.format(training_args.output_dir))

if __name__ == "__main__":
    main()