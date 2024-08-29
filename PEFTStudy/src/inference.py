import torch
import logging
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForCausalLM, 
    HfArgumentParser
)
from peft import PeftModel
from dataclasses import dataclass, field
from finetune import get_base_llm_model_tokenizer


# 设置模型微调的参数类
@dataclass
class InferenceArguments:
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../../../model/Qwen2-7B-Instruct")
    peft_type: str = field(default="lora")
    merge_save_path: str = field(default="../out/merge_model/lora/")
    use_merge_model: str = field(default='True')
    log_path: str = field(default="../log/lora_output.log")


def get_peft_llm_model_tokenizer(inference_args):
    # 判断是否用peft_llm_model
    if bool(inference_args.use_merge_model):
        # 读取模型地址
        llm_model_name = inference_args.llm_model_name
        if inference_args.peft_type == "lora":
            # lora：直接用merge model的path
            llm_model_path = inference_args.merge_save_path  
        else:
            # 其他：先加载底座大模型，再加载peft参数
            llm_model_path = inference_args.llm_model_path  
        peft_model_path = inference_args.merge_save_path

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
        
        # # 配置模型
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.is_parallelizable = True
        model.model_parallel = True

        # 加载peft参数
        if inference_args.peft_type != "lora":
            model = PeftModel.from_pretrained(model, model_id=peft_model_path)

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    
    else:
        model, tokenizer = get_base_llm_model_tokenizer(inference_args)

    return model, tokenizer
    


def get_llm_response(query_list: list[str], inference_args):
    # 加载模型
    llm_model, llm_tokenizer = get_peft_llm_model_tokenizer(inference_args)
    llm_model.to('cuda')
    llm_model.eval()
    logger.info('PEFT LLMs {} load successfully! LLM path::: {}'.format(inference_args.llm_model_name, inference_args.merge_save_path))
    # 设置message
    llm_response_mp = {}
    for query in query_list:
        messages = [
            {"role": "system", "content": "你是百科全书智能回答助手"},
            {"role": "user", "content": query}
        ]
        messages = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = llm_tokenizer([messages], return_tensors="pt").to('cuda')
        generated_ids = llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        llm_response_mp[query] = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return llm_response_mp


def main():
    # 忽略警告
    warnings.filterwarnings("ignore")

    # 加载命令行参数
    inference_args = HfArgumentParser(
        (InferenceArguments)
    ).parse_args_into_dataclasses()[0]

    # 设置logger
    logging.basicConfig(
        level=logging.DEBUG,
        filename=inference_args.log_path,  #保存日志的本地目录
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )

    # 将logger声明为全局变量
    global logger
    logger = logging.getLogger(__name__)
    logger.debug("命令行参数")
    logger.debug("inference_args:")
    logger.debug(inference_args.__repr__())

    # 模型推理
    query_list = [
        "如何关闭华硕主板的xmp功能？"
    ]
    llm_resp = get_llm_response(query_list=query_list, inference_args=inference_args)
    print(llm_resp)
    logger.info('LLMs:::{}; Query:::{}; Response:::{}'.format(inference_args.llm_model_name, str(query_list), str(llm_resp)))


if __name__ == "__main__":
    main()
