import logging
import warnings
from peft import PeftModel
from transformers import HfArgumentParser
from peft import AutoPeftModelForCausalLM
from src.finetune import get_base_llm_model_tokenizer


# 设置模型微调的参数类
@dataclass
class InferenceArguments:
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../../../model/Qwen2-7B-Instruct")
    merge_save_path: str = field(default="../out/merge_model/lora/")
    use_merge_model: bool = field(default=True)
    log_path: str = field(default="../log/lora_output.log")


def get_llm_response(query_list: list[str], inference_args):
    # 加载模型
    llm_model, llm_tokenizer = get_base_llm_model_tokenizer(
        inference_args, 
        merge_model_path=inference_args.merge_save_path,
        use_merge_model=True
    )
    logger.info('Base LLMs {} load successfully! LLM path::: {}'.format(inference_args.llm_model_name, inference_args.llm_model_path))
    # 判断是否用merge模型
    llm_response_mp = {}
    for query in query_list:
        ipt = tokenizer("Human: {}\n{}".format(query, "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
        resp = tokenizer.decode(model.generate(**ipt,max_length=256,do_sample=False)[0],skip_special_tokens=True)
        llm_response_mp[query] = resp
    return llm_response_mp


def main():
    # 忽略警告
    warnings.filterwarnings("ignore")

    # 加载命令行参数
    inference_args = HfArgumentParser(
        (InferenceArguments)
    ).parse_args_into_dataclasses()

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
    llm_resp = get_llm_response(query_list=query_list, llm_model, llm_tokenizer, device)
    print(llm_resp)
    logger.info('LLMs:::{}; Query:::{}; Response:::{}'.format(inference_args.llm_model_name, str(query_list), str(llm_resp)))


if __name__ == "__main__":
    main()

