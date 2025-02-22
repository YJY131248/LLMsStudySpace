import logging
import warnings
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    HfArgumentParser
)
from typing import Optional, Union
from peft import PeftModelForCausalLM
from dataclasses import dataclass, field
from finetune import get_llm_model_tokenizer
from tqdm import tqdm

# Define the inference argument class
@dataclass
class InferenceArguments:
    peft_type: str = field(default="lora")
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../model/Qwen2-7B-Instruct")
    peft_model_path: str = field(default="../out/car_lora_model/checkpoint-3000/")
    use_peft_model: bool = field(default=False)
    log_path: str = field(default="../log/car_model_inference.log")
    max_new_tokens: int = field(default=1024)
    do_sample: bool = field(default=False)
    top_p: float = field(default=0.1)
    temperature: float = field(default=0.1)
    repetition_penalty: float = field(default=1.2)

def get_peft_llm_model_tokenizer(
    llm_model_name: str,
    llm_model_path: str,
    peft_model_path: Optional[str],
    peft_type: Optional[str],
    use_peft_model: bool = False, 
):
    # load the base LLM model and tokenizer
    model, tokenizer = get_llm_model_tokenizer(llm_model_name, llm_model_path, peft_type=peft_type)
    # set merge model arguments
    if use_peft_model:
        # load the PEFT model
        if peft_type != "lora":
            model = PeftModelForCausalLM.from_pretrained(model, model_id=peft_model_path)
    return model, tokenizer

def get_llm_response(
    query_list: list[str], 
    model: Union[AutoModelForCausalLM, PeftModelForCausalLM], 
    tokenizer: AutoTokenizer, 
    **kwargs
):
    # load the LLM model and tokenizer
    model = model.cuda()
    model.eval()
    # set the response map
    llm_response_mp = {}
    for query in tqdm(query_list):
        try:
            # qwen/llama/mistral/glm
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # baichuan
            messages = f"Human:{query}\nYou are a helpful assistant.\n\nAssistant: "
        # encode the input
        model_inputs = tokenizer([messages], return_tensors="pt").to(model.device)
        # update the generate kwargs
        generate_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": False,
            "top_p": 0.8,
            "temperature": 0.8,
            "repetition_penalty": 1.2,
            "eos_token_id": model.config.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id
        }
        generate_kwargs.update(kwargs)
        # generate the response
        try:
            generated_ids = model.generate(**model_inputs, **generate_kwargs)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            llm_response_mp[query] = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except:
            llm_response_mp[query] = "-1"
            continue

    return llm_response_mp

def main():
    # ignore warnings
    warnings.filterwarnings("ignore")

    # load arguments
    inference_args = HfArgumentParser(
        (InferenceArguments)
    ).parse_args_into_dataclasses()[0]

    # set logger
    logging.basicConfig(
        level=logging.INFO,
        filename=inference_args.log_path,
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Arguments: ")
    logger.info("inference_args:")
    logger.info(inference_args.__repr__())

    # load model
    llm_model, llm_tokenizer = get_peft_llm_model_tokenizer(
        inference_args.llm_model_name,
        inference_args.llm_model_path,
        inference_args.peft_model_path,
        inference_args.peft_type,
        inference_args.use_peft_model
    )
    logger.info('LLMs {} load successfully! LLM path::: {}'.format(inference_args.llm_model_name, inference_args.peft_model_path))

    # model inference
    query_list = [
        "You are a specialized cognitive impairment assessment expert. Your task is to analyze text input and classify the cognitive status into one of three categories:\n[0] Normal Cognition\n[1] Mild Cognitive Impairment (MCI)\n[2] Moderate to Severe Cognitive Impairment\n\nContext:\nThe text input represents natural language responses or conversations from individuals. Classification should be based on established clinical criteria and linguistic markers of cognitive decline.\n\nKey Assessment Dimensions:\n1. Language Processing\n- Coherence and clarity of expression\n- Semantic accuracy and appropriateness\n- Syntactic complexity and grammatical accuracy\n\n2. Memory Function\n- Information recall accuracy\n- Temporal sequencing\n- Context maintenance\n\n3. Executive Function\n- Logic and reasoning\n- Problem-solving ability\n- Abstract thinking\n\n4. Attention and Processing\n- Focus maintenance\n- Response relevance\n- Information processing speed\n\nClassification Guidelines:\n[0] Normal Cognition:\n- Clear, coherent communication\n- Appropriate context maintenance\n- Complex sentence structures\n- Accurate information processing\n- Strong logical reasoning\n\n[1] Mild Cognitive Impairment:\n- Minor communication inconsistencies\n- Occasional context loss\n- Simplified sentence structures\n- Slight delays in processing\n- Some logical gaps\n\n[2] Moderate to Severe Impairment:\n- Significant communication difficulties\n- Frequent context loss\n- Basic/fragmented sentences\n- Major processing delays\n- Obvious logical breakdowns\n\nOutput Format:\nAnalyze the input text and provide:\n1. Classification label (0, 1, or 2)\n2. Confidence score (0-1)\n3. Brief reasoning (\u226450 words)\n\nInput: how excess iron raises your risk for alzheimers\n\nTask: Based on the provided guidelines, analyze the input text and classify the cognitive status. Provide your assessment in the specified format."
    ]
    llm_resp = get_llm_response(
        query_list=query_list, 
        model=llm_model, 
        tokenizer=llm_tokenizer,
        max_new_tokens=inference_args.max_new_tokens,
        top_p=inference_args.top_p,
        temperature=inference_args.temperature,
        repetition_penalty=inference_args.repetition_penalty,
        do_sample=inference_args.do_sample,
    )
    print(llm_resp)
    logger.info('LLMs:::{}; Query:::{}; Response:::{}'.format(inference_args.llm_model_name, str(query_list), str(llm_resp)))

if __name__ == "__main__":
    main()