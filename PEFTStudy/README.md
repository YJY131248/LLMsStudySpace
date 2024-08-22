# PEFT实战（based on Qwen2-7B-Instruct）


>- 大模型底座：**Qwen2-7B-Instruct** 
>- 数据集：**shibing624/alpaca-zh**
>- 微调：**LoRA/P-tuning/Prefix-Tuning/Prompt-Tuning**


## 1. 环境/数据准备
安装项目相关依赖
```shell
pip install datasets
pip install peft
pip install transformers
```
从Hugging Face安装将要使用的LLMs以及embedding model，这里我们选择**Qwen/Qwen2-7B-Instruct**作为大模型底座，选择**shibing624/alpaca-zh**作为数据集
这里介绍快速下载huggingface模型的命令行方法：
```shell
1. 首先安装依赖
pip install -U huggingface_hub
pip install -U hf-transfer 
1. 设置环境变量(设置hf环境变量为1用于提升下载速度；设置镜像站地址)
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT="https://hf-mirror.com"
1. 安装相应的模型(以Qwen2-7B-Instruct为例，前面是huggingface上的模型名，后面是本地下载的路径)
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ./Qwen2-7B-Instruct
huggingface-cli download --repo-type dataset shibing624/alpaca-zh --local-dir ./data/
```
该数据集是基于GPT4得到的self-instruct数据，遵循**Alpaca格式**，json文件中每个条目包括：
  
- **用户指令**（Instruct，必填）
- **用户输入**（Input，额外的补充文本，选填）
- **模型回答**（Output，必填）
- 系统提示词（选填，shibing624/alpaca-zh数据集中没有该字段）
- 对话历史（选填，shibing624/alpaca-zh数据集中没有该字段）  


```json
[
  {
    "instruction": "辨认不同之处。 ",
    "input": "输入：Twitter，Instagram，Telegram。",
    "output": "Twitter，Instagram 和 Telegram 都是流行的社交媒体平台，但它们之间也有一些不同之处：\n\n- Twitter: 是一个微博社交媒体平台，用户可以发布 280 字符以内的消息（即推文）并与其他用户互动。它主要关注文字交流，尽管用户也可以分享照片和视频。\n- Instagram: 是一个图片和视频分享的社交媒体平台，用户可以在 Instagram 上发布照片和视频并与其他用户互动。它主要关注图片和视频的分享，尽管用户也可以添加文字说明。\n- Telegram: 是一个即时通讯和社交媒体平台，用户可以通过文本、语音、图片和视频与其他用户即时交流。与 Twitter 和 Instagram 不同，Telegram 更侧重于即时通讯，它还提供了端到端加密的私人聊天功能。"
  },
  ...
]
```

## 2. 训练集/测试集划分

在该节，我们编写了两个函数：

- **get_alpaca_dataset：** 使用**datasets**加载json文件，并按照0.9:0.1的比例构建训练集和测试集
- **get_tokenizer_dataset：** 对输入的dataset，使用tokenizer获取**input_ids, attention_mask, labels**；同时该函数会考虑传入datasets和tokenizer为空的情况，通过json_path和tokenizer_path自动加载datasets和tokenizer，如果上面两个参数也没有，则raise报错

```python
from transformers import AutoTokenizer
from datasets import load_dataset


def get_alpaca_dataset(json_path: str, test_size: float=0.1):
    dataset = load_dataset(
        'json', 
        data_files=json_path,
        split="train"
    )
    dataset = dataset.train_test_split(test_size=test_size)
    return dataset


def get_tokenizer_dataset(
        dataset, 
        tokenizer,
        max_length: int=256,
        json_path: str="",
        tokenizer_path: str="",
    ):

    def process_sample(sample):
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            "\n".join([
                "Human:" + sample["instruction"],
                sample["input"]
            ]).strip()
            + "\n\nAssistant: "
        )
        responese = tokenizer(sample["output"] + tokenizer.eos_token)
        input_ids = instruction["input_ids"] + responese["input_ids"]
        attention_mask = instruction["attention_mask"] + responese["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + responese["input_ids"]
        # 最大长度截断
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        # 返回结果
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    # 如果没有传入dataset
    if dataset is None:
        # 如果传入json_path，则自动执行get_alpaca_dataset获取dataset
        if json_path != "":
            dataset = get_alpaca_dataset(json_path=json_path, test_size=0.1)
        # 否则，直接报错
        else:
            raise ValueError("错误参数：dataset不能为空")

    # 如果没有传入tokenizer
    if tokenizer is None:
        # 如果传入tokenizer_path，则加载tokenizer
        if tokenizer_path != "":
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # 否则，直接报错
        else:
            raise ValueError("错误参数：tokenizer不能为空")
        
    return dataset.map(process_sample, remove_columns=dataset['train'].column_names)
```

## 3. 微调finetune

### 3.1 导入依赖

我们主要使用transformers框架用于加载大模型，使用peft库进行高效微调
```python
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
    TaskType, 
    get_peft_model
)
from dataclasses import dataclass, field
from dataset_load import get_alpaca_dataset, get_tokenizer_dataset
```

### 3.2 设置logger

我们初始化一个**日志记录logger**，用于记录运行过程中的信息
```python
import logging

# 设置logger
logging.basicConfig(
    level=logging.DEBUG,
    filename='../out/output.log',  # 替换成保存日志的本地目录
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

### 3.3 加载命令行参数

我们需要从sh运行文件中解析需要的参数（包括大模型路径、peft类型、结果保存路径以及max_length等超参数），这里我们基于transformers的**HfArgumentParser**自动解析命令行的参数，并保存至**TrainingArguments（源于transformers库）**和**FinetuneArguments（自定义）**

```python
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

# 加载命令行参数
finetune_args, training_args = HfArgumentParser(
    (FinetuneArguments, TrainingArguments)
).parse_args_into_dataclasses()
```

### 3.4 加载大模型

我们的代码中允许用户使用**Qwen、ChatGLM以及BaiChuan**三类模型，并且按照模型类型设置model的加载方式（AutoModelForCausalLM、AutoModel），对于其他模型，我们直接raise报错（当然你也可以手动对其他模型进行扩充），后文我们都将使用Qwen2-7B-Instruct模型

同时在该代码中，我们也初始化了tokenizer（基于transformers的AutoTokenizer）

```python
# 加载LLMs model/tokenizer
def get_base_llm_model_tokenizer(finetune_args):
    # 读取模型类型
    llm_model_name = finetune_args.llm_model_name
    llm_model_path = finetune_args.llm_model_path
    # 加载llm_model
    if llm_model_name == "Qwen" or llm_model_name == "BaiChuan":
        model = AutoModelForCausalLM.from_pretrained(llm_model_path, low_cpu_mem_usage=True, torch_dtype=torch.half)
    elif llm_model_name == "ChatGLM":
        model = AutoModel.from_pretrained(llm_model_path, low_cpu_mem_usage=True, torch_dtype=torch.half)
    # 模型不是为本项目支持的模型
    else:
        logger.error("错误参数：底座模型必须是Qwen/ChatGLM/BaiChuan")
        raise ValueError("错误参数：底座模型必须是Qwen/ChatGLM/BaiChuan")
    # 配置模型
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.is_parallelizable = True
    model.model_parallel = True
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    return model, tokenizer
```

### 3.5 设置PEFT的配置config

这里我们同样考虑了四种常见的peft方法：LoRA、p-tuning、prefix-tuning，prompt-tuning。并按照peft_type类型判断当前使用的peft方法，设置相应的配置文件config。如果命令行中传入的peft_type不在上述四种方法中，我们也会raise报错

```python
# 根据peft类型返回相应的config
def get_peft_config(finetune_args):
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
            encoder_dropout=0.1, 
            encoder_num_layers=5, 
            encoder_hidden_size=512
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
            num_virtual_tokens=10,
            prompt_tuning_init_text="Classify if the tweet is a complaint or not:",  # 自行设置prompt
            tokenizer_name_or_path=finetune_args.llm_model_path
        )
    else:
        logger.error("错误参数：peft类型必须为lora/p-tuning/prefix-tuning/prompt-tuning")
        raise ValueError("错误参数：peft类型必须为lora/p-tuning/prefix-tuning/prompt-tuning")

    return peft_config
```

### 3.6 进行模型微调

模型微调则使用**get_peft_model**方法，该函数传入大模型model以及peft的config，使用**transformers的Trainer**类完成大模型的微调任务

```python
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
```

### 3.7 main函数

这里我们编写了main函数，穿起前面提到的各个流程：

- 加载命令行参数
- 设置logger（采用全局变量global）
- 加载LLM的model和tokenizer
- 获取peft_config参数
- 加载微调所使用的数据（基于第二节的get_alpaca_dataset和get_tokenizer_dataset）
- 完成微调

```python
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
    peft_config = get_peft_config(finetune_args)
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

    # # 合并LoRA参数
    # qwen_model = qwen_model.cuda()
    # lora_model = PeftModel.from_pretrained(qwen_model, model_id=)
    # mergemodel = model.merge_and_unload()
    # mergemodel.save_pretrained("./merge_model")



if __name__ == "__main__":
    main()
```

在sh文件中，我们传入peft所需要的所有参数：

- **finetune_args:**
  - peft类型
  - 大模型名字、大模型本地路径（相对）
  - 数据集本地路径（相对）
  - log文档保存路径（相对）
  - 文本最大长度max_length
  - LoRA的秩
    
- **training_args:**
  - 模型保存路径
  - 训练阶段的batch size
  - 验证阶段的batch size
  - 训练轮数
  - 最大步长
  - 保存checkoutpoint周期
  - 学习率等

sh文件
```shell
python3 finetune.py \
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

```
