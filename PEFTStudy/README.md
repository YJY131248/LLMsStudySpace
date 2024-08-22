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
