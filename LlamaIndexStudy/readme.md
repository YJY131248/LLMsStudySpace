# RAG框架快速构建（based on Qwen2-7B-Instruct）

> 基于**LlamaIndex**框架，以**Qwen2-7B-Instruct**作为大模型底座，**bge-base-zh-v1.5**作为embedding模型，构建RAG基础链路。数据集选用**cmrc2018**数据集（该数据集禁止商用）


## 1. 环境准备
安装LlamaIndex的相关依赖
```shell
pip install llama-index
pip install llama-index-llms-huggingface
pip install llama-index-embeddings-huggingface
pip install datasets
```
从Hugging Face安装将要使用的LLMs以及embedding model，这里我们选择**Qwen/Qwen2-7B-Instruct**作为大模型底座，选择**BAAI/bge-base-zh-v1.5**作为embedding模型，用于对文档进行向量化表征
这里介绍快速下载huggingface模型的命令行方法：
```shell
1. 首先安装依赖
pip install -U huggingface_hub
pip install -U hf-transfer 
2. 设置环境变量(设置hf环境变量为1用于提升下载速度；设置镜像站地址)
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT="https://hf-mirror.com"
3. 安装相应的模型(以Qwen2-7B-Instruct为例，前面是huggingface上的模型名，后面是本地下载的路径)
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ./Qwen2-7B-Instruct
huggingface-cli download BAAI/bge-base-zh-v1.5 --local-dir ./bge-base-zh-v1.5
```

## 2. 数据准备
使用开源数据集 **[cmrc2018](https://huggingface.co/datasets/hfl/cmrc2018)** 构建本地知识库，该数据集由人类专家在**维基百科**段落上标注的近2万个真实问题组成，包含**上下文（可以用于构建本地知识库）、问题和答案**

![image](https://github.com/YJY131248/LLMsStudySpace/blob/main/LlamaIndexStudy/img/1_data.png)


我们读取cmrc的train数据集，并将**question，answer以及context**存储到本地data目录下的**cmrc-eval-zh.jsonl**文件下，代码如下：
```python
import json
from tqdm import tqdm
from datasets import load_dataset

cmrc = load_dataset("cmrc2018")
with open("../data/cmrc-eval-zh.jsonl", "w") as f:
    for train_sample in tqdm(cmrc["train"]):
        qa_context_mp = {
            "question": train_sample["question"],
            "reference_answer": train_sample["answers"]["text"][0],
            "reference_context": train_sample["context"]
        }
        f.write(json.dumps(qa_context_mp, ensure_ascii=False) + "\n")
```

![image](https://github.com/YJY131248/LLMsStudySpace/blob/main/LlamaIndexStudy/img/2-json.png)

## 3. RAG框架构建
在具体构建rag链路之前，我们初始化一个**日志记录logger**，用于记录运行过程中的信息
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

### 3.1 数据读取 + 数据切块
读取第二节中构建的**cmrc-eval-zh.jsonl**文件，将**reference_context字段**读取出来保存到list中作为后续知识库的文本，另外将**question-answer pair**进行保存，用于后续模型的推理。然后构建l**lama_index.core的Document**对象。如果是本地的txt文档知识库，也可以直接使用**SimpleDirectoryReader**方法
```python
from llama_index.core import Document, SimpleDirectoryReader

def get_documents_qa_data(documents_path):
    # 遍历jsonl文件，读取reference_context、question、reference_answer字段
    text_list = []
    qa_data_mp = []
    with open(documents_path, 'r', encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if sample["reference_context"] not in text_list:
                text_list.append(sample["reference_context"])
            qa_data_mp.append({
                "question": sample["question"],
                "reference_answer": sample["reference_answer"]
            })
    # 构建Document对象列表
    documents = [Document(text=t) for t in text_list]
    """
    # 如果直接读取本地txt文档
    documents = SimpleDirectoryReader(
        input_files='xxx.txt'
    ).load_data()
    """
    # 如果documents长度为0，则在日志中记录
    if len(documents) == 0:
        logger.warning('documents list length::: 0')
    logger.info('documents build successfully!')
    return documents, qa_data_mp
```

考虑到实际RAG应用场景中，很多文档的长度过长，一方面**难以直接放入LLM的上下文**中，另一方面在**可能导致检索过程忽略一些相关的内容导致参考信息质量不够**，一般会采用对文本进行**chunk**的操作，将一段长文本切分成多个小块，这些小块在LlamaIndex中表示为**Node**节点。上述过程代码如下，我们将documents传入到下面定义的函数中，通过**SimpleNodeParser**对文本进行切块，并设置chunk的**size为1024**
```python
from llama_index.core.node_parser import SimpleNodeParser

def get_nodes_from_documents_by_chunk(documents):
    # 对文本进行chunk
    node_paeser = SimpleNodeParser.from_defaults(chunk_size=1024)
    # [node1,node2,...] 每个node的结构为{'Node_ID':xxx, 'Text':xxx}
    nodes = node_paeser.get_nodes_from_documents(documents=documents)
    # 如果nodes长度为0，则在日志中记录
    if len(nodes) == 0:
        logger.warning('nodes list length::: 0')
    logger.info('nodes build successfully!')
    return nodes
```

### 3.2 构建向量索引
在RAG的retrieval模块中，一般采用的检索方式有两种：
- 基于**文本匹配**的检索（例如，**bm25**算法）
- 基于**向量相似度**的检索（embedding+relevance computing，例如**bge**等模型）

本文对前面构建的node节点中的文本进行**向量化表征**（基于**BAAI/bge-base-zh-v1.5**），然后构建每个node的索引，这里的embedding模型我们在第一节环境准备过程中已经下载至本地目录
此外，在构建索引后，可以使用**StorageContext**将index存储到本地空间，后续调用get_vector_index时可以先判断本地是否有存储过storage_context，如果有则直接加载即可（通过**load_index_from_storage**），如果没有则通过传入的nodes参数再次构建向量索引

```python
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 设置embedding model(bge-base-zh-v1.5)
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "../../../model/bge-base-zh-v1.5"
)

def get_vector_index(nodes=None, documents=None, store_path='../store/', use_store=True):
    # 如果已经进行过持久化数据库的存储，则直接加载
    if use_store:
        storage_context = StorageContext.from_defaults(persist_dir=store_path)
        index = load_index_from_storage(storage_context)
    # 如果没有存储，则直接构建vectore-store-index
    else:
        # 构建向量索引vector-index
        if nodes is not None:
            index = VectorStoreIndex(
                nodes, # 构建的节点，如果没有进行chunk，则直接传入documents即可
                embed_model=Settings.embed_model # embedding model设置
            )
        else:
            index = VectorStoreIndex(
                documents, # 没有进行chunk，则直接传入documents
                embed_model=Settings.embed_model # embedding model设置
            )
        #进行持久化存储
        index.storage_context.persist(store_path)  
    logger.info('vector-index build successfully!\nindex stores in the path:::{}'.format(store_path))
    return index
```
构建好的index保存到本地'../store/'目录下：

![image](https://github.com/YJY131248/LLMsStudySpace/blob/main/LlamaIndexStudy/img/3-store.png)

### 3.3 检索增强
接下来，我们在代码中设置将要使用的LLMs，本文选择通义千问的**Qwen2-7B-Instruct**模型，在第一节中也已经下载至本地，通过**HuggingFaceLLM**设置。（其他大模型的使用方式也是类似，如果是OpenAI的大模型则不使用该方式，此类教程很多，本文不在赘述）

下面的代码中，首先使用**HuggingFaceLLM**设置通义千问大模型；同时根据通义千问的官方文档中的LlamaIndex使用demo，完成**messages_to_prompts和completion_to_prompt**两个函数的设置(新起一个utils.py用于存放着两个函数)
```python
from llama_index.core.llms import ChatMessage
from llama_index.llms.huggingface import HuggingFaceLLM
from utils import messages_to_prompt, completion_to_prompt

# 设置llm(Qwen2-7B-Instruct)
Settings.llm = HuggingFaceLLM(
    model_name="../../../model/Qwen1.5-7B-Chat",
    tokenizer_name="../../../model/Qwen1.5-7B-Chat",
    context_window=30000,
    max_new_tokens=2000,
    generate_kwargs={
        "temperature": 0.7,
        "top_k": 50, 
        "top_p": 0.95
    },
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto"
)

def get_llm_answer_with_rag(query, index, use_rag=True):
    if use_rag:
        query_engine = index.as_query_engine()
        resp = query_engine.query(query).response
        logger.info('use rag, query:::{}, resp:::{}'.format(query, resp))
    else:
        resp = Settings.llm.chat(messages=[ChatMessage(content=query)])
        logger.info('not use rag, query:::{}, resp:::{}'.format(query, resp))
    return resp
```

```python
# utils.py
def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt

def completion_to_prompt(completion):
   return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"
```

### 3.4 main函数
最后，我们在main函数里面讲前面的整个链路打通，同时我们也从**cmrc-eval-zh.jsonl**中读取qa对
```python
from collections import defaultdict

def main():
    documents_path = '../data/cmrc-eval-zh.jsonl'
    store_path = '../store/'
    # 加载文档
    documents, qa_data_mp = get_documents_qa_data(documents_path)
    # 进行chunk
    nodes = get_nodes_from_documents_by_chunk(documents)
    # 构建向量索引
    index = get_vector_index(nodes=nodes, store_path=store_path, use_store=False)
    # 获取检索增强的llm回复
    for qa_data in qa_data_mp:
        query = qa_data["question"]
        reference_answer = qa_data["reference_answer"]
        llm_resp = get_llm_answer_with_rag(query, index, use_rag=True)
        print("query::: {}".format(query))
        print("reference answer::: {}".format(reference_answer))
        print("answer::: {}".format(llm_resp))
        print("*"*100)
        
       
if __name__ == "__main__":
    main()
```
运行后结果如下：

![image](https://github.com/YJY131248/LLMsStudySpace/blob/main/LlamaIndexStudy/img/4-res.png)

# 参考
1. [LlamaIndex官方文档](https://llama-index.readthedocs.io/zh/latest/index.html)
2. [Qwen官方文档](https://qwen.readthedocs.io/zh-cn/latest/framework/LlamaIndex.html)
3. [本项目Github链接](https://github.com/YJY131248/LLMsStudySpace/edit/main/LlamaIndexStudy/readme.md)
