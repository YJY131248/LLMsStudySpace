import json
import warnings
import logging
from llama_index.core import Settings, Document, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.llms import ChatMessage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from utils import messages_to_prompt, completion_to_prompt

# 设置忽略警告
warnings.filterwarnings("ignore")

# 设置logger
logging.basicConfig(
    level=logging.DEBUG,
    filename='../out/output.log',
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置embedding model(bge-base-zh-v1.5)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="../../../model/bge-base-zh-v1.5"
)

# 设置llm(Qwen2-7B-Instruct)
Settings.llm = HuggingFaceLLM(
    model_name="../../../model/Qwen2-7B-Instruct",
    tokenizer_name="../../../model/Qwen2-7B-Instruct",
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

def get_documents_qa_data(documents_path: str):
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


def get_nodes_from_documents_by_chunk(documents: list):
    # 对文本进行chunk
    node_paeser = SimpleNodeParser.from_defaults(chunk_size=1024)
    # [node1,node2,...] 每个node的结构为{'Node_ID':xxx, 'Text':xxx}
    nodes = node_paeser.get_nodes_from_documents(documents=documents)
    # 如果nodes长度为0，则在日志中记录
    if len(nodes) == 0:
        logger.warning('nodes list length::: 0')
    logger.info('nodes build successfully!')
    return nodes


def get_vector_index(nodes=None, documents=None, store_path: str='../store/', use_store: bool=True):
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


def get_llm_answer_with_rag(query: str, index=None, use_retrieval: bool=True, use_rerank: bool=True):
    # 使用检索
    if use_retrieval:
        # 自定义retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
        )
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
        # 使用rerank（基于bge_rerank）
        if use_rerank:
            reranker = FlagEmbeddingReranker(
                model="../../../model/bge-reranker-base", 
                top_n=5
            )
            node_postprocessors.append(reranker)
        # 构建query_engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=node_postprocessors
        )
        # 获取大模型的response
        resp = query_engine.query(query)
        logger.info('use rag, query:::{}, resp:::{}'.format(query, resp))
    # 不使用RAG, 直接由LLMs进行输出
    else:
        resp = Settings.llm.chat(messages=[ChatMessage(content=query)])
        logger.info('not use rag, query:::{}, resp:::{}'.format(query, resp))
    return resp


def main():
    documents_path = '../data/cmrc-eval-zh.jsonl'
    store_path = '../store/'
    # 加载文档
    documents, qa_data_mp = get_documents_qa_data(documents_path)
    # 进行chunk
    nodes = get_nodes_from_documents_by_chunk(documents)
    # 构建向量索引
    index = get_vector_index(nodes=nodes, store_path=store_path, use_store=True)
    # 获取检索增强的llm回复
    for idx, qa_data in enumerate(qa_data_mp):
        if idx > 2:
            break
        query = qa_data["question"]
        reference_answer = qa_data["reference_answer"]
        llm_resp = get_llm_answer_with_rag(query, index, use_retrieval=True, use_rerank=True)
        print("query::: {}".format(query))
        print("reference answer::: {}".format(reference_answer))
        print("answer::: {}".format(llm_resp.response))
        print("source_nodes::: {}".format(llm_resp.source_nodes))
        print("formatted_sources::: {}".format(llm_resp.get_formatted_sources()))
        print("*"*100)
        


if __name__ == "__main__":
    main()
