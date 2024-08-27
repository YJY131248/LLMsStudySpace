from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def get_document_text(doc_path_list: list[str]) -> list[str]:
    text_list = []
    for doc_path in doc_path_list:
        with open(doc_path, 'r', encoding='utf-8') as f:
            text = f.read()
        text_list.append(text)
    return text_list


def character_chunking(text_list: list[str], character_type: str="char"):
    if character_type == "char":
        # 字符级
        text_splitter = CharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=128, 
            separator="\n", 
            strip_whitespace=True
        )
    elif character_type == "token":
        # token级别
        # tiktoken是一种快速 BPE tokenizer
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=512, 
            chunk_overlap=128, 
            separator="\n", 
            strip_whitespace=True
        )
    else:
        return
    
    chunking_res_list = text_splitter.create_documents(text_list)
    for chunking_res in chunking_res_list:
        print(chunking_res)
        print("*"*100)


def recursive_character_chunking(text_list: list[str], character_type: str="char"):
    if character_type == "char":
        # 字符级
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=128, 
            separators=["\n\n", "\n", "。", ".", "?", "？", "!", "！"], 
            strip_whitespace=True
        )
    elif character_type == "token":
        # token级别 
        # tiktoken是一种快速 BPE tokenizer
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=512, 
            chunk_overlap=128, 
            separators=["\n\n", "\n", "。", ".", "?", "？", "!", "！"], 
            strip_whitespace=True
        )
    else:
        return

    chunking_res_list = text_splitter.create_documents(text_list)
    for chunking_res in chunking_res_list:
        print(chunking_res)
        print("*"*100)


def semantic_chunking(text_list: list[str]): 
    # embeddings = OpenAIEmbeddings()  # 使用openai模型
    embeddings = HuggingFaceBgeEmbeddings(  
        model_name = '../../../model/bge-base-zh-v1.5'
    ) # 使用huggingface的bge embeddings模型
    text_splitter = SemanticChunker(
        embeddings = embeddings,
        breakpoint_threshold_type = "percentile",  # 百分位数
        breakpoint_threshold_amount = 30,  # 百分比
        sentence_split_regex = r"(?<=[。？！])\s+"  # 正则，用于分句
    )
    chunking_res_list = text_splitter.create_documents(text_list)
    for chunking_res in chunking_res_list:
        print(chunking_res)
        print("*"*100)


if __name__ == "__main__":
    doc_path_list = [
        '../data/chunking_test.txt'
    ]
    text_list = get_document_text(doc_path_list)
    # character_chunking(text_list)
    recursive_character_chunking(text_list, character_type="token")
    # semantic_chunking(text_list)
