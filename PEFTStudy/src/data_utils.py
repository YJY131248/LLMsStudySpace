import re
import string
from transformers import AutoTokenizer
from datasets import load_dataset

def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = text.strip()
    return text

def get_alpaca_dataset(json_path: str, test_size: float=0.1):
    dataset = load_dataset(
        'json', 
        data_files=json_path,
        split="train"
    )
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
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
        # Truncate
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    if dataset is None:
        if json_path != "":
            dataset = get_alpaca_dataset(json_path=json_path, test_size=0.1)
        else:
            raise ValueError("error：dataset is None")

    if tokenizer is None:
        if tokenizer_path != "":
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            raise ValueError("error：tokenizer is none")
        
    return dataset.map(process_sample, remove_columns=dataset['train'].column_names)