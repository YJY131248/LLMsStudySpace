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
