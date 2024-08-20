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
