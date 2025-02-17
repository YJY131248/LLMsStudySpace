import logging
import warnings
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score
from typing import List
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from sklearn.metrics import classification_report
from data_utils import get_alpaca_dataset
from inference import get_peft_llm_model_tokenizer, get_llm_response

# set merge model arguments
@dataclass
class EvalArguments:
    peft_type: str = field(default="lora")
    task_type: str = field(default="classification")
    llm_model_name: str = field(default="Qwen")
    llm_model_path: str = field(default="../model/car_lora_qwen2.5_7b")
    peft_model_path: str = field(default="../out/lora_peft/checkpoint-3000/")
    dataset_path: str = field(default="../data/trainset/sft_dataset.json")
    save_eval_res_path: str = field(default="../out/lora_peft/eval_res/eval_res.csv")
    log_path: str = field(default="../log/model_eval.log")
    use_peft_model: bool = field(default=False)
    max_new_tokens: int = field(default=1024)
    do_sample: bool = field(default=False)
    top_p: float = field(default=0.1)
    temperature: float = field(default=0.1)
    repetition_penalty: float = field(default=1.2)

# Custom compute_metrics function for classification and QA tasks
def compute_metrics(
    preds: List[str],
    labels: List[str],
    task_type: str
):
    # Classification task
    if task_type == "classification":
        # Convert labels  str to int
        preds = [int(pred) for pred in preds]
        labels = [int(label) for label in labels]
        print(classification_report(labels, preds, digits=4))
        report = classification_report(labels, preds, output_dict=True, digits=4)
        return {
            "classification_report": report
        }

    # QA task
    elif task_type == "qa":
        # Filter out empty strings
        preds = [p if p else " " for p in preds]
        refs = [[r if r else " "] for r in labels]

        # BLEU Score Calculation
        bleu_score = corpus_bleu(refs, preds)

        # ROUGE Score Calculation
        rough_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        for r, p in zip(refs, preds):
            scores = rough_scorer.score(r[0], p)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        rouge1_score = np.mean(rouge_scores['rouge1'])
        rouge2_score = np.mean(rouge_scores['rouge2'])
        rougeL_score = np.mean(rouge_scores['rougeL'])

        # METEOR Score Calculation
        meteor_scores = [meteor_score([r[0].split()], p.split()) for r, p in zip(refs, preds)]
        meteor_score_mean = np.mean(meteor_scores)

        # BERTScore Calculation
        P, R, F1 = score(preds, [r[0] for r in refs], model_type="../model/roberta-large", num_layers=17, lang="en")
        bert_score_precision = np.mean(P.numpy())
        bert_score_recall = np.mean(R.numpy())
        bert_score_f1 = np.mean(F1.numpy())

        return {
            "bleu": bleu_score / 100,
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougeL": rougeL_score,
            "meteor": meteor_score_mean,
            "bert_score_precision": bert_score_precision,
            "bert_score_recall": bert_score_recall,
            "bert_score_f1": bert_score_f1,
        }

    else:
        raise ValueError("Task type must be either 'classification' or 'qa'")

def main():
    # ignore warnings
    warnings.filterwarnings("ignore")

    # load arguments
    eval_args = HfArgumentParser(
        (EvalArguments)
    ).parse_args_into_dataclasses()[0]

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        filename=eval_args.log_path,
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s'
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Arguments: ")
    logger.info("eval_args:")
    logger.info(eval_args.__repr__())

    # load the dataset and tokenizer dataset
    dataset = get_alpaca_dataset(eval_args.dataset_path, test_size=0.1)
    dataset = dataset['test']
    logger.info('dataset build successfully!')

    # get llm resp
    # load model
    llm_model, llm_tokenizer = get_peft_llm_model_tokenizer(
        eval_args.llm_model_name,
        eval_args.llm_model_path,
        eval_args.peft_model_path,
        eval_args.peft_type,
        eval_args.use_peft_model
    )
    logger.info('LLMs {} load successfully! LLM path::: {}'.format(eval_args.llm_model_name, eval_args.llm_model_path))

    # get llm resp
    query_list = [sample['instruction'] for sample in dataset]
    llm_response_mp = get_llm_response(
        query_list=query_list,
        model = llm_model,
        tokenizer = llm_tokenizer,
        max_new_tokens=eval_args.max_new_tokens,
        top_p=eval_args.top_p,
        temperature=eval_args.temperature,
        repetition_penalty=eval_args.repetition_penalty,
        do_sample=eval_args.do_sample
    )
    preds = [llm_response_mp[query] for query in query_list]
    labels = [sample['output'] for sample in dataset]
    logger.info('LLMs {} get response successfully!'.format(eval_args.llm_model_name))
    logger.info('pred::: {}\nlabels::: {}'.format(preds, labels))
    
    # save preds and labels
    with open(eval_args.save_eval_res_path, 'w') as f:
        f.write('pred,label\n')
        for pred, label in zip(preds, labels):
            f.write(pred + ',' + label + '\n')
    
    # eval
    eval_metrics = compute_metrics(
        preds=preds,
        labels=labels,
        task_type=eval_args.task_type,
    )
    print(eval_metrics)
    logger.info('eval metrics: {}'.format(eval_metrics))
    logger.info('eval done!')

if __name__ == "__main__":
    main()