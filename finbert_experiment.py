import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import evaluate
import logging
import argparse


def pre_process_ds(ds, _tokenizer, field="body", **kwargs):
    def to_lower_w_len(example):
        return {"body_length": len(example["body"].split()),
                "title_length": len(example["title"].split()),
                #"label": example["label"].lower()
               }
    
    clean_ds = ds.filter(lambda example: example["weighted_label"] != "None")#, batched=True)
    pre_processed = clean_ds.map(to_lower_w_len)#, batched=True)
    batch = True if field == "title" else False
    truncation = not batch
    return pre_processed.map(lambda example: 
                             _tokenizer(example[field], 
                                        truncation=truncation,
                                        **kwargs),
                            batched=batch,
                            )
    
def label2ids(example, mapping):
    return {"label": mapping[example["label"]] }
    
def compute_metrics(eval_pred, _metrics):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    results = {}
    for name, metric in _metrics.items():
        if name == "accuracy":
            results[name] = metric.compute(predictions=predictions, references=labels)[name]
        else:
            results[name] = metric.compute(predictions=predictions, references=labels, average="weighted")[name]
    return results

def split_datasets_in_splits(ds, splits):
    datasets = {"eval_dataset":None, "train_dataset": None, "test_dataset":None}
    if len(splits) == 1:
        datasets["eval_dataset"] = tokenized_ds[splits[0]]
    else:
        for split in splits:
            if split == "val":
                datasets[f"e{split}_dataset"] = tokenized_ds[split]
            else:
                datasets[f"{split}_dataset"] = tokenized_ds[split]
    return datasets
    

def setup_args(parser):
    parser.add_argument("--txt_field", "-f", type=str, default="title", choices=["title", "body"], help="Field to use for NLP model")
    parser.add_argument("--splits", "-s", type=str, nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--dataset", "-d", type=str, nargs="?", default="msg_en", choices=["msg_en", "msg_no", "msg_all"])
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for running FinBert with various splits and text fields.')
    setup_args(parser)
    parser.print_help()
    args = vars(parser.parse_args())
    txt_field, splits = args["txt_field"], args["splits"]
    logging.basicConfig(filename=f"finbert_{txt_field}_raw.log",encoding='utf-8', level=logging.INFO, filemode="w")
    
    logging.info(f"Script args:\n{args}")
    
    
    p_data = Path("data")
    msg_sets_path = p_data / "sets_new"
    logging.info("Loading dataset")
    msg_en_test_ds = load_dataset("parquet", 
                             data_dir=msg_sets_path / args["dataset"], 
                             data_files={"test": "test.parquet",
                                         "train": "train.parquet", 
                                        "val": "val.parquet"}
                             )
                                  
    model_checkpoint = "ProsusAI/finbert"
    
    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    logging.info(msg_en_test_ds)
    
    logging.info(f"Tokenizing dataset on {txt_field}")
    tokenized_ds = pre_process_ds(msg_en_test_ds, tokenizer,  
                                        field=txt_field, 
                                        padding="max_length")
    logging.info(tokenized_ds)
    pre_tokenized_field_len = max([len(sample.split(" ")) for sample in tokenized_ds[splits[0]][txt_field]])
    logging.info(f"Longest {txt_field} in {splits[0]} (words): {pre_tokenized_field_len}")
                 
    logging.info("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        
    # Split into kwargs for Trainer
    datasets = split_datasets_in_splits(tokenized_ds, splits)
    logging.info("Converting labels")
    datasets_labelled = {k: v.map(lambda example: label2ids(example, model.bert.config.label2id)) for k,v in datasets.items() if v}
    metric_names = ["accuracy", "precision", "recall", "f1"]
    metrics = {_metric: evaluate.load(_metric) for _metric in metric_names}
    training_args = TrainingArguments(output_dir=f"finbert_{txt_field}", 
                                      per_device_eval_batch_size=256,
                                     )
    trainer = Trainer(model=model, args=training_args, 
                      **datasets_labelled,
                      compute_metrics=lambda eval_pred:
                          compute_metrics(eval_pred, metrics)
                     )
    
    logging.info("Predicting on test")
    #logging.info(trainer.evaluate())
    results = trainer.evaluate()
    logging.info(results)
    logging.info("COMPLETE")
    #print("RESULTS GOTTEN")
    #logging.info(results)
    
    