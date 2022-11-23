import logging 
import numpy as np
from datasets import load_dataset

    
id2label = {0: "positive", 1: "neutral", 2: "negative"}    
label2id = {v:k for k,v in id2label.items()}


def setup_logging(log_file, **kwargs):
    base_kwargs = {"format":'%(asctime)s | %(levelname)s: %(message)s',
                   "level":logging.DEBUG, 
                   "filemode":"w",}
    new_kwargs = {**base_kwargs, **kwargs}
    logging.basicConfig(filename=log_file, **new_kwargs)
    
    
    
def compute_metrics(eval_pred, _metrics):
    """Inputs are an EvalPredObject for Huggingface and a dictionary
    of metrics loaded from `evaluate` with names as the ones they are loaded as."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    results = {}
    for name, metric in _metrics.items():
        if name == "accuracy":
            results[name] = metric.compute(predictions=predictions, references=labels)[name]
        else:
            results[name] = metric.compute(predictions=predictions, references=labels, average="weighted")[name]
    return results

def _load_ds(ds_path):
    """Loads parqet dataset files, already split"""
    return load_dataset("parquet", 
                     data_dir=ds_path,
                     data_files={"test": "test.parquet",
                                 "train": "train.parquet", 
                                "val": "val.parquet"}
                     )  