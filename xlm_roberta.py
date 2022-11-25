import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding
import evaluate
import logging
import argparse
from utils import setup_logging, compute_metrics


def pre_process_ds(ds, _tokenizer, field="body", **kwargs):
    clean_ds = ds.filter(lambda example: example["weighted_label"] != "None")#, batched=True)
    batch = True if field == "title" else False
    truncation = not batch
    return clean_ds.map(lambda example: 
                             _tokenizer(example[field], 
                                        truncation=truncation,
                                        **kwargs),
                            batched=batch,
                            )
    
    
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
    
def post_process_ds(ds):
    """Rename, map labels and remove columns"""
    keep_columns = set(['input_ids', 'attention_mask', "label"])
    processed = {}
    remove_cols = list(set(ds["train"].column_names).difference(keep_columns))
    return (ds.rename_column("label", "labels")
            .map(lambda example: {"labels": label2id[example["labels"]]}, 
                 remove_columns=remove_cols)
                           )
        
def _load_ds(ds_path):
    """Loads parqet dataset files, already split"""
    return load_dataset("parquet", 
                     data_dir=ds_path,
                     data_files={"test": "test.parquet",
                                 "train": "train.parquet", 
                                "val": "val.parquet"}
                     )   

def setup_args():
    parser = argparse.ArgumentParser(description='Script for running experiments with various splits and text fields.')
    parser.print_help()
    parser.add_argument("--txt_field", "-f", type=str, default="title", choices=["title", "body"], help="Field to use for NLP model")
    parser.add_argument("--splits", "-s", type=str, nargs="*", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--raw_dataset", type=str, nargs="?", choices=["msg_en", "msg_all", "msg_no"])
    parser.add_argument("--dataset", "-d", type=str, nargs="?")
    parser.add_argument("--model_checkpoint", "-m", type=str, default="xlm-roberta-base", 
                        choices=["xlm-roberta-base", "xlm-roberta-large", "ProsusAI/finbert"])
    parser.add_argument("--num_epochs", "-n", type=int, default=1)
    parser.add_argument("--batch_size", "-b", type=int, default=256)
    parser.add_argument("--resume_from_checkpoint", "-r", action="store_true")
    
    return parser
    
    
id2label = {0: "positive", 1: "neutral", 2: "negative"}    
label2id = {v:k for k,v in id2label.items()}

if __name__ == "__main__":
    parser = setup_args()
    args = vars(parser.parse_args())
    txt_field, splits, model_checkpoint, NUM_TRAIN_EPOCHS, BATCH_SIZE = (args["txt_field"], 
                                                                         args["splits"], 
                                                                         args["model_checkpoint"], 
                                                                         args["num_epochs"], args["batch_size"])
    model_name = Path(model_checkpoint).name
    raw_dataset_path, dataset_path = args["raw_dataset"], args["dataset"]
    log_path = f"{model_name}_{txt_field}_{args['dataset']}_raw.log"
    filemode = "a" if args["resume_from_checkpoint"] and Path(log_path).exists() else "w"
    setup_logging(log_path, encoding='utf-8', level=logging.INFO, filemode=filemode)
    
    logging.info(f"Script args:\n{args}")
    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    p_data = Path("data")
    ds_path= p_data / dataset_path
    if not raw_dataset_path and ds_path.exists():
        logging.info(f"Loading existing dataset {ds_path}")
        datasets = load_from_disk(str(ds_path))
    else:
        msg_sets_path = p_data / "sets_new"
        logging.info("Loading new dataset")

        msg_ds = _load_ds(msg_sets_path / raw_dataset_path)


        logging.info(f"Tokenizing dataset on {txt_field}")
        tokenized_ds = pre_process_ds(msg_ds, tokenizer,  
                                            field=txt_field) 

        pre_tokenized_field_len = max([len(sample.split(" ")) for sample in tokenized_ds[splits[0]][txt_field]])
        logging.info(f"Longest {txt_field} in {splits[0]} (words): {pre_tokenized_field_len}")


        logging.info("Removing columns and OHE labels")
        # One hot encode labels and post_process
        datasets = post_process_ds(tokenized_ds)
        logging.info(f"Store dataset to {ds_path}")
        datasets.save_to_disk(ds_path)

    logging.info("Creating DataPaddingCollator")
    # Use Data collator for on the fly padding of samples
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logging.info("Loading autoconfig")
    num_labels = len(label2id.keys())
    config = AutoConfig.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label, num_labels=num_labels)
    logging.info(f"Overriden AutoConfig:\n{config}")

    logging.info("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
        
    metric_names = ["accuracy", "precision", "recall", "f1"]
    metrics = {_metric: evaluate.load(_metric) for _metric in metric_names}
     
    output_dir = f"{model_name}_{txt_field}_{dataset_path}_{NUM_TRAIN_EPOCHS}_{BATCH_SIZE}"
    
    # TRAINING ARGS
    training_args = TrainingArguments(output_dir=output_dir, 
                                      per_device_train_batch_size=BATCH_SIZE,
                                      per_device_eval_batch_size=BATCH_SIZE,
                                      num_train_epochs=NUM_TRAIN_EPOCHS,
                                      weight_decay=0.01,
                                      save_strategy="steps",
                                      save_steps=50,
                                      save_total_limit=4,
                                      metric_for_best_model="f1",
                                      evaluation_strategy="steps",
                                      fp16=True,
                                      eval_steps=50,
                                      load_best_model_at_end=True,
                                      push_to_hub=True,
                                      seed=37
                                     )
    logging.info("Setup Trainer")
    trainer = Trainer(model=model, 
                      args=training_args, 
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                        train_dataset=datasets["train"],
                        eval_dataset=datasets["val"],
                      
                      compute_metrics=lambda eval_pred:
                          compute_metrics(eval_pred, metrics)
                     )

    
    logging.info("Initial results:")
    init_res = trainer.predict(datasets["test"])
    logging.info(init_res)
    
    if args["resume_from_checkpoint"]:
        logging.info("Loading local model and resuming training")
        trainer.train(resume_from_checkpoint=True)
    else:
        logging.info("STARTING TRAINING from scratch")
        trainer.train()
    
    logging.info("Eval results on best model")
    results = trainer.evaluate()
    logging.info(results)
    
    logging.info("Predicting on test")
    test_results = trainer.predict(datasets["test"])
    logging.info(test_results)
    
    logging.info("COMPLETE")
    
