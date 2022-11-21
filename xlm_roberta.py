
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset, IterableDatasetDict
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
    
def post_process_ds(ds_dict):
    keep_columns = set(['input_ids', 'attention_mask', "label"])
    processed = {}
    for split, ds in datasets.items():
        if not ds:
            continue
        remove_cols = list(set(ds.column_names).difference(keep_columns))
        processed[split] = (ds.remove_columns(remove_cols)
                            .rename_column("label", "labels")
                            .map(lambda example: {"labels": label2id[example["labels"]]})
                           )
    return processed
        
    

def setup_args():
    parser = argparse.ArgumentParser(description='Script for running experiments with various splits and text fields.')
    parser.print_help()
    parser.add_argument("--txt_field", "-f", type=str, default="title", choices=["title", "body"], help="Field to use for NLP model")
    parser.add_argument("--splits", "-s", type=str, nargs="*", default=["test"], choices=["train", "val", "test"])
    parser.add_argument("--dataset", "-d", type=str, nargs="?", default="msg_en", choices=["msg_en", "msg_no", "msg_all"])
    parser.add_argument("--model_checkpoint", "-m", type=str, default="xlm-roberta-base", choices=["xlm-roberta-base", "xlm-roberta-large",])
    parser.add_argument("--num_epochs", "-n", type=int, default=1)
    parser.add_argument("--batch_size", "-b", type=int, default=256)
    parser.add_argument("--resume_from_checkpoint", "-r", action="store_true")
    
    
    return parser
    #TODO: Extend to finetune bool argument to switch between fine tuning and finbert -- Look into HF ArgParser
    
    
id2label = {0: "positive", 1: "neutral", 2: "negative"}    
label2id = {v:k for k,v in id2label.items()}

if __name__ == "__main__":
    parser = setup_args()
    args = vars(parser.parse_args())
    txt_field, splits, model_checkpoint, NUM_TRAIN_EPOCHS, BATCH_SIZE = args["txt_field"], args["splits"], args["model_checkpoint"], args["num_epochs"], args["batch_size"]
    log_path = f"{model_checkpoint}_{txt_field}_{args['dataset']}_raw.log"
    filemode = "a" if args["resume_from_checkpoint"] and Path(log_path).exists() else "w"
    setup_logging(log_path, encoding='utf-8', level=logging.INFO, filemode=filemode)
    
    logging.info(f"Script args:\n{args}")
    #if (mdl_chkp:=Path(model_checkpoint)).isdir() and (mdl_chckp / "runs").parent== mdl_chckp:
    #    logging.info(f"Loading local model {mdl_ckp}
    
    p_data = Path("data")
    msg_sets_path = p_data / "sets_new"
    logging.info("Loading dataset")
    msg_ds = load_dataset("parquet", 
                             data_dir=msg_sets_path / args["dataset"], 
                             data_files={"test": "test.parquet",
                                         "train": "train.parquet", 
                                        "val": "val.parquet"}
                             )
    logging.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    logging.info(f"Tokenizing dataset on {txt_field}")
    tokenized_ds = pre_process_ds(msg_ds, tokenizer,  
                                        field=txt_field) 
    
    pre_tokenized_field_len = max([len(sample.split(" ")) for sample in tokenized_ds[splits[0]][txt_field]])
    logging.info(f"Longest {txt_field} in {splits[0]} (words): {pre_tokenized_field_len}")
    
    logging.info("Creating DataPaddingCollator")
    # Use Data collator for on the fly padding of samples
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    logging.info("Loading autoconfig")
    num_labels = len(label2id.keys())
    config = AutoConfig.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label, num_labels=num_labels)
    logging.info(f"Overriden AutoConfig:\n{config}")
    
    logging.info("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)
        
    # Split into kwargs for Trainer
    logging.info("Dividing and OHE labels")
    datasets = split_datasets_in_splits(tokenized_ds, splits)
    # One hot encode labels and post_process
    datasets_labelled = post_process_ds(datasets)
    logging.info("Post process features")
    test_ds = datasets_labelled.pop("test_dataset")
    logging.info(test_ds.features)
    metric_names = ["accuracy", "precision", "recall", "f1"]
    metrics = {_metric: evaluate.load(_metric) for _metric in metric_names}
    output_dir = f"{model_checkpoint}_{txt_field}_{args['dataset']}_{NUM_TRAIN_EPOCHS}_{BATCH_SIZE}"
    
    # TRAINING ARGS
    training_args = TrainingArguments(output_dir=output_dir, 
                                      per_device_train_batch_size=BATCH_SIZE,
                                      per_device_eval_batch_size=BATCH_SIZE // 2,
                                      num_train_epochs=NUM_TRAIN_EPOCHS,
                                      weight_decay=0.01,
                                      save_strategy="steps",
                                      save_steps=1000,
                                      save_total_limit=4,
                                      metric_for_best_model="f1",
                                      evaluation_strategy="steps",
                                      eval_steps=500,
                                      load_best_model_at_end=True,
                                      seed=37
                                     )
    logging.info("Setup Trainer")
    trainer = Trainer(model=model, 
                      args=training_args, 
                      tokenizer=tokenizer,
                      data_collator=data_collator,
                      **datasets_labelled,
                      compute_metrics=lambda eval_pred:
                          compute_metrics(eval_pred, metrics)
                     )

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
    test_results = trainer.predict(test_ds)
    logging.info(test_results)
    #logging.info("All results from best model:")
    #trainer.log_metrics("eval", results)
    #trainer.log_metrics("test", test_results)
    
    #with open(f"logs/{output_dir}_history.log", "w") as file:
    #    file.write(trainer.state.history_log)
    logging.info("COMPLETE")
    
