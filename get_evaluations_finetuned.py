import json
from pathlib import Path
from datasets import load_from_disk, load_dataset
import evaluate
import logging
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          BertForSequenceClassification,
                          XLMRobertaForSequenceClassification,
                          TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding)
from utils import setup_logging, compute_metrics, label2id, id2label, _load_ds
import torch
import numpy as np

MODEL_CHECKPOINTS = ["xlm-roberta-base", "ProsusAI/finbert"]
RAW_DATASETS = ["msg_all", "msg_en"]
# DATASET PATHS ARE RELATIVE TO root project dir
CONFIGS = {"model_checkpoints": {
    "xlm-roberta-base": {"body": 
                                  {"msg_all": {
                                        "ds_path": None,
                                        "results": None,
                                        "logits": None,
                                       }, 
                                   "msg_en": {
                                        "ds_path": None,
                                        "results": None,
                                        "logits": None,
                                       }, 
                                  },
                              "title": 
                                  {"msg_all": {
                                        "ds_path": None,
                                        "results": None,
                                        "logits": None,
                                       }, 
                                   "msg_en": {
                                        "ds_path": None,
                                        "results": None,
                                        "logits": None,
                                       }, 
                                  }
                             } 
                        },
    "ProsusAI/finbert": {"body": 
                                  {"msg_all": {
                                        "ds_path": None,
                                        "results": None,
                                        "logits": None,
                                       }, 
                                   "msg_en": {
                                        "ds_path": None,
                                        "results": None,
                                        "logits": None,
                                       }, 
                                  },
                         "title": 
                                  {"msg_all": {
                                        "ds_path": None,
                                        "results": None,
                                        "logits": None,
                                       }, 
                                   "msg_en": {
                                        "ds_path": None,
                                        "results": None,
                                        "logits": None,
                                       }, 
                                  }
                        },
          }

def prepare_ds(ds_path, txt_field, out_path):
    logging.info("\t>> Loading new dataset")
    raw_ds = _load_ds(ds_path)


    logging.info(f"\t>> Tokenizing dataset on {txt_field}")
    tokenized_ds = pre_process_ds(raw_ds, tokenizer,  
                                        field=txt_field) 

    tokenized_field_len = max([len(sample.split(" ")) for sample in tokenized_ds["test"][txt_field]])
    logging.info(f"\t>> Longest {txt_field} in test (tokens): {tokenized_field_len}")

    logging.info("\t>> Removing columns and OHE labels")
    # One hot encode labels and post_process
    datasets = post_process_ds(tokenized_ds)
    logging.info(f"\t>> Store dataset to {out_path}")
    datasets.save_to_disk(out_path)
    return datasets

def pre_process_ds(ds, _tokenizer, field="body", **kwargs):
    """Remove empty bodies and call tokenizer"""
    clean_ds = ds.filter(lambda example: example["weighted_label"] != "None")#, batched=True)
    batch = True if field == "title" else False
    truncation = not batch
    return clean_ds.map(lambda example: 
                             _tokenizer(example[field], 
                                        truncation=truncation,
                                        **kwargs),
                            batched=batch,
                            )

def post_process_ds(ds):
    """Rename, map labels and remove columns"""
    keep_columns = set(['input_ids', 'attention_mask', "label"])
    remove_cols = list(set(ds["train"].column_names).difference(keep_columns))
    return (ds.rename_column("label", "labels")
            .map(lambda example: {"labels": label2id[example["labels"]]}, 
                 remove_columns=remove_cols)
                           )

def setup_training_args(output_dir, batch_size, **kwargs):
    default_kwargs = {"weight_decay": 0.01,
                      "save_strategy": "steps",
                      "save_steps": 50,
                      "save_total_limit": 4,
                      "metric_for_best_model": "f1",
                      "evaluation_strategy": "steps",
                      "fp16": True,
                      "eval_steps": 50,
                      "load_best_model_at_end": True,
                      "seed":37
                     }
    return TrainingArguments(output_dir=output_dir, 
                              per_device_train_batch_size=batch_size,
                              per_device_eval_batch_size=batch_size,
                              **default_kwargs,
                             **kwargs
                                 )
PRETRAINED_OUT_FILE = "pretrained_eval_results.json"
FINETUNED_OUT_FILE = "finetuned_eval_results.json"

# Fin, body, en, 50, 128
# Fin, title, en, 36, 256
# Xlm, body, all, 50, 64
# Xlm, body, en, 50, 64
# xlm, title, all, 50, 256, 
# xlm, title, en, 50, 256
FINETUNED_CHECKPOINTS = ["finbert_body_finbert_en_body_50_128/checkpoint-11800",
                         "finbert_title_finbert_en_title_36_256/checkpoint-700",
                         "xlm-roberta-base_body_msg_en_50_64/checkpoint-50000",
                         "xlm-roberta-base_body_msg_all_50_64/checkpoint-1000",
                         "xlm-roberta-base_title_msg_all_50/checkpoint-19000",
                         "xlm-roberta-base_title_msg_en_50_256/checkpoint-1000"
                        ]

def get_chckpt_map():
    chckpt2base_chckpt= {}
    for chckpt in FINETUNED_CHECKPOINTS:
        if "finbert" in chckpt:
            chckpt2base_chckpt[chckpt] = "ProsusAI/finbert"
        else:
            chckpt2base_chckpt[chckpt] = "xlm-roberta-base"
    return chckpt2base_chckpt

def checkpoint_save_predictions(result, out_file=PRETRAINED_OUT_FILE):
    logging.info(f"Saving outfile {out_file}")
    with open(out_file, "w") as out_file:
        json.dump(json.dumps(result), out_file, indent=4)
        
def get_log_history(model_variation):
    checkpoints_path = Path(model_variation)
    assert checkpoints_path.exists()
    # Only iterating checkpoint dirs
    checkpoints_paths = list(checkpoints_path.glob("checkpoint-*"))
    latest_idx = np.argmax(([int(checkpoint.name.split("-")[-1]) for checkpoint in checkpoints_paths
                  if checkpoint.is_dir()]))
    latest = checkpoints_paths[latest_idx]
    logging.info(f"Fetching training history from latest chckpt: {latest}")
    with open(latest / "trainer_state.json", "r") as json_file:
        return json.load(json_file)["log_history"]
    

if __name__ == "__main__":
    #with open("empty_results.json", "w") as json_file:
    #    json.dump(json.dumps(CONFIGS), json_file, indent=4) 
    setup_logging("finetuned_eval_results.log", level=logging.INFO)
    chckpt2batch_size = {"xlm-roberta-base": {"title": 256, "body": 64},
                     "ProsusAI/finbert": {"title": 256, "body": 128}
                    }
    data_path = Path("data")
    checkpoint_map = get_chckpt_map()
    for model_checkpoint in FINETUNED_CHECKPOINTS:
        model_variation, chckpt_dir = model_checkpoint.split("/")
        model_base = checkpoint_map[model_checkpoint]
        dataset_lang = "en" if "en" in  model_variation else "all"
        dataset_field = "body" if "body" in model_variation else "title"
        out_file=f"eval_finetune_{model_variation}_{chckpt_dir}.json"
        model_type = model_variation.split("_")[0]
        logging.info(f"Currently evaluating {model_type} on {dataset_lang} {dataset_field}, chkpt: {chckpt_dir.split('-')[-1]}")
        dataset_name = f"{model_type}_{dataset_lang}_{dataset_field}"
        logging.info(f"Loading dataset {dataset_name}")
        dataset = load_from_disk(data_path / dataset_name)
        
        # loading model base defaults
        tokenizer = AutoTokenizer.from_pretrained(model_base)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        ## Load Initial model
        logging.info("Loading autoconfig")
        num_labels = len(label2id.keys())
        config = AutoConfig.from_pretrained(model_base, label2id=label2id, id2label=id2label, num_labels=num_labels)
        logging.info(f"Overriden AutoConfig:\n{config}")

        logging.info("Loading best performed model")
        #model_class = BertForSequenceClassification if "finbert" in model_variation else XLMRobertaForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

        metric_names = ["accuracy", "precision", "recall", "f1"]
        metrics = {_metric: evaluate.load(_metric) for _metric in metric_names}

        logging.info("Setup training args")
        batch_size = chckpt2batch_size[model_base][dataset_field]
        training_args = setup_training_args(f"eval_{model_checkpoint}_base_{dataset_field}_{dataset_lang}", batch_size)
        trainer = Trainer(model=model, 
                  args=training_args, 
                  tokenizer=tokenizer,
                  data_collator=data_collator,
                  compute_metrics=lambda eval_pred:
                      compute_metrics(eval_pred, metrics)
                 )
        logging.info("Predicting")
        logits, label_ids, pred_result = trainer.predict(dataset["test"])
        logging.info("Prediction results:")
        logging.info(pred_result)
        result = {"logits": logits.tolist(), 
                  "label_ids": label_ids.tolist(), 
                  "results": pred_result,
                  "log_history": get_log_history(model_variation)
                 }
        checkpoint_save_predictions(result, out_file=out_file)
        
        logging.info("Clearing GPU cache")
        model = None
        torch.cuda.empty_cache()
    logging.info("Complete")     
        
        
