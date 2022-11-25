import json
from pathlib import Path
from datasets import load_from_disk, load_dataset
import evaluate
import logging
from utils import setup_logging, compute_metrics, _load_ds
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          TrainingArguments, Trainer, AutoConfig, DataCollatorWithPadding)
from utils import setup_logging, compute_metrics, label2id, id2label
import json
import torch

MODEL_CHECKPOINTS = ["xlm-roberta-base", "ProsusAI/finbert"]
RAW_DATASETS = ["msg_all", "msg_en"]
# DATASET PATHS ARE RELATIVE TO root project dir
CONFIGS = {
    "xlm-roberta-base": {"body": 
                                  {"msg_all": {
                                        "ds_path": "xlm-roberta-base_all_body",
                                        "results": None,
                                        "logits": None,
                                       }, 
                                   "msg_en": {
                                        "ds_path": "xlm-roberta-base_en_body",
                                        "results": None,
                                        "logits": None,
                                       }, 
                                  },
                              "title": 
                                  {"msg_all": {
                                        "ds_path": "xlm-roberta-base_all_title",
                                        "results": None,
                                        "logits": None,
                                       }, 
                                   "msg_en": {
                                        "ds_path": "xlm-roberta-base_en_title",
                                        "results": None,
                                        "logits": None,
                                       }, 
                                  }
                             }, 
    "ProsusAI/finbert": {"body": {
                                   "msg_en": {
                                        "ds_path": "finbert_en_body",
                                        "results": None,
                                        "logits": None,
                                       }
                                  },
                         "title": {
                                   "msg_en": {
                                        "ds_path": "finbert_en_title",
                                        "results": None,
                                        "logits": None,
                                       },
                                  }
                        }
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

def get_base_model(model_variation):
        if "finbert" in model_variation.lower():
            return "ProsusAI/finbert"
        return "xlm-roberta-base"


def checkpoint_save_predictions(result, out_file=PRETRAINED_OUT_FILE):
    logging.info(f"Saving outfile {out_file}")
    with open(out_file, "w") as out_file:
        json.dump(json.dumps(result), out_file, indent=4)

if __name__ == "__main__":
    #with open("empty_results.json", "w") as json_file:
    #    json.dump(json.dumps(CONFIGS), json_file, indent=4) 
    setup_logging("pretrained_eval_results.log", level=logging.INFO)
    chckpt2batch_size = {"xlm-roberta-base": {"title": 256, "body": 64},
                     "ProsusAI/finbert": {"title": 256, "body": 128}
                    }
    data_path = Path("data")
    raw_sets_path = data_path / "sets_new"
    results = None
    with open("empty_results.json", "r") as json_file:
        results = json.loads(json.load(json_file))
    print(results)
    if (existing_results_path:=Path(PRETRAINED_OUT_FILE)).exists():
        logging.info("Loading existing results")
        with open(existing_results_path, "r") as json_file:
                existing_results = json.loads(json.load(json_file))
                # Overriding only xlm_roberta
        results = {**results, **existing_results}
                
        
    for model_checkpoint, fields in results.items():
        logging.info(f"Eval of {model_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model_name = model_checkpoint
        if "/" in model_checkpoint:
            model_name = model_checkpoint.split("/")[-1].lower()
        for field, raw_ds in fields.items():
            #print(raw_ds)
            for ds_name, ds_results in raw_ds.items():
                logging.info("Current model variation:")
                logging.info(f"{model_checkpoint} - {field} - {ds_name}")
                if ds_results["logits"]:
                    logging.info("Already predicted -> Skipping model")
                    continue
                    
                model_base = get_base_model(model_checkpoint)
                if "finbert" in model_checkpoint and ds_name == "msg_all":
                    logging.info(f"Skipping due to irrelevant dataset for model - {ds_name}")
                dataset_path = Path(f"{model_base}_{ds_name.split('_')[-1]}_{field}")
                if not dataset_path.exists():
                    out_path = data_path / f"{model_name}_{ds_name.split('_')[-1]}_{field}"
                    logging.info(f"Preparing dataset from {ds_name}")
                    dataset = prepare_ds(raw_sets_path / ds_name, field, out_path)
                    results[model_checkpoint][field][ds_name]["ds_path"] = str(out_path)
                    checkpoint_save_predictions(results)
                else:
                    logging.info(f"Loading existing dataset: {ds_name}")
                    dataset = load_from_disk(dataset_path)

                logging.info("Creating DataPaddingCollator")
                # Use Data collator for on the fly padding of samples
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

                ## Load Initial model
                logging.info("Loading autoconfig")
                num_labels = len(label2id.keys())
                config = AutoConfig.from_pretrained(model_checkpoint, label2id=label2id, id2label=id2label, num_labels=num_labels)
                logging.info(f"Overriden AutoConfig:\n{config}")

                logging.info("Loading model")
                model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

                metric_names = ["accuracy", "precision", "recall", "f1"]
                metrics = {_metric: evaluate.load(_metric) for _metric in metric_names}

                logging.info("Setup training args")
                batch_size = chckpt2batch_size[model_checkpoint][field]
                training_args = setup_training_args(f"eval_{model_checkpoint}_base_{field}_{ds_name}", batch_size)
                trainer = Trainer(model=model, 
                          args=training_args, 
                          tokenizer=tokenizer,
                          data_collator=data_collator,
                          #  train_dataset=dataset["train"],
                          #  eval_dataset=dataset["val"],

                          compute_metrics=lambda eval_pred:
                              compute_metrics(eval_pred, metrics)
                         )
                logging.info("Predicting")
                logits, label_ids, pred_result = trainer.predict(dataset["test"])
                logging.info("Prediction results:")
                logging.info(pred_result)
                results[model_checkpoint][field][ds_name]["logits"] = logits.tolist()
                results[model_checkpoint][field][ds_name]["label_ids"] = label_ids.tolist()
                results[model_checkpoint][field][ds_name]["results"] = pred_result
                checkpoint_save_predictions(results)
                
                logging.info("#"*15)
        # When changing model_checkpoint
        logging.info("Clearing GPU cache")
        tokenizer = model = None
        torch.cuda.empty_cache()
    logging.info("Complete")
            # TODO: LOAD TRAINED MODEL
            #model.from_pretrained(model_checkpoint/checkpoint_dir_to_best_model)
            

            # All are trained with 50 epochs, doesn't matter
            
                         # Load both best model and pre_trained model, store as dictionary for predictions and result
