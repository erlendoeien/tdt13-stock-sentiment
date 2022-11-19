from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from utils import setup_logging


def prepare_msg_df(msg_df, known_symbols):
    """ Filter to only known symbols and """
    msg_known_sym = msg_df[msg_df["symbol"].isin(known_symbols)]
    closing_hours = pd.to_datetime(["16:30:00"]).time[0]
    is_next_day = (msg_known_sym["publishedTime_dt"].dt.time > closing_hours)#.astype(int)
    return msg_known_sym.assign(is_next_day=is_next_day, 
                                next_day=(msg_known_sym["publishedTime_dt"]
                                          .transform(lambda dt_idx: dt_idx.date + pd.offsets.BDay() * (dt_idx.time > closing_hours).astype(int))
                                               )
                               )

def merge_msg_hist(msg_df, hist_df):
    """History is multiindex on ["symbol, date"]"""
    return msg_df.merge(hist_df, left_on=["symbol", "next_day"], right_index=True)

def calculate_ground_truth(msg_hist_df, labels_lower=True, 
                           occurence_col="intra_day_open_close_pct", 
                           compare_to_day=3):
    
    labels = ["positive", "negative", "neutral"]
    if not labels_lower:
        labels = [label.upper() for label in labels]
    occ_col = msg_hist_df[occurence_col]
    compare_to_col = msg_hist_df[f"ewm_std_{compare_to_day}"]
    div_by_col = msg_hist_df[f"ewm_{compare_to_day}"]
    return msg_hist_df.assign(is_positive=((occ_col > compare_to_col.div(div_by_col))
                                           .astype(int)
                                           .replace({0: np.nan, 1: labels[0]})),
                              is_negative=((occ_col < compare_to_col.div(div_by_col))
                                           .astype(int)
                                           .replace({0: np.nan, 1: labels[1]}))
                               )


def label_dataset(msg_hist_df, labels_lower=True):
    prep_df = msg_hist_df.assign(label=np.nan)
    return prep_df.assign(label=(prep_df["label"]
                                 .fillna(prep_df["is_positive"])
                                 .fillna(prep_df["is_negative"])
                                 .fillna("neutral" if labels_lower else "NEUTRAL")).astype(str))

def get_language_df_splits(df):
    """Assuming most swedish, danish and other are in fact norwegian (which is likely the case)
     Also, due to similararities in languages, it is likely that the sentiment can be similar"""
    subset = df[['id', 'title', 'body', 'category', 'issuerId', 'publishedTime', 
       'title_clean', 'title_lang', 'title_lang_score', 
       'par_len', 'par_label', 'par_label_score', 'weighted_scores',
       'weighted_label', 'weighted_label_conf', 'n_paragraphs', 
       'symbol', 'name','publishedTime_dt', 'close', 'open', 'high', 'low', 'volume',
       'is_next_day', 'next_day', 'dol_volume', 'intra_day_high_low_pct', 'intra_day_open_close_pct',
       'gap_pct', 'ewm_3', 'ewm_std_3','d3_avg_pct', 'is_positive', 'is_negative', 'label']
               ]
    return {"msg_en": subset[subset["weighted_label"] == "__label__en"], "msg_all": subset, "msg_no": subset[subset["weighted_label"] != "__label__en"]}

def save_data_splits(dfs, out_dir, val=0.1, test=0.2, random_state=37):
    """Split so languages are roughly equally split"""

    for data_name, df in dfs.items():
        logging.info("Storing:", data_name)
        dataset_path = out_dir / data_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        strat = None
        # Remove languages with only 1 language
        freqs = df["weighted_label"].value_counts()
        X = df[df["weighted_label"].isin(freqs[freqs > 2].index)]
        y = X.pop("label")
        if data_name != "msg_en":
            strat = X["weighted_label"]
        X_t_v, X_test, y_t_v, y_test = train_test_split(X, y, test_size=test, 
                                                        random_state=random_state,
                                                        stratify=strat)
        relative_val_size = val / (val + (1-val-test))
        strat = X_t_v["weighted_label"] if strat is not None else None
        X_train, X_val, y_train, y_val = train_test_split(X_t_v, y_t_v, test_size=relative_val_size,
                                                          random_state=random_state,
                                                          stratify=strat)
        
        pd.concat([X_train, y_train], axis=1).to_parquet(dataset_path / f"train.parquet")
        pd.concat([X_val, y_val], axis=1).to_parquet(dataset_path / f"val.parquet")
        pd.concat([X_test, y_test], axis=1).to_parquet(dataset_path / f"test.parquet")
    
    
if __name__ == "__main__":
    setup_logging("create_dataset.log", level=logging.DEBUG)
    data_path = Path("data")
    hist_path = data_path / "processed_history_new.parquet"
    msg_path = data_path / "msgs_w_issuer.parquet"
    logging.info("Loading messages")
    msg_df = pd.read_parquet(msg_path)
    logging.info("Loading stock history")
    hist_df = pd.read_parquet(hist_path)
    
    logging.info("Before cleaning:")
    logging.info(f"\tNumber of messages: {msg_df.shape[0]}")
    logging.info(f"\tNumber of symbols from messages: {msg_df['symbol'].nunique()}")
    
    
    rel_symbols = hist_df.index.get_level_values(0).unique()
    logging.info(f"\tNumber of symbols from history: {rel_symbols.shape[0]}")
    logging.info("Calculating relevant history for each message")
    prep_msg_df = prepare_msg_df(msg_df, rel_symbols)
    
    logging.info("Merging each message with relevant history")
    merged_df = merge_msg_hist(prep_msg_df, hist_df)
    
    logging.info("After cleaning:")
    logging.info(f"\tNumber of messages: {merged_df.shape[0]}")
    logging.info(f"\tNumber of symbols from messages: {merged_df['symbol'].nunique()}")
    
    logging.info("Labelling dataset into Positive, neutral and negative")
    df_w_ground_truth = calculate_ground_truth(merged_df)
    labelled_df = label_dataset(df_w_ground_truth)
    logging.info(f"Label distribution:\n{labelled_df['label'].value_counts()}")
    
    file_path = "ds_labelled_lower.parquet"
    logging.info(f"Saving labelled dataset {file_path}")
    labelled_df.to_parquet(data_path / file_path)
    logging.info("Splitting and storing datasets into train, test and valid")
    language_dfs = get_language_df_splits(labelled_df)
    save_data_splits(language_dfs, data_path / "sets_new")
    