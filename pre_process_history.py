from pathlib import Path
import pandas as pd
import numpy as np
import logging
from utils import setup_logging

def load_clean(path):
    df = pd.read_csv(path, header=1, parse_dates=["Date"], thousands=",", index_col="Date",
                     converters={k:lambda col: float(col[3:].replace(",", "")) if "kr" in col else np.nan for k in DAILY_COLS},
                     dtype={"Volume": float})#.set_index("Date")
    df.columns = df.columns.str.lower()
    df["symbol"] = path.stem
    return df

def extract_intraday_feats(df):
    """ Extracts Price action and volume moving averages.
    Percentages are not multipled by 100."""
    
    other_vals = {"dol_volume": df[daily_cols].mean(axis="columns").mul(df["volume"], axis=0),
                "intra_day_high_low_pct": (df["high"] - df["low"]).divide(df["low"]),#*100,
              "intra_day_open_close_pct": (df["close"] - df["open"]).divide(df["open"]),#*100,
              "gap_pct": (df["close"] - df["open"].shift(1)).divide(df["open"].shift(1)),#*100
             }
    return df.assign(**other_vals)

def extract_rolling_feats(df):
    """Extracting moving features based on dollar volume and daily prices.
    Currently it uses the average daily price as a compromise."""
    action_mean = df[daily_cols].mean(axis="columns")
    
    # Exponential moving averages of prices
    ewms = {f"ewm_{k}": action_mean.ewm(span=k).mean() for k in rolling_vals}
    ewms_std = {f"ewm_std_{k}": action_mean.ewm(span=k).std() for k in rolling_vals}
    
    # EMA for dollar volume
    dol_vol_ewm = {f"dol_vol_ewm_{k}": df["volume"].ewm(span=k).mean() for k in rolling_vals}
    dol_vol_ewm_std = {f"dol_vol_ewm_std_{k}": df["volume"].ewm(span=k).std() for k in rolling_vals}
    return df.assign(**{**ewms, **ewms_std, **dol_vol_ewm, **dol_vol_ewm_std})

def extract_future_features(df):
    """Calculate periodic change in percentage for avg daily price (as decimal).
    Consistent with rolling, so if average is moved to open/close, so should the other.
    """
    pct_changes = {f"d{k}_avg_pct": df["close"].pct_change(periods=k) for k in rolling_vals}
    return df.assign(**pct_changes)

def _process_df(df):
    """Process each df"""
    logging.debug("\tExtracting intra day features")
    intra_feats = extract_intraday_feats(df)
    logging.debug("\tExtracting rolling features")
    rolling_feats = extract_rolling_feats(intra_feats)
    logging.debug("\tExtracting future price features")
    future_df = extract_future_features(rolling_feats)
    return future_df.set_index(["symbol", future_df.index])

def process_dfs(dfs_dir):
    logging.info("Processing all dfs")
    dfs = []
    BLACKLIST = []
    for idx, hist_path in enumerate(list(stock_history_path.glob("*.csv")), start=1):
        try: 
            if hist_path.stem in BLACKLIST:
                continue
            logging.info(f"\tREADING {idx}: {hist_path.stem}")
            df = load_clean(hist_path)
            dfs.append(_process_df(df))
        except pd.errors.ParserError as e:
            if "header=1" in str(e):
                logging.warning(f"New blacklist: {hist_path.stem}")
                BLACKLIST.append(hist_path.stem)
            else:
                logging.error(str(e))
    return pd.concat(dfs)

    
    

if __name__ == "__main__":
    setup_logging("preprocess_history.log", level=logging.INFO)
    DAILY_COLS = ["Close", "Open", "High", "Low"]
    daily_cols = [x.lower() for x in DAILY_COLS]
    rolling_vals = [3, 7, 15, 30]
    stock_history_path = Path("data/stock_history")
    complete_df = process_dfs(stock_history_path)
    logging.info("Saving processed dfs to one parquet file")
    complete_df.to_parquet("data/processed_history_new.parquet")
    