from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

from sklearn import metrics
from collections import defaultdict
from time import time
from pathlib import Path
import numpy as np
import pandas as pd
from time import time

import logging
from utils import setup_logging

def fit(km, X_train, X_test, y, test=True, name=None, n_runs=5):
    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X_train)
        train_times.append(time() - t0)
        preds = km.predict(X_test) if test else km.labels_

        scores["Homogeneity"].append(metrics.homogeneity_score(y, preds))
        scores["Completeness"].append(metrics.completeness_score(y, preds))
        scores["V-measure"].append(metrics.v_measure_score(y, preds))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(y, preds)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(y.reshape(-1,1), preds, sample_size=20000)
        )
    train_times = np.asarray(train_times)
    logging.info(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    return scores, train_times

def print_eval(scores, train_times, name=None):
    evaluations = []
    evaluations_std = []

    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        logging.info(f"\t{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score
    evaluations.append(evaluation)
    evaluations_std.append(evaluation_std)
    return evaluations, evaluation_std

def fit_and_evaluate(km, X_train, y, X_test=None, test=True, name=None, n_runs=5):
    name = km.__class__.__name__ if name is None else name
    scores, train_times = fit(km, X_train, X_test, y, test, name, n_runs)
    return print_eval(scores, train_times, name)


if __name__ == "__main__":
    setup_logging("k_means.log")
    data_path = Path("data/sets_new/msg_en")
    # Loading datasets
    train, val, test = (pd.read_parquet(data_path / "train.parquet"), 
                        pd.read_parquet(data_path / "val.parquet"), 
                        pd.read_parquet(data_path / "test.parquet"))
    logging.info(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    le = LabelEncoder()
    X_train, y_train_raw = train["body"], train["label"]
    y_train = le.fit_transform(y_train_raw) # all labels are in train
    logging.info(np.unique(y_train, return_counts=True))
    X_test, y_test = test["body"], le.transform(test["label"])
    logging.info(np.unique(y_test, return_counts=True))
    
    ### HYPER PARAMETERS
    svd_hp = {
        "n_components": 1000,
        "random_state":37
    }

    tfidf_hp = {
                "max_df": 0.5,
                "min_df": 5,
                "stop_words": "english"
    }
    hashing_hp = {
         "stop_words": "english",
        "n_features": 100_000
    }
    mini_batch_hp = {
            "n_clusters": 3,
            "n_init": 10,
            "init_size": 1000,
            "batch_size": 1000
    }
        
    
    logging.info(f"HyperParameters:\n\t\tSVD:{svd_hp}\n\t\tTFiDF:{tfidf_hp}\n\t\tHasing: {hashing_hp}")
    logging.info(f"MiniBatchKM:{mini_batch_hp}")
    
    logging.info("LSA PIPELINE")
    lsa = make_pipeline(
                    TfidfVectorizer(**tfidf_hp),
                    TruncatedSVD(**svd_hp),
                    Normalizer(copy=False))
    
    logging.info("LSA fit & transform")
    t0 = time()
    X_lsa_train = lsa.fit_transform(X_train)

    explained_variance = lsa[1].explained_variance_ratio_.sum()
    logging.info(f"LSA done in {time() - t0:.3f} s")
    logging.info(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")
    
    X_lsa_test = lsa.transform(X_test)
    
    minibatch_kmeans = MiniBatchKMeans(**mini_batch_hp)
    
    mini_evals = fit_and_evaluate(
                minibatch_kmeans,
                X_lsa_train,
                y_test,
                X_test=X_lsa_test,
                test=True,
                name="MiniBatchKMeans\nwith LSA on tf-idf vectors",
    )
    logging.info("################################")
    
    hashing_lsa = lsa_vectorizer = make_pipeline(
                        HashingVectorizer(**hashing_hp),
                        TfidfTransformer(),
                        TruncatedSVD(**svd_hp),
                        Normalizer(copy=False),
                    )
    logging.info("HASHING PIPELINE")
    t0 = time()
    X_hashed_train = hashing_lsa.fit_transform(X_train)
    logging.info(f"Hashing pipeline done in {time() - t0:.3f} s")
    logging.info("transforming X_test")
    X_hashed_test = hashing_lsa.transform(X_test)
    
    hashed_mini_evals = fit_and_evaluate(
                minibatch_kmeans,
                X_hashed_train,
                y_test,
                X_test=X_hashed_test,
                test=True,
                name="MiniBatchKMeans\nwith LSA on tf-idf vectors",
    )
    
    logging.info("Example test with minibatch LSA")
    
    minibatch_kmeans.set_params(random_state=42)
    minibatch_kmeans.fit(X_lsa_train)
    preds = minibatch_kmeans.predict(X_lsa_test)
    logging.info(f"Report:\n{metrics.classification_report(y_test, preds,target_names=list(le.classes_))}")
    cm = metrics.confusion_matrix(y_test, preds)#, labels=le.classes_)
    cm_disp = metrics.ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    
    logging.info(f"Confusion matrix:\n{cm}")
    logging.info("Storing Confusion Matrix")
    cm_disp.plot()
    plt.savefig("k-means_cm.svg")
    