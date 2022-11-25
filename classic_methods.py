from pathlib import Path
import logging
from utils import setup_logging, compute_metrics
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV#, RandomizedSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import xgboost as xgb



def setup_data(train, test, encoder, field):
    """Refitting label encoder every time"""
    X_train, y_train_raw = train[field], train["label"]
    y_train = encoder.fit_transform(y_train_raw) # all labels are in train
    X_test, y_test = test[field], encoder.transform(test["label"])
    return X_train, y_train, X_test, y_test

def tfidf(X_train, X_test, vectorizer):
    """Refitting vectorizer on each call"""
    X_train_tfidf = vectorizer.fit_transform(X_train)
    return X_train_tfidf, vectorizer.transform(X_test)


def load_datasets(data_path):
    return (pd.read_parquet(data_path / "train.parquet"), 
                    pd.read_parquet(data_path / "val.parquet"), 
                    pd.read_parquet(data_path / "test.parquet"))

def load_sentiment_words(sentiment_csv_path):
    """Loading Mcdonald wordlist and returning lower case list of sentiment words"""
    # 2345 negative
    # 347 positive
    # 903 litigious
    # 19 strong_modal
    # 27 weak_modal
    # 184 constraining
    word_dict = pd.read_csv(sentiment_csv_path)
    # Getting rows which are sentiment words
    sentiment_words = word_dict[word_dict.iloc[:, 7:14].ne(0).any(axis=1)]
    return sentiment_words["Word"].str.lower().values

def sentiment2class(word, sentiment_dict):
        pass

# HYPER PARAMETERS
SVC_PARAMS = {"kernel": ["linear", "poly", "rbf"], "C":[0.001, 0.01, 0.1, 1, 10], "degree":[2,3,4,5]}
CNB_PARAMS = {"alpha": [0.5, 1, 1.5,], "norm": [True, False]}
XGB_PARAMS = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5, 1], 
          "gamma": [0.1, 0.5, 1], 
          "reg_alpha": [0.1, 0.5, 1], 
          "reg_lambda": [0.1, 0.5, 1], 
          "subsample": [0.5, 0.7, 0.9],
         "colsample_bytree": [0.5, 0.7, 0.9]}

# GLOBAL CONFIG
METRICS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
RANDOM_STATE = 37


if __name__ == "__main__":
    setup_logging("classic_methods.log", level=logging.INFO)
    data_path = Path("data/sets_new/msg_en")
    logging.info("Loading train, val and test datasets")
    train, val, test = load_datasets(data_path)
    
    CLASSIFIERS = {"SVM": (SVC(class_weight="balanced", random_state=RANDOM_STATE), SVC_PARAMS)} 
               #"CNB": (ComplementNB(), CNB_PARAMS)}
    #CLASSIFIERS = { "XGB": (xgb.XGBClassifier(random_state=RANDOM_STATE), XGB_PARAMS)}
    
    logging.info("Loading sentiment dictionary")
    vocabulary = load_sentiment_words("data/Loughran-McDonald_MasterDictionary_1993-2021.csv")
    
    
    le = LabelEncoder()
    tfidf_vectorizer = TfidfVectorizer(
        # no bigrams
        #ngram_range=(1,2), # Include bigrams - Not sure if works with unigram dictionary
        vocabulary = vocabulary,
        min_df = 5
    )
    
    for field in ["title", "body"]:
        logging.info(f"Setup data for {field}")
        logging.info("Loading both train and validation set for training set")
        X_train, y_train, X_test, y_test = setup_data(pd.concat([train, val]), test, le, field)
        logging.info("Tfidf vectorizing")
        X_train_vec, X_test_vec = tfidf(X_train, X_test, tfidf_vectorizer)

        for classifier_name, (classifier, params) in CLASSIFIERS.items():
            logging.info(f"Running Grid search for {classifier_name} - {field}")
            logging.info(classifier)
            default_keywords = {"n_jobs":-1, "verbose":3, "scoring":METRICS, "refit":"f1_weighted"}
            #if classifier_name == "XGB":
            #    search = RandomizedSearchCv(classifier, params, **default_keywords)
            #else:
            # Refit on best parameters to prepar for prediction - Default
            search = GridSearchCV(classifier, params, **default_keywords)
            search.fit(X_train_vec, y_train)
            logging.info("Best parameter (CV score=%0.3f):" % search.best_score_)
            logging.info(search.best_params_)
            logging.info("Predicting with best params")
            predictions = search.predict(X_test_vec)
            logging.info("Classifaction_report:")
            report_matrix = metrics.classification_report(le.inverse_transform(y_test), le.inverse_transform(predictions), zero_division=0)
            logging.info(report_matrix)
            logging.info("#"*20)


