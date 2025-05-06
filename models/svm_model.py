# models/svm_model.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse import hstack
import numpy as np
import joblib
import json
import sys
import re

# Eigene Module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import clean_text, evaluate_model
from config import MODEL_VERSION, USE_NGRAMS, NGRAM_RANGE, TFIDF_MAX_FEATURES, ALPHA

def load_dataset(filepath=None):
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "SMSSpamCollection"
    df = pd.read_csv(filepath, sep='\t', header=None, names=["label", "text"])
    df['clean_text'] = df['text'].apply(clean_text)
    return df

def vectorize_text(text_series):
    """
    Converts the text data into numerical features using TF-IDF vectorization.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE if USE_NGRAMS else (1, 1),
        max_features=TFIDF_MAX_FEATURES
    )
    X_tfidf = vectorizer.fit_transform(text_series)
    return X_tfidf, vectorizer

def transform_with_extra(vectorizer, text_series):
    """
    Transforms the text data using the provided vectorizer.
    """
    X_tfidf = vectorizer.transform(text_series)
    return X_tfidf

def extract_additional_features(text_series):
    """
    Extracts additional numeric features from the text.
    """
    return pd.DataFrame({
        "msg_length": text_series.str.len(),
        "num_digits": text_series.str.count(r'\d'),
        "num_links": text_series.str.count(r'https?://'),
        "num_uppercase": text_series.str.count(r'\b[A-Z]{2,}\b')
    })



 
def optimize_threshold(y_true, y_probs):
    best_threshold = 0.5
    best_f1 = 0
    thresholds = np.arange(0.0, 1.1, 0.05)  # Check thresholds from 0.0 to 1.0 with steps of 0.05
    for threshold in thresholds:
         # Classify based on the current threshold
         y_pred = (y_probs >= threshold).astype(int)

        # Calculate the F1-score
         f1 = f1_score(y_true, y_pred)

       # If the F1-score is better, save the threshold
         if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1



if __name__ == "__main__":
    print("📂 Loading dataset...")
    df = load_dataset()

    print("✂️ Splitting data...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    print("🔧 Vectorizing text...")
    X_train, vectorizer = vectorize_text(df_train["clean_text"])
    X_test = transform_with_extra(vectorizer, df_test["clean_text"])

    print("🏷 Encoding labels...")
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df_train["label"])
    y_test = encoder.transform(df_test["label"])



    # Training a Support Vector Machine with a linear kernel (Linear SVM) and balanced class weights.
    # This model is trained directly on the training data without hyperparameter tuning.
    # The block is commented out in favor of using GridSearchCV for optimized performance.
    # Linear SVM: Benefits from more features, as it uses them to improve linear separation, but struggles with too few features.

    '''print("🤖 Training SVM model...")
    model = SVC(kernel="linear", probability=True, class_weight="balanced")
    model.fit(X_train, y_train)
    '''
    


    # Hyperparameter tuning for Support Vector Machine using GridSearchCV (Tuned SVM).
    # Tests combinations of 'C', 'kernel' (linear and rbf), and 'gamma' to find the best parameters.
    # The model is trained using the best parameters and then evaluated on the test set.
    # RBF SVM: Performs well with fewer features due to its ability to model non-linear patterns, but too many features can cause overfitting.
    print("🔍 Starting GridSearchCV for SVM...")

    param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  # Relevant only for 'rbf'
    }

    grid = GridSearchCV(
        SVC(probability=True, class_weight="balanced"),
        param_grid,
        scoring='f1',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    print(f"✅ Best parameters found: {grid.best_params_}")
    
    print("🔍 Predicting probabilities...")
    y_probs = model.predict_proba(X_test)[:, 1]

     # Optimize threshold
    best_threshold, best_f1 = optimize_threshold(y_test, y_probs)

    
    print(f"✅ Best threshold: {best_threshold:.2f}")
    print(f"✅ Best F1-Score: {best_f1:.3f}")
    
    # Optimized predictions based on the best threshold 
    y_pred_best = (y_probs >= best_threshold).astype(int)

    print("\n📊 Evaluation:")
    evaluate_model(model, X_test, y_pred_best, label_names=["Ham", "Spam"],
                   model_name=f"SVM (Threshold={best_threshold})")

    results_path = Path(__file__).parent.parent / "results"
    results_path.mkdir(exist_ok=True)

    df_test = df_test.copy()
    df_test["true_label"] = y_test
    df_test["spam_prob"] = y_probs
    df_test["prediction"] = y_pred_best
    df_test["prediction_label"] = df_test["prediction"].map({0: "Ham", 1: "Spam"})
    df_test["true_label_name"] = df_test["true_label"].map({0: "Ham", 1: "Spam"})
    df_test["correct"] = df_test["true_label"] == df_test["prediction"]

    df_test_export = df_test[["text", "true_label_name", "prediction_label", "spam_prob", "correct"]]
    predictions_file = results_path / f"predictions_svm_{MODEL_VERSION}.csv"
    df_test_export.to_csv(predictions_file, index=False)

    metrics = {
    "model": "SVM",
    "model_version": MODEL_VERSION,
    "threshold": best_threshold,
    "precision": float(precision_score(y_test, y_pred_best)),
    "recall": float(recall_score(y_test, y_pred_best)),
    "f1_score": float(f1_score(y_test, y_pred_best)),
    "accuracy": float((y_test == y_pred_best).mean()),
    "config": {
        "USE_NGRAMS": USE_NGRAMS,
        "NGRAM_RANGE": NGRAM_RANGE,
        "TFIDF_MAX_FEATURES": TFIDF_MAX_FEATURES,
        "ALPHA": ALPHA
    }
}

    metrics_file = results_path / f"metrics_svm_{MODEL_VERSION}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📄 Predictions saved to: {predictions_file}")
    print(f"📄 Metrics saved to: {metrics_file}")
