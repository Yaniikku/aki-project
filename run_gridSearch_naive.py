import itertools
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path

from utils.utils import clean_text

# === Grid: Alle Parameterkombinationen ===
PARAM_GRID = {
    "USE_NGRAMS": [False, True],
    "NGRAM_RANGE": [(1, 1), (1, 2)],
    "TFIDF_MAX_FEATURES": [1000, 3000, None],
    "ALPHA": [0.1, 0.5, 1.0]
}

# === Daten laden & vorbereiten ===


data_path = Path(__file__).resolve().parent.parent / "aki-project" / "data" / "SMSSpamCollection"
df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])

df["clean_text"] = df["text"].apply(clean_text)
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

X = df["clean_text"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results_path = Path("results")
results_path.mkdir(exist_ok=True)

# === Alle Kombinationen durchprobieren ===
combinations = list(itertools.product(
    PARAM_GRID["USE_NGRAMS"],
    PARAM_GRID["NGRAM_RANGE"],
    PARAM_GRID["TFIDF_MAX_FEATURES"],
    PARAM_GRID["ALPHA"]
))

for idx, (use_ngrams, ngram_range, max_features, alpha) in enumerate(combinations):
    if not use_ngrams and ngram_range != (1, 1):
        continue  # Ignoriere N-Gramme, wenn deaktiviert

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range if use_ngrams else (1, 1),
        max_features=max_features,
        sublinear_tf=True,
        stop_words='english'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_vec, y_train)
    y_probs = model.predict_proba(X_test_vec)[:, 1]

    best_f1 = 0
    best_thresh = 0

    for threshold in [round(x * 0.1, 1) for x in range(1, 9)]:
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = threshold

    y_pred_final = (y_probs >= best_thresh).astype(int)
    precision = precision_score(y_test, y_pred_final)
    recall = recall_score(y_test, y_pred_final)

    # === Speichern ===
    model_version = f"grid_{idx:03d}"
    metrics = {
        "model": "Naive Bayes",
        "model_version": model_version,
        "threshold": best_thresh,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(best_f1, 4),
        "accuracy": round((y_test == y_pred_final).mean(), 4),
        "config": {
            "USE_NGRAMS": use_ngrams,
            "NGRAM_RANGE": ngram_range,
            "TFIDF_MAX_FEATURES": max_features,
            "ALPHA": alpha
        }
    }

    with open(results_path / f"metrics_{model_version}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Saved: metrics_{model_version}.json | F1: {best_f1:.3f}")
