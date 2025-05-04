# models/naive_bayes.py

# === Imports ===
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import json
import sys
import os

# Add the parent directory to sys.path to import custom modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import custom utility functions and configuration
from utils.utils import clean_text, evaluate_model
from config import MODEL_VERSION, USE_NGRAMS, NGRAM_RANGE, TFIDF_MAX_FEATURES, ALPHA

# === 1. Load Dataset ===
def load_dataset(filepath=None):
    """
    Loads the dataset from the specified file path.
    If no path is provided, it defaults to 'data/SMSSpamCollection'.
    The function renames columns, selects relevant ones, and applies text cleaning.
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "SMSSpamCollection"
    # Load the dataset with tab-separated values
    df = pd.read_csv(filepath, sep='\t', header=None, names=["label", "text"])
    # Keep only the relevant columns
    df = df[["label", "text"]]
    # Apply text cleaning to the 'text' column
    df['clean_text'] = df['text'].apply(clean_text)
    return df

# === 2. Feature Engineering ===
def vectorize_text(text_series):
    """
    Converts the text data into numerical features using TF-IDF vectorization.
    Returns the transformed features and the vectorizer object.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE if USE_NGRAMS else (1, 1),  # Use n-grams if enabled
        max_features=TFIDF_MAX_FEATURES  # Limit the number of features
    )
    X = vectorizer.fit_transform(text_series)  # Fit and transform the text data
    return X, vectorizer

# === 3. Label Encoding ===
def encode_labels(label_series):
    """
    Encodes the labels ('ham' and 'spam') into numerical values.
    Returns the encoded labels and the encoder object.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(label_series)  # 'ham' -> 0, 'spam' -> 1
    return y, encoder

# === 4. Train Model ===
def train_model(X_train, y_train):
    """
    Trains a Naive Bayes model using the training data.
    Returns the trained model.
    """
    model = MultinomialNB(alpha=ALPHA)  # Initialize the model with smoothing parameter
    model.fit(X_train, y_train)  # Train the model
    return model

# === Main Program ===
if __name__ == "__main__":
    # Step 1: Load the dataset
    print("📂 Loading dataset...")
    df = load_dataset()

    # Step 2: Split the dataset into training and testing sets
    print("✂️ Splitting data into training and testing sets...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Step 3: Prepare features and labels
    print("🔧 Preparing features and labels...")
    X_train, vectorizer = vectorize_text(df_train["clean_text"])  # Vectorize training text
    X_test = vectorizer.transform(df_test["clean_text"])  # Transform testing text
    y_train, encoder = encode_labels(df_train["label"])  # Encode training labels
    y_test = encoder.transform(df_test["label"])  # Encode testing labels

    # Display vectorizer configuration and example features
    print(f"🧪 Vectorizer config: USE_NGRAMS={USE_NGRAMS}, RANGE={NGRAM_RANGE}")
    print(f"🔢 Number of Features: {len(vectorizer.get_feature_names_out())}")
    print(f"🔤 Example Features: {vectorizer.get_feature_names_out()[:10]}")

    # Step 4: Train the Naive Bayes model
    print("🤖 Training the Naive Bayes model...")
    model = train_model(X_train, y_train)

    # Step 5: Perform threshold analysis
    print("\n🔍 Performing threshold analysis...")
    y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (spam)

    best_f1 = 0
    best_threshold = 0

    # Iterate through different thresholds to find the best F1 score
    for threshold in [round(x * 0.1, 1) for x in range(1, 9)]:
        y_pred_thresh = (y_probs >= threshold).astype(int)  # Apply threshold
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        print(f"Threshold: {threshold:.1f} → Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n✅ Best Threshold: {best_threshold} with F1-Score: {best_f1:.2f}")

    # Step 6: Evaluate the model with the best threshold
    print("\n📊 Evaluating the model with the best threshold...")
    y_pred_best = (y_probs >= best_threshold).astype(int)
    evaluate_model(model, X_test, y_pred_best, label_names=["Ham", "Spam"],
                   model_name=f"Naive Bayes (Threshold={best_threshold})")

    # Step 7: Save predictions and metrics
    print("\n💾 Saving predictions and metrics...")
    results_path = Path(__file__).parent.parent / "results"
    results_path.mkdir(exist_ok=True)  # Ensure the results directory exists

    # Save predictions
    df_test = df_test.copy()
    df_test["true_label"] = y_test
    df_test["spam_prob"] = y_probs
    df_test["prediction"] = y_pred_best
    df_test["prediction_label"] = df_test["prediction"].map({0: "Ham", 1: "Spam"})
    df_test["true_label_name"] = df_test["true_label"].map({0: "Ham", 1: "Spam"})
    df_test["correct"] = df_test["true_label"] == df_test["prediction"]

    df_test_export = df_test[["text", "true_label_name", "prediction_label", "spam_prob", "correct"]]
    predictions_file = results_path / f"predictions_{MODEL_VERSION}.csv"
    df_test_export.to_csv(predictions_file, index=False)

    # Save metrics
    metrics = {
        "model": "Naive Bayes",
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

    metrics_file = results_path / f"metrics_naive_{MODEL_VERSION}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📄 Predictions saved to: {predictions_file}")
    print(f"📄 Metrics saved to: {metrics_file}")

    # Step 8: Save the model and vectorizer
    model_file = results_path / f"naive_bayes_model_{MODEL_VERSION}.joblib"
    vectorizer_file = results_path / f"vectorizer_{MODEL_VERSION}.joblib"
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    print(f"🧠 Model saved to: {model_file}")
    print(f"🔤 Vectorizer saved to: {vectorizer_file}")
