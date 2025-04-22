# models/naive_bayes.py

# This file implements a Naive Bayes classifier for spam detection.
# It includes data loading, preprocessing, feature engineering, model training, and evaluation.
# The Naive Bayes model is applied to classify text messages as "Ham" (not spam) or "Spam".

# === Imports ===
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import sys

# Add the parent directory to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import clean_text, evaluate_model

# === 1. Load Dataset ===
def load_dataset(filepath=None):
    """
    Loads the dataset from the specified file path.
    If no path is provided, it defaults to 'data/spam.csv'.
    The function renames columns, selects relevant ones, and applies text cleaning.
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "spam.csv"
    # Load the dataset with proper encoding
    df = pd.read_csv(filepath, encoding='latin-1')
    # Rename columns for clarity
    df = df.rename(columns={"v1": "label", "v2": "text"})
    # Select only the relevant columns
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
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer

# === 3. Label Encoding ===
def encode_labels(label_series):
    """
    Encodes the labels ('Ham' and 'Spam') into numerical values.
    Returns the encoded labels and the encoder object.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(label_series)  # 'Ham' -> 0, 'Spam' -> 1
    return y, encoder

# === 4. Train Model ===
def train_model(X_train, y_train):
    """
    Trains a Naive Bayes model using the training data.
    Returns the trained model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

# === Main Program ===
if __name__ == "__main__":
    # 1. Load the dataset
    print("📂 Loading dataset...")
    df = load_dataset()

    # 2. Prepare features and labels
    print("🔧 Preparing features and labels...")
    X, vectorizer = vectorize_text(df['clean_text'])  # Vectorize the cleaned text
    y, encoder = encode_labels(df['label'])  # Encode the labels
    df["label_encoded"] = y  # Add encoded labels to the DataFrame

    # 3. Split the data into training and testing sets
    print("✂️ Splitting data into training and testing sets...")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = vectorizer.transform(df_train["clean_text"])  # Transform training text
    X_test = vectorizer.transform(df_test["clean_text"])  # Transform testing text
    y_train = df_train["label_encoded"]
    y_test = df_test["label_encoded"]

    # 4. Train the model
    print("🤖 Training the Naive Bayes model...")
    model = train_model(X_train, y_train)

    # === 5. Threshold Tuning ===
    print("\n🔍 Performing threshold analysis...")
    y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (Spam)

    best_f1 = 0
    best_threshold = 0

    # Iterate through different thresholds to find the best F1 score
    for threshold in [round(x * 0.1, 1) for x in range(2, 9)]:
        y_pred_thresh = (y_probs >= threshold).astype(int)  # Apply threshold
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)

        print(f"Threshold: {threshold:.1f} → Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # === 6. Use the Best Threshold ===
    print(f"\n✅ Best Threshold: {best_threshold} with F1-Score: {best_f1:.2f}")

    # Evaluate the model using the best threshold
    print("\n📊 Evaluating the model with the best threshold...")
    y_pred_best = (y_probs >= best_threshold).astype(int)
    evaluate_model(model, X_test, y_pred_best, label_names=["Ham", "Spam"], model_name=f"Naive Bayes (Threshold={best_threshold})")

    # Example prediction
    print("\n📩 Example prediction:")
    sample_text = "Win a free iPhone now! Click here!"
    sample_clean = clean_text(sample_text)
    sample_vec = vectorizer.transform([sample_clean])
    sample_prob = model.predict_proba(sample_vec)[0, 1]
    sample_pred = "Spam" if sample_prob >= best_threshold else "Ham"

    print(f"→ Text: '{sample_text}'")
    print(f"→ Spam Probability: {sample_prob:.2f}")
    print(f"→ Prediction at Threshold {best_threshold}: {sample_pred}")

    # === Export Predictions to CSV ===
    print("\n💾 Saving predictions to predictions.csv...")
    df_test = df_test.copy()
    df_test["true_label"] = y_test
    df_test["spam_prob"] = y_probs
    df_test["prediction"] = (y_probs >= best_threshold).astype(int)
    df_test["prediction_label"] = df_test["prediction"].map({0: "Ham", 1: "Spam"})
    df_test["true_label_name"] = df_test["true_label"].map({0: "Ham", 1: "Spam"})
    df_test["correct"] = df_test["true_label"] == df_test["prediction"]

    # Reorder columns for export
    df_test_export = df_test[["text", "true_label_name", "prediction_label", "spam_prob", "correct"]]

    # Save to CSV
    results_path = Path(__file__).parent.parent / "results"
    results_path.mkdir(exist_ok=True)
    results_file = results_path / "predictions.csv"
    df_test_export.to_csv(results_file, index=False)

    print(f"✅ Predictions saved to: {results_file}")


