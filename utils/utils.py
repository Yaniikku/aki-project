# utils.py

import re
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import json
from sklearn.metrics import classification_report, confusion_matrix


# === 1. Text Cleaning Funktion ===
def clean_text(text):
    """
    Standardisiert Text:
    - Alles klein
    - Entfernt Sonderzeichen und Zahlen
    """
    text = text.lower()
    text = re.sub(r'\b\d{5,}\b', 'PHONENUMBER', text)  # NEU: ersetzt Telefonnummern
    text = re.sub(r'[^a-z\s]', '', text)  # weiterhin alle Sonderzeichen raus
    return text


# === 2. Evaluation-Funktion ===
def evaluate_model(model, X_test, y_test, label_names=["Ham", "Spam"], model_name="Modell"):
    """
    Druckt Klassifikationsbericht & zeigt Verwirrungsmatrix.
    Für beliebige Klassifikationsmodelle.
    """
    y_pred = model.predict(X_test)

    print(f"📊 Evaluation für {model_name}:\n")
    print(classification_report(y_test, y_pred, target_names=label_names))

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
                xticklabels=label_names, yticklabels=label_names, cmap='Blues')
    plt.title(f"Verwirrungsmatrix: {model_name}")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")
    plt.tight_layout()
    plt.show()


# === 3. Beispielvorhersage für neuen Text ===
def predict_and_print(model, vectorizer, raw_text, label_names=["Ham", "Spam"]):
    """
    Führt Vorhersage auf neuer Nachricht durch und gibt lesbare Ausgabe.
    """
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    print(f"\n📩 Nachricht: {raw_text}")
    print(f"📊 Vorhersage: {label_names[pred]}")

def generate_config_hash(config):
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]
