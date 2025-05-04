# bert_classifier.py

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
import torch
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# === 1. Daten laden ===
df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=["label", "text"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# === 2. Aufteilen in Train/Test ===
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# === 3. Tokenizer laden ===
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

# === 4. In Hugging Face Dataset konvertieren ===
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)
train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# === 5. Modell laden ===
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# === 6. Trainingsargumente definieren ===
training_args = TrainingArguments(
    output_dir="./results_bert",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # ← Diese Zeile hinzufügen!
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=1
)


# === 7. Metrics definieren ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# === 8. Trainer starten ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

# === 9. Evaluation ausgeben ===
metrics = trainer.evaluate()
print("\n📊 Final Evaluation:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# === 10. Modell und Tokenizer speichern ===
model_path = Path("results") / "bert_model_v1"
model_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"\n🧠 Modell gespeichert unter: {model_path}")

# === 11. Metriken als JSON speichern ===
metrics["model"] = "BERT"
metrics["model_version"] = "v1"
metrics_file = model_path / "metrics_bert_v1.json"

with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"📄 Metriken gespeichert unter: {metrics_file}")
