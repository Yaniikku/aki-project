import torch
from datasets import load_dataset
from transformers import DistilBERTTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    # Load dataset
    dataset = load_dataset("sms_spam")
    dataset = dataset.rename_column("label", "labels")
    dataset = dataset.class_encode_column("labels")
    dataset = dataset.train_test_split(test_size=0.2)

    # Load tokenizer and tokenize
    tokenizer = DistilBERTTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize(example):
        return tokenizer(example['sms'], truncation=True, padding=True)

    tokenized = dataset.map(tokenize, batched=True)

    # Load model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results_transformer",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    print("Final Evaluation:", results)

if __name__ == "__main__":
    main()
