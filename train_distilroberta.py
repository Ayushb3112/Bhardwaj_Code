import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import os
from sklearn.metrics import accuracy_score, f1_score
import joblib

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1": f1}

def train_transformer(data_dir, model_dir):
    # Load data
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Initialize tokenizer and model
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)

    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Split training data into train and validation based on fold 0
    # (Simplified for the demonstration - we use fold 0 as validation)
    train_split = tokenized_train.filter(lambda x: x['fold'] != 0)
    val_split = tokenized_train.filter(lambda x: x['fold'] == 0)
    
    # Initialize Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(model_dir, "results"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=100,
        report_to="none"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split,
        eval_dataset=val_split,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train
    print("Fine-tuning DistilRoBERTa...")
    trainer.train()
    
    # Evaluate on final hold-out test set
    print("\n--- DistilRoBERTa (Hold-out Test Prediction) ---")
    predictions = trainer.predict(tokenized_test)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids
    
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")
    
    # Save the best model
    model.save_pretrained(os.path.join(model_dir, "distilroberta_best"))
    tokenizer.save_pretrained(os.path.join(model_dir, "distilroberta_best"))
    
    # Save test results for comparison logic
    transformer_results = {
        'acc': test_acc,
        'f1': test_f1,
        'y_pred': y_pred.tolist(),
        'y_true': y_true.tolist()
    }
    joblib.dump(transformer_results, os.path.join(model_dir, 'transformer_results.joblib'))
    
    print(f"\nDistilRoBERTa results saved to {model_dir}")

if __name__ == "__main__":
    DATA_DIR = '/Users/ayushb3112/Documents/tdt4241/data_splits'
    MODEL_DIR = '/Users/ayushb3112/Documents/tdt4241/models'
    train_transformer(DATA_DIR, MODEL_DIR)
