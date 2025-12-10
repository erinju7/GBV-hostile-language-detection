# ALBERT-v2 fine-tuning on Jigsaw GBV dataset
# ------------------------------------------

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    balanced_accuracy_score,
)

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# ---------- 0. CONFIG ----------

# Project base dir: .../CLEAN_PROJECT
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "jigsaw_gbv.csv"

MODEL_NAME = "albert-base-v2"

MODEL_OUTPUT_BASE_DIR = "/tmp/erinju_albert_jigsaw_model"
RESULTS_OUTPUT_BASE_DIR = "/tmp/erinju_albert_jigsaw_results"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

os.makedirs(MODEL_OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_BASE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

# ---------- 1. LOAD DATA ----------

df = pd.read_csv(CSV_PATH)

# Rename + add columns expected by helper functions
df = df.rename(columns={"comment_text": "text"})
df["group"] = "jigsaw_gbv"      # dummy group label
df["data_name"] = "jigsaw_gbv"  # dataset identifier for output

# ---------- 2. TRAIN / TEST SPLIT ----------

train_data, test_data = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"]
)

print("Train size:", len(train_data))
print("Test size:", len(test_data))

# ---------- 3. TOKENIZER & TOKENIZE FUNCTION ----------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

# ---------- 4. TRAINING HELPER ----------

def train_model(
    train_data: pd.DataFrame,
    model_path: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    model_output_base_dir: str,
    dataset_name: str,
    seed: int,
):
    np.random.seed(seed)

    num_labels = len(train_data["label"].unique())
    print(f"Number of unique labels: {num_labels}")

    model_output_dir = os.path.join(model_output_base_dir, dataset_name)
    os.makedirs(model_output_dir, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    # small train/validation split
    train_df, val_df = train_test_split(
        train_data,
        test_size=0.2,
        random_state=seed,
        stratify=train_data["label"],
    )

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    train_ds_tok = (
        train_ds
        .map(tokenize_function, batched=True)
        .map(lambda ex: {"labels": ex["label"]})
    )
    val_ds_tok = (
        val_ds
        .map(tokenize_function, batched=True)
        .map(lambda ex: {"labels": ex["label"]})
    )

    print("Sample tokenized train example:", train_ds_tok[0])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        bal_acc = balanced_accuracy_score(labels, preds)
        return {
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "balanced_accuracy": bal_acc,
        }

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        # keep args compatible with older transformers
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds_tok,
        eval_dataset=val_ds_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    print("Validation metrics:", val_metrics)

    trainer.save_model(model_output_dir)
    return model_output_dir

# ---------- 5. EVALUATION HELPER ----------

def evaluate_model(
    test_data: pd.DataFrame,
    model_output_dir: str,
    result_output_base_dir: str,
    dataset_name: str,
    seed: int,
):
    np.random.seed(seed)

    num_labels = len(test_data["label"].unique())
    print(f"Number of unique labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    tokenizer_local = AutoTokenizer.from_pretrained(model_output_dir)

    def tokenize_function_local(examples):
        return tokenizer_local(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    test_ds = Dataset.from_pandas(test_data)
    test_ds_tok = (
        test_ds
        .map(tokenize_function_local, batched=True)
        .map(lambda ex: {"labels": ex["label"]})
    )

    result_output_dir = os.path.join(result_output_base_dir, dataset_name)
    os.makedirs(result_output_dir, exist_ok=True)

    # Use Trainer.predict for evaluation
    eval_args = TrainingArguments(
        output_dir=os.path.join(result_output_dir, "tmp_eval"),
        per_device_eval_batch_size=BATCH_SIZE,
    )

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer_local,
    )

    pred_output = eval_trainer.predict(test_ds_tok)
    logits = pred_output.predictions
    preds = np.argmax(logits, axis=-1)
    y_true = test_data["label"].to_numpy()

    # probability of positive class (if binary)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    if probs.shape[1] == 2:
        prob_pos = probs[:, 1]
    else:
        # multi-class: use max prob
        prob_pos = probs.max(axis=1)

    # save full results
    results_df = pd.DataFrame(
        {
            "text": test_data["text"],
            "predicted_label": preds,
            "predicted_probability": prob_pos,
            "actual_label": y_true,
            "group": test_data["group"],
            "dataset_name": test_data["data_name"],
        }
    )

    results_file_path = os.path.join(result_output_dir, "full_results.csv")
    results_df.to_csv(results_file_path, index=False)
    print("Saved full results to:", results_file_path)

    # save classification report
    report = classification_report(y_true, preds, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    report_file_path = os.path.join(
        result_output_dir, "classification_report.csv"
    )
    df_report.to_csv(report_file_path)
    print("Saved classification report to:", report_file_path)

    return df_report

# ---------- 6. MAIN PIPELINE ----------

if __name__ == "__main__":
    # 1. Train ALBERT-v2 on Jigsaw GBV
    model_output_dir = train_model(
        train_data=train_data,
        model_path=MODEL_NAME,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        model_output_base_dir=MODEL_OUTPUT_BASE_DIR,
        dataset_name="jigsaw_gbv_trained",
        seed=RANDOM_STATE,
    )

    # 2. Evaluate on held-out test set
    report = evaluate_model(
        test_data=test_data,
        model_output_dir=model_output_dir,
        result_output_base_dir=RESULTS_OUTPUT_BASE_DIR,
        dataset_name="jigsaw_gbv",
        seed=RANDOM_STATE,
    )

    print("\nClassification report (macro):")
    print(report)