# ALBERT-v2 fine-tuning on Jigsaw GBV dataset
# ------------------------------------------

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
# Project base dir (works in local .py and in Colab)
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

# If runs from /content (Colab), try to locate repo folder
if not (BASE_DIR / "data").exists() and (BASE_DIR / "GBV-hostile-language-detection").exists():
    BASE_DIR = BASE_DIR / "GBV-hostile-language-detection"

CSV_PATH = BASE_DIR / "data" / "jigsaw_gbv.csv"

MODEL_NAME = "albert-base-v2"

MODEL_DIR = BASE_DIR / "models" / "albert_gbv"
RESULTS_DIR = BASE_DIR / "results" / "albert_gbv"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

# ---------- 1. LOAD DATA ----------
df = pd.read_csv(CSV_PATH)

# Rename to match expected column name
df = df.rename(columns={"comment_text": "text"})

# Optional metadata fields for saving results
df["group"] = "jigsaw_gbv"
df["data_name"] = "jigsaw_gbv"

# ---------- 2. TRAIN / TEST SPLIT ----------
train_data, test_data = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"],
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

# ---------- 4. TRAINING ----------
def train_model(
    train_data: pd.DataFrame,
    model_name: str,
    model_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
):
    np.random.seed(seed)

    num_labels = len(train_data["label"].unique())
    print(f"Number of unique labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
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
        output_dir=str(model_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",  
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_ds_tok,
        eval_dataset=val_ds_tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    print("Validation metrics:", val_metrics)

    # Save final model to models/albert_gbv/
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    return str(model_dir)

# ---------- 5. EVALUATION ----------
def evaluate_model(
    test_data: pd.DataFrame,
    model_dir: str,
    results_dir: Path,
    seed: int,
):
    np.random.seed(seed)

    num_labels = len(test_data["label"].unique())
    print(f"Number of unique labels: {num_labels}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    tokenizer_local = AutoTokenizer.from_pretrained(model_dir)

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

    eval_args = TrainingArguments(
        output_dir=str(results_dir / "tmp_eval"),
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
    )

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer_local,
    )

    pred_output = eval_trainer.predict(test_ds_tok)
    logits = pred_output.predictions
    preds = np.argmax(logits, axis=-1)
    y_true = test_data["label"].to_numpy()

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    prob_pos = probs[:, 1] if probs.shape[1] == 2 else probs.max(axis=1)

    results_df = pd.DataFrame(
        {
            "text": test_data["text"],
            "predicted_label": preds,
            "predicted_probability": prob_pos,
            "actual_label": y_true,
            "group": test_data.get("group", "jigsaw_gbv"),
            "dataset_name": test_data.get("data_name", "jigsaw_gbv"),
        }
    )

    results_file_path = results_dir / "full_results.csv"
    results_df.to_csv(results_file_path, index=False)
    print("Saved full results to:", results_file_path)

    report = classification_report(y_true, preds, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    report_file_path = results_dir / "classification_report.csv"
    df_report.to_csv(report_file_path)
    print("Saved classification report to:", report_file_path)

    return df_report

# ---------- 6. MAIN PIPELINE ----------
if __name__ == "__main__":
    model_path = train_model(
        train_data=train_data,
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=RANDOM_STATE,
    )

    report = evaluate_model(
        test_data=test_data,
        model_dir=model_path,
        results_dir=RESULTS_DIR,
        seed=RANDOM_STATE,
    )

    print("\nClassification report (macro):")
    print(report)