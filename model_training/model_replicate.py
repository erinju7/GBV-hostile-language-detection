# ---------- 0. CONFIG ----------
import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    classification_report,
)

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "MGSD - Expanded.csv"

TEXT_COL = "text"
LABEL_COL = "label"
GROUP_COL = "stereotype_type"

MODEL_NAME = "albert-base-v2"
OUTPUT_DIR = "/tmp/erinju_albert_mgsd_model"
RESULTS_DIR = "/tmp/erinju_albert_mgsd_results"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5

# 1. Load CSV
df = pd.read_csv(CSV_PATH)

# 2. Map labels
def map_label(x):
    x = str(x).strip().lower()
    if x.startswith("stereotype_"):
        return 1
    else:
        return 0

df["label"] = df["label"].apply(map_label)
print(df["label"].value_counts())   # should show both 0 and 1

# ---------- 3. TRAIN / TEST SPLIT ----------

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df[LABEL_COL],
    random_state=RANDOM_STATE,
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("Train label counts:\n", train_df[LABEL_COL].value_counts())
print("Test label counts:\n", test_df[LABEL_COL].value_counts())

# ---------- 4. HUGGINGFACE DATASETS ----------

cols = [TEXT_COL, LABEL_COL]
if GROUP_COL in df.columns:
    cols.append(GROUP_COL)

train_ds = Dataset.from_pandas(train_df[cols])
test_ds = Dataset.from_pandas(test_df[cols])

print("Example from train_ds:", train_ds[0])

# ---------- 5. TOKENIZATION ----------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(examples):
    return tokenizer(
        examples[TEXT_COL],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

train_ds_tok = train_ds.map(tokenize_batch, batched=True)
test_ds_tok = test_ds.map(tokenize_batch, batched=True)

# rename label column → "labels" for HF Trainer
train_ds_tok = train_ds_tok.rename_column(LABEL_COL, "labels")
test_ds_tok = test_ds_tok.rename_column(LABEL_COL, "labels")

train_ds_tok.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)
test_ds_tok.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

label2id = {0: 0, 1: 1}
id2label = {0: "non_stereotype", 1: "stereotype"}

# ---------- 6. MODEL + TRAINER ----------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
)
model.to(device)

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds_tok,
    eval_dataset=test_ds_tok,   # evaluate on held-out test set
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ---------- 7. TRAIN ----------

trainer.train()
trainer.save_model(OUTPUT_DIR)

# ---------- 8. EVALUATE & SAVE METRICS ----------

pred_output = trainer.predict(test_ds_tok)
logits = pred_output.predictions          # shape: (N, 2)
test_preds = np.argmax(logits, axis=1)    # predicted class (0/1)
test_labels = np.array(test_df[LABEL_COL].tolist())

print("\nClassification report (test):")
print(classification_report(test_labels, test_preds, digits=4))

report_dict = classification_report(
    test_labels, test_preds, digits=4, output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose()
report_path = os.path.join(RESULTS_DIR, "classification_report_test.csv")
report_df.to_csv(report_path)
print("Saved classification report to:", report_path)

# ---------- 9. SAVE PREDICTIONS + TEXT + GROUP ----------

results_df = test_df.copy()
results_df["pred_label"] = test_preds

# convert numpy logits → torch tensor on CPU, then softmax
probs = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
results_df["pred_prob_stereotype"] = probs

preds_path = os.path.join(RESULTS_DIR, "test_predictions_with_text.csv")
results_df.to_csv(preds_path, index=False)
print("Saved predictions to:", preds_path)

print("\nDone. Model dir:", OUTPUT_DIR)
