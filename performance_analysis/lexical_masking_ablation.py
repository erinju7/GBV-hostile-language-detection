# performance_analysis/lexical_masking_ablation.py
# Local-only version (assumes you run from the repo root locally)

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------- CONFIG --------------------

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()
# If runs notebook from /content, try to locate repo folder
if not (BASE_DIR / "data").exists() and (BASE_DIR / "GBV-hostile-language-detection").exists():
    BASE_DIR = BASE_DIR / "GBV-hostile-language-detection"

MODEL_DIR = BASE_DIR / "models" / "albert_gbv" / "jigsaw_gbv_trained"
RESULTS_PATH = BASE_DIR / "results" / "albert_gbv" / "full_results.csv"
OUT_DIR = BASE_DIR / "performance_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 128
BATCH_SIZE = 32
THRESHOLD = 0.5
MASK_TOKEN = "***"

# How many samples to evaluate (for speed). Set None to use all.
N_SAMPLES = 2000

# Build mask list from data (fast heuristic).
# Strategy: extract most frequent tokens from true positives (actual=1 & predicted=1)
MASK_TOP_K = 80
MIN_TOKEN_LEN = 3


# -------------------- HELPERS --------------------
def simple_tokenize(text: str):
    return re.findall(r"[A-Za-z']+", str(text).lower())


def build_mask_list_from_true_positives(df: pd.DataFrame, top_k=80):
    tp = df[(df["actual_label"] == 1) & (df["predicted_label"] == 1)]
    if tp.empty:
        raise ValueError("No true positives found in full_results.csv; cannot build mask list.")

    counts = {}
    for t in tp["text"].astype(str).tolist():
        for tok in simple_tokenize(t):
            if len(tok) < MIN_TOKEN_LEN:
                continue
            counts[tok] = counts.get(tok, 0) + 1

    return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def mask_lexical_cues(text: str, mask_list, mask_token="***") -> str:
    if not isinstance(text, str) or not mask_list:
        return str(text)
    pattern = r"\b(" + "|".join(re.escape(w) for w in mask_list) + r")\b"
    return re.sub(pattern, mask_token, text, flags=re.IGNORECASE)


def batched_predict_proba(model, tokenizer, texts, device, batch_size=32):
    """Return hostile probabilities for a list of texts (class index 1)."""
    probs_all = []
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

        probs_all.append(probs)

    return np.concatenate(probs_all, axis=0)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    assert MODEL_DIR.exists(), f"MODEL_DIR not found: {MODEL_DIR}"
    assert RESULTS_PATH.exists(), f"RESULTS_PATH not found: {RESULTS_PATH}"

    print("Loading model from:", MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    ).to(device)

    print("Loading results from:", RESULTS_PATH)
    df = pd.read_csv(RESULTS_PATH)

    required_cols = {"text", "actual_label", "predicted_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"full_results.csv missing columns: {missing}")

    if N_SAMPLES is not None and N_SAMPLES < len(df):
        df = df.sample(n=N_SAMPLES, random_state=42).copy()

    mask_list = build_mask_list_from_true_positives(df, top_k=MASK_TOP_K)
    print(f"Built mask_list with {len(mask_list)} tokens (top {MASK_TOP_K} from TPs).")

    texts_orig = df["text"].astype(str).tolist()
    texts_mask = [mask_lexical_cues(t, mask_list, MASK_TOKEN) for t in texts_orig]

    p_orig = batched_predict_proba(model, tokenizer, texts_orig, device, batch_size=BATCH_SIZE)
    p_mask = batched_predict_proba(model, tokenizer, texts_mask, device, batch_size=BATCH_SIZE)

    y_true = df["actual_label"].to_numpy().astype(int)
    y_pred_orig = (p_orig >= THRESHOLD).astype(int)
    y_pred_mask = (p_mask >= THRESHOLD).astype(int)

    macro_f1_orig = f1_score(y_true, y_pred_orig, average="macro")
    macro_f1_mask = f1_score(y_true, y_pred_mask, average="macro")

    hostile_recall_orig = recall_score(y_true, y_pred_orig, pos_label=1)
    hostile_recall_mask = recall_score(y_true, y_pred_mask, pos_label=1)

    idx_hostile = (y_true == 1)
    mean_delta = float(np.mean(p_orig[idx_hostile] - p_mask[idx_hostile])) if idx_hostile.any() else float("nan")
    median_delta = float(np.median(p_orig[idx_hostile] - p_mask[idx_hostile])) if idx_hostile.any() else float("nan")

    flip_h2n = float(np.mean((y_pred_orig == 1) & (y_pred_mask == 0)))

    summary = {
        "n_samples": int(len(df)),
        "threshold": float(THRESHOLD),
        "mask_token": MASK_TOKEN,
        "mask_list_size": int(len(mask_list)),
        "macro_f1_original": float(macro_f1_orig),
        "macro_f1_masked": float(macro_f1_mask),
        "hostile_recall_original": float(hostile_recall_orig),
        "hostile_recall_masked": float(hostile_recall_mask),
        "mean_prob_drop_on_true_hostile": mean_delta,
        "median_prob_drop_on_true_hostile": median_delta,
        "flip_rate_hostile_to_nonhostile": flip_h2n,
    }

    summary_path = OUT_DIR / "lexical_masking_ablation_summary.csv"
    pd.Series(summary).to_csv(summary_path)
    print("Saved:", summary_path)
    print(summary)

    per_sample = pd.DataFrame(
        {
            "text_original": texts_orig,
            "text_masked": texts_mask,
            "y_true": y_true,
            "p_hostile_original": p_orig,
            "p_hostile_masked": p_mask,
            "y_pred_original": y_pred_orig,
            "y_pred_masked": y_pred_mask,
            "delta_p": p_orig - p_mask,
        }
    )
    per_sample_path = OUT_DIR / "lexical_masking_ablation_per_sample.csv"
    per_sample.to_csv(per_sample_path, index=False)
    print("Saved:", per_sample_path)