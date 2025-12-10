import os
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- 0. PATHS & BASIC CONFIG ----------

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd()

MODEL_DIR   = BASE_DIR / "models" / "albert_gbv"
RESULTS_PATH = BASE_DIR / "results" / "albert_gbv" / "full_results.csv"
OUT_DIR     = BASE_DIR / "performance_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"BASE_DIR     = {BASE_DIR}")
print(f"MODEL_DIR    = {MODEL_DIR}")
print(f"RESULTS_PATH = {RESULTS_PATH}")

# ---------- 1. LOAD MODEL & TOKENIZER ----------

print("\nLoading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()
print("Model and tokenizer loaded.\n")

# 0 = non-hostile, 1 = hostile
class_names = ["non-hostile", "hostile"]

# ---------- 2. PREDICTION FUNCTION FOR LIME ----------

def predict_proba(texts):
    """
    texts: list[str]
    Returns an array of shape [n_samples, num_labels]
    """
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    return probs

# ---------- 3. LIME EXPLAINER ----------

explainer = LimeTextExplainer(class_names=class_names)

# ---------- 4. PICK ONE TP AND ONE FN (FOR PLOTTING) ----------

def choose_example_from_results(csv_path: Path, example_type: str):
    """
    example_type:
        "tp" = actual=1, predicted=1
        "fn" = actual=1, predicted=0
    """
    df = pd.read_csv(csv_path)
    text_col = "text" if "text" in df.columns else "comment_text"

    if example_type == "tp":
        subset = df[(df["actual_label"] == 1) & (df["predicted_label"] == 1)]
    elif example_type == "fn":
        subset = df[(df["actual_label"] == 1) & (df["predicted_label"] == 0)]
    else:
        raise ValueError("example_type must be 'tp' or 'fn'")

    if subset.empty:
        raise ValueError(f"No examples found for type: {example_type}")

    row = subset.sample(n=1, random_state=42).iloc[0]

    print(f"\nSelected {example_type.upper()} example:")
    print("TEXT:", row[text_col])
    print("actual_label:", row["actual_label"])
    print("predicted_label:", row["predicted_label"])

    return row[text_col]

# ---------- 5. HELPER: PLOT LIME BAR ----------

def plot_lime_bar_on_ax(exp, ax, title):
    lime_df = pd.DataFrame(exp.as_list(label=1), columns=["token", "weight"])
    lime_df["abs_weight"] = lime_df["weight"].abs()

    # Sort by importance and keep top 8 tokens for clarity
    lime_df = lime_df.sort_values("abs_weight", ascending=False).head(8)

    lime_df["direction"] = lime_df["weight"].apply(
        lambda x: "hostile" if x > 0 else "non-hostile"
    )
    palette = sns.color_palette("muted")

    sns.barplot(
        data=lime_df,
        x="weight",
        y="token",
        hue="direction",
        palette={"hostile": palette[1], "non-hostile": palette[0]},
        dodge=False,
        ax=ax,
    )

    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Local contribution to prediction")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(title="", loc="lower right")

# ---------- 6. BUILD LIME TABLE FOR FP / FN / TP ----------

def build_lime_confusion_table(csv_path: Path):
    df = pd.read_csv(csv_path)

    examples = {
        "FP (pred=1, true=0)": df[(df.actual_label == 0) & (df.predicted_label == 1)],
        "FN (pred=0, true=1)": df[(df.actual_label == 1) & (df.predicted_label == 0)],
        "TP (pred=1, true=1)": df[(df.actual_label == 1) & (df.predicted_label == 1)],
    }

    rows = []

    for case, subset in examples.items():
        if subset.empty:
            print(f"No examples found for {case}")
            continue

        row = subset.sample(1, random_state=42).iloc[0]
        text = row["text"]

        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=6,
            num_samples=500
        )

        lime_weights = exp.as_list(label=1)

        token_string = ", ".join(
            [f'"{t}": {w:.3f}' for t, w in lime_weights]
        )

        rows.append({
            "Case": case,
            "Text": text[:180] + ("..." if len(text) > 180 else ""),
            "Predicted Label": int(row["predicted_label"]),
            "Actual Label": int(row["actual_label"]),
            "Token Rankings (LIME)": token_string
        })

    lime_table = pd.DataFrame(rows)
    out_csv = OUT_DIR / "lime_confusion_examples.csv"
    lime_table.to_csv(out_csv, index=False)
    print("Saved LIME explanation table to:", out_csv)

# ---------- 7. MAIN ----------

if __name__ == "__main__":
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # 7.1 TP / FN barplots for the poster
    tp_text = choose_example_from_results(RESULTS_PATH, "tp")
    fn_text = choose_example_from_results(RESULTS_PATH, "fn")

    exp_tp = explainer.explain_instance(
        tp_text, predict_proba, num_features=10, num_samples=500
    )
    exp_fn = explainer.explain_instance(
        fn_text, predict_proba, num_features=10, num_samples=500
    )

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    plot_lime_bar_on_ax(
        exp_tp, axes[0],
        "LIME explanation – Correct hostile prediction (TP)"
    )
    plot_lime_bar_on_ax(
        exp_fn, axes[1],
        "LIME explanation – Missed hostile comment (FN)"
    )

    plt.tight_layout()
    out_path = OUT_DIR / "lime_tp_fn_subplots.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved combined TP/FN LIME plot to:", out_path)

    # 7.2 FP / FN / TP examples table for LaTeX
    build_lime_confusion_table(RESULTS_PATH)