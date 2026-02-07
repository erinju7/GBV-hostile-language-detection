# Explainable Detection of Gender-Based Violence Online

## Project Description

This project replicates and adapts the [HEARTS Framework](https://arxiv.org/abs/2409.11579) to investigate the automated detection of **gender-based hostile language** in online comments using **deep learning and explainable AI** techniques. Online gender-based violence (GBV) includes misogynistic, abusive, and harmful language directed at women and girls, a phenomenon that has become increasingly prevalent on digital platforms.

The **ALBERT transformer model** is adapted for binary classification of hostile versus non-hostile content, leveraging its parameter efficiency while maintaining strong classification performance. The model is trained on a curated subset of the [Jigsaw Unintended Bias Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), filtered to retain comments explicitly related to women and gender-targeted discourse.

To address concerns around transparency and accountability in automated content moderation systems, this project incorporates **Local Interpretable Model-agnostic Explanations (LIME)**. LIME is used to interpret individual model predictions by identifying which words contribute most strongly to **correct classifications** as well as **misclassifications**, helping to reveal both model strengths and limitations.

This project aligns with the **United Nations Sustainable Development Goals (SDGs)**, particularly:
- **SDG 5**: Gender Equality  
- **SDG 16**: Peace, Justice, and Strong Institutions

## Project Structure
```text
GBV-hostile-language-detection/
│
├── preprocess/                     # Data preprocessing
│   └── process_jigsaw.ipynb
│
├── model_training/                 # Model training & replication
│   ├── model_replicate.py
│   └── training_gbv.py
│
├── performance_analysis/            # Evaluation, explainability & ablation studies
│   ├── lime_explain.py              # LIME-based local explanations (TP / FN analysis)
│   ├── lexical_masking_ablation.py  # Lexical masking ablation experiment
│   ├── lime_tp_fn_subplots.png      # LIME visualization for correct / missed hostile cases
│   ├── lexical_masking_ablation_summary.csv   # Summary metrics for masking ablation
│   └── lexical_masking_ablation_per_sample.csv # Per-sample ablation outputs (local only)
│
├── EDA/                             # Exploratory data analysis
│   ├── EDA.py
│   ├── label_distribution.csv
│   ├── text_length_distribution_sns.png
│   └── wordcloud.png
│
├── data/                            # Datasets
│   ├── jigsaw_gbv.csv               # GBV-focused curated subset
│   ├── jigsaw_dataset.csv           # Full Jigsaw dataset
│   └── MGSD-Expanded.csv            # HEARTS baseline replication dataset
│
├── report/                          # Report writing and figures
│
├── results/                         # Prediction outputs and evaluation metrics
│
├── .gitattributes                   # Repository file attributes
├── .gitignore                       # Ignored files and directories
└── README.md
```
## Project Report

A full technical report describing the dataset construction, model training, evaluation, and interpretability analysis is available here:
- [Project Report (PDF)](report/project_report.pdf)

## Excluded files

**Trained model checkpoints**

- **Directory:** `models/albert_gbv_checkpoint_5205`
- **Description:** Fine-tuned ALBERT-v2 model weights and training artifacts.
- **Rationale:** Model checkpoints are excluded due to file size constraints and reproducibility considerations. All experiments can be reproduced using the provided training scripts.

