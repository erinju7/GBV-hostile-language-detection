# Explainable Detection of Gender-Based Violence Online

## Project Description

This project replicates and adapts the [HEARTS Framework](https://arxiv.org/abs/2409.11579) to investigate the automated detection of **gender-based hostile language** in online comments using **deep learning and explainable AI** techniques. Online gender-based violence (GBV) includes misogynistic, abusive, and harmful language directed at women and girls, a phenomenon that has become increasingly prevalent on digital platforms.

The **ALBERT transformer model** is adapted for binary classification of hostile versus non-hostile content, leveraging its parameter efficiency while maintaining strong classification performance. The model is trained on a curated subset of the [Jigsaw Unintended Bias Dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), filtered to retain comments explicitly related to women and gender-targeted discourse.

This project aligns with the **United Nations Sustainable Development Goals (SDGs)**, particularly:
- **SDG 5**: Gender Equality  
- **SDG 16**: Peace, Justice, and Strong Institutions

## Project Structure
```text
GBV-hostile-language-detection/
│
├── preprocess/
│   └── process_jigsaw.ipynb
│
├── model_training/
│   ├── model_replicate.py
│   └── training_gbv.py
│
├── performance_analysis/
│   ├── lime_explain.py
│   ├── lexical_masking_ablation.py
│   ├── lime_tp_fn_subplots.png
│   ├── lexical_masking_ablation_summary.csv
│   └── lexical_masking_ablation_per_sample.csv
│
├── EDA/
│   ├── EDA.py
│   ├── label_distribution.csv
│   ├── text_length_distribution_sns.png
│   └── wordcloud.png
│
├── data/
│   ├── jigsaw_gbv.csv
│   ├── jigsaw_dataset.csv
│   └── MGSD-Expanded.csv
│
├── results/
│   ├── albert_gbv/
│   │   ├── full_results.csv
│   │   └── classification_report.csv
│   └── albert_mgsd/
│       ├── full_results.csv
│       └── classification_report.csv
│
├── report/
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt

```
## Project Report

A full technical report describing the dataset construction, model training, evaluation, and interpretability analysis is available here:
- [Project Report (PDF)](report/project_report.pdf)

## Excluded files
Model checkpoints are excluded due to file size constraints and reproducibility considerations. All experiments can be reproduced using the provided training scripts.
- **Directory:** `models/`
