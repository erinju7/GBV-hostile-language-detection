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
├── scripts/                 # Data preprocessing and model training
│   ├── preprocess_jigsaw.py
│   ├── train_gbv.py
│   └── evaluate_gbv.py
│
├── performance_analysis/    # Interpretability and evaluation
│   ├── lime_tp_fn.py
│   └── lime_tp_fn_subplots.png
│
├── notebooks/               # Exploratory analysis and visualisation
│   ├── EDA_jigsaw.ipynb
│   └── error_analysis.ipynb
│
├── figs/                    # Figures used in poster or report
│
├── data/                    # Dataset directory (not tracked in GitHub)
├── models/                  # Trained models (not tracked in GitHub)
├── results/                 # Prediction outputs (not tracked in GitHub)
│
├── README.md
├── requirements.txt
└── .gitignore
```
## Project Report

A full technical report describing the dataset construction, model training, evaluation, and interpretability analysis is available here:
- [Project Report (PDF)](report/project_report.pdf)
