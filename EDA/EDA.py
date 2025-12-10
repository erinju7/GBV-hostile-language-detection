import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import seaborn as sns


# Save output in same folder as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")

def save_csv(df, name):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path)
    print(f"Saved CSV: {path}")

# Load processed dataset
df = pd.read_csv("data/jigsaw_gbv.csv")
print("Loaded:", df.shape)

# 1. distribution of your binary label
label_dist = df["label"].value_counts()
print("\nLabel distribution:")
print(label_dist)

# Save label distribution
save_csv(label_dist, "label_distribution.csv")

# 2. text length analysis
df["text_length"] = df["comment_text"].fillna("").astype(str).apply(len)

# Save text length CSV
save_csv(df[["text_length"]], "text_length_analysis.csv")

# Set seaborn theme
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="text_length",
    bins=50,
    kde=False,
    color=sns.color_palette("muted")[0]
)

plt.title("Text Length Distribution")
plt.xlabel("Text Length (characters)")
plt.ylabel("Count")

save_fig("text_length_distribution_sns.png")
plt.close()

# 4. word cloud
text = " ".join(df["comment_text"].astype(str).tolist())
wordcloud = WordCloud(width=1600, height=800,
                      background_color="white").generate(text)

plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
save_fig("wordcloud.png")
plt.close()
