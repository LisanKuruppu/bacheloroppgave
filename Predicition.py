import pandas as pd
import numpy as np
from Logistic_bootstrap_metrics import *
import re

# --- Load data ---
df = pd.read_csv("data/TrainTest_Table.csv")

# --- Parse first 30 prompt embeddings ---
embeddings = []
with open("data/MMSE_Prompts_Train.csv", "r", encoding="utf-8") as f:
    raw_text = f.read()

raw_entries = raw_text.split('","[')[1:]
for entry in raw_entries[:30]:
    embedding_str = "[" + entry.strip().rstrip('"\n')
    floats = re.findall(r"[-+]?\d*\.\d+e[+-]?\d+", embedding_str)
    vector = np.array([float(x) for x in floats])
    embeddings.append(vector)

if len(embeddings) != 30:
    raise ValueError(f"Expected 30 embeddings, got {len(embeddings)}")

embeddings_matrix = np.stack(embeddings)

# --- MMSE question columns ---
mmse_columns = [
    "MMYEAR", "MMMONTH", "MMDAY", "MMSEASON", "MMDATE",
    "MMSTATE", "MMCITY", "MMAREA", "MMHOSPIT", "MMFLOOR",
    "WORD1", "WORD2", "WORD3",
    "MMD", "MML", "MMR", "MMO", "MMW",
    "WORD1DL", "WORD2DL", "WORD3DL",
    "MMWATCH", "MMPENCIL", "MMREPEAT",
    "MMHAND", "MMFOLD", "MMONFLR",
    "MMREAD", "MMWRITE", "MMDRAW"
]

# --- Create subject embeddings ---
mmse_scores = df[mmse_columns].values
weighted_embeddings = mmse_scores @ embeddings_matrix / mmse_scores.sum(axis=1, keepdims=True)
embedding_df = pd.DataFrame(weighted_embeddings, columns=[f"emb_{i}" for i in range(768)])

# --- Merge with metadata ---
df = pd.concat([df.reset_index(drop=True), embedding_df], axis=1)

# --- Recode DIAGNOSIS to binary AD status ---
df = df[df["DIAGNOSIS"].isin([1, 2, 3])]
df["AD_binary"] = df["DIAGNOSIS"].apply(lambda x: 1 if x == 3 else 0)

# --- Train/test split ---
train_df = df[df["Split"] == "Train"]
test_df = df[df["Split"] == "Test"]

# --- Define predictors ---
independent_vars = [f"emb_{i}" for i in range(768)]
X_train = train_df[independent_vars]
y_train = train_df["AD_binary"]
X_test = test_df[independent_vars]
y_test = test_df["AD_binary"]

# --- Run bootstrapped logistic regression ---
results = bootstrap_metrics(
    df_train=X_train,
    df_test=X_test,
    independent_vars=independent_vars,
    dep_var_train=y_train,
    dep_var_test=y_test,
    n_bootstrap=1000,
    threshold=0.5
)

# --- Show Results ---
print("\nOdds Ratios:")
print(results["Odds Ratios (one-time fit)"])

print("\nBootstrapped Performance Metrics:")
for metric, (mean, ci) in results["Bootstrapped Metrics"].items():
    print(f"{metric}: {mean:.3f} (95% CI: {ci[0]:.3f}–{ci[1]:.3f})")

# Optional: Adjusted OR for 0.1 change
adjusted_or_df = compute_adjusted_or(results["model"], increase=0.1)
print("\nAdjusted ORs (0.1 increase):")
print(adjusted_or_df)
