import pandas as pd

# Load the dataset
custom_table = pd.read_csv("data/CUSTOM_TABLE.csv")

# Function to count unique subjects
def count_subjects(df):
    return len(df["subject_id"].unique())

# Initial count of subjects
subject_count = count_subjects(custom_table)
print(f"Initial Subject Count: {subject_count}")

# 1. Filter missing MMSE Score
custom_table = custom_table.dropna(subset=["MMSCORE"])
subject_count = count_subjects(custom_table)
print(f"Subject Count after MMSE Score Filter: {subject_count}")

# 2. Filter missing WORD1, WORD2, or WORD3 (at least one must be present)
word_condition = custom_table[["WORD1", "WORD2", "WORD3"]].notna().any(axis=1)
custom_table = custom_table[word_condition]
subject_count = count_subjects(custom_table)
print(f"Subject Count after WORD Filter: {subject_count}")

# 3. Filter missing MMSE columns (MMD, MML, MMR, MMO, MMW) or (MMLTR1-7 + WORLDSCORE)
mmse_condition = custom_table[["MMD", "MML", "MMR", "MMO", "MMW"]].notna().sum(axis=1) == 5
alt_condition = (
    custom_table[["MMLTR1", "MMLTR2", "MMLTR3", "MMLTR4", "MMLTR5", "MMLTR6", "MMLTR7"]].notna().sum(axis=1) == 7
) & custom_table["WORLDSCORE"].notna()
custom_table = custom_table[mmse_condition | alt_condition]
subject_count = count_subjects(custom_table)
print(f"Subject Count after MMSE Filter: {subject_count}")

# 4. Filter missing Subject Age
custom_table = custom_table.dropna(subset=["subject_age"])
subject_count = count_subjects(custom_table)
print(f"Subject Count after Age Filter: {subject_count}")

# 5. Filter missing Gender (PTGENDER)
custom_table = custom_table.dropna(subset=["PTGENDER"])
subject_count = count_subjects(custom_table)
print(f"Subject Count after Gender Filter: {subject_count}")

# 6. Filter missing CDRGLOBAL
custom_table = custom_table.dropna(subset=["CDGLOBAL"])
subject_count = count_subjects(custom_table)
print(f"Subject Count after CDRGLOBAL Filter: {subject_count}")

# 7. Filter missing GENOTYPE (APOE)
custom_table = custom_table.dropna(subset=["GENOTYPE"])
subject_count = count_subjects(custom_table)
print(f"Subject Count after GENOTYPE Filter: {subject_count}")

# 8. Filter missing Diagnosis
custom_table = custom_table.dropna(subset=["DIAGNOSIS"])
subject_count = count_subjects(custom_table)
print(f"Subject Count after Diagnosis Filter: {subject_count}")

# 9. Filter missing Scan
custom_table = custom_table.dropna(subset=["SCAN"])
subject_count = count_subjects(custom_table)
print(f"Subject Count after Scan Filter: {subject_count}")

# 10. Filter missing Field Strength
custom_table = custom_table.dropna(subset=["FIELD_STRENGTH"])
subject_count = count_subjects(custom_table)
print(f"Subject Count after Field Strength Filter: {subject_count}")

# 11. Filter individuals with < 2 visits
visit_counts = custom_table["subject_id"].value_counts()
valid_subjects = visit_counts[visit_counts >= 2].index
custom_table = custom_table[custom_table["subject_id"].isin(valid_subjects)]
subject_count = count_subjects(custom_table)
print(f"Subject Count after Visit Filter: {subject_count}")

# Save the filtered dataset
filtered_path = "data/Filtered_CUSTOM_TABLE.csv"
custom_table.to_csv(filtered_path, index=False)