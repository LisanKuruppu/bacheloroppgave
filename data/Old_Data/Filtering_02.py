import pandas as pd

# Load datasets with low_memory=False to prevent dtype warnings
data = pd.read_csv('data/CUSTOM_TABLE.csv', low_memory=False)

# Ensure column names are stripped of extra spaces
data.columns = data.columns.str.strip()

# Print the number of unique subjects
print(f"Number of unique subjects: \n\
      {data['subject_id'].nunique()} \n\
        ")

# Ensure at least one of the WORD columns (WORD1, WORD2, WORD3) is present
required_mmse_cols = [
    "MMYEAR", "MMDATE", "MMSEASON", "MMDAY", "MMMONTH",
    "MMSTATE", "MMCITY", "MMAREA", "MMHOSPIT", "MMFLOOR",
    "WORD1DL", "WORD2DL", "WORD3DL", "MMWATCH", "MMPENCIL", 
    "MMREPEAT", "MMHAND", "MMFOLD", "MMONFLR", "MMREAD", 
    "MMWRITE", "MMDRAW", "MMSCORE"
]

data = data.dropna(subset=required_mmse_cols)

word_conditions = data[["WORD1", "WORD2", "WORD3"]].notnull().any(axis=1)
data = data[word_conditions]

mmse_cols = ["MMD", "MML", "MMR", "MMO", "MMW"]
mmltr_cols = ["MMLTR1", "MMLTR2", "MMLTR3", "MMLTR4", "MMLTR5", "MMLTR6", "MMLTR7", "WORLDSCORE"]

# Condition 1: Ensure all five MMSE columns (MMD, MML, MMR, MMO, MMW) are filled
mmse_conditions = data[mmse_cols].notnull().all(axis=1)

# Condition 2: Ensure all MMLTR1 - MMLTR7 AND WORLDSCORE are available
mmltr_conditions = data[mmltr_cols].notnull().all(axis=1)

# Apply both conditions
data = data[mmse_conditions | mmltr_conditions]

# Print the number of unique subjects after filtering
print(f"After MMSE filtering: \n\
        {data['subject_id'].nunique()}  \n\
        ")

# Remove individuals with missing values for critical fields
required_columns = ["subject_age","PTGENDER","CDGLOBAL", "GENOTYPE", "DIAGNOSIS", "SCAN", "FIELD_STRENGTH"]
data = data.dropna(subset=required_columns)

# Retain individuals with at least 2 visits
visit_counts = data.groupby('subject_id').size()
valid_subjects = visit_counts[visit_counts >= 2].index
data = data[data['subject_id'].isin(valid_subjects)]

# Print the number of unique subjects after filtering
print(f"After additional filtering: \n\
        {data['subject_id'].nunique()}  \n\
        ")

# Save the filtered data to a new CSV file
data.to_csv('data/CUSTOM_TABLE_FILTERED.csv', index=False)