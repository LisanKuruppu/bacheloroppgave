import pandas as pd


# Load datasets with low_memory=False to prevent dtype warnings
my_table = pd.read_csv('data/MY TABLE.csv', low_memory=False)
ab42 = pd.read_csv('data/AB42.csv', low_memory=False)
cdr = pd.read_csv('data/CDR.csv', low_memory=False)
demographics = pd.read_csv('data/DEMOGRAPHICS.csv', low_memory=False)
genotype = pd.read_csv('data/GENOTYPE.csv', low_memory=False)

# Ensure column names are stripped of extra spaces
my_table.columns = my_table.columns.str.strip()
ab42.columns = ab42.columns.str.strip()
cdr.columns = cdr.columns.str.strip()
demographics.columns = demographics.columns.str.strip()
genotype.columns = genotype.columns.str.strip()

# Print the number of rows in each dataset.
print(f"Before filtering: \n\
        MY TABLE: {my_table.shape[0]} \n\
        AB42: {ab42.shape[0]} \n\
        CDR: {cdr.shape[0]} \n\
        DEMOGRAPHICS: {demographics.shape[0]} \n\
        GENOTYPE: {genotype.shape[0]} \n\
        ")

# Rename subject_id in MY TABLE to PTID
my_table.rename(columns={'subject_id': 'PTID'}, inplace=True)

# Filter step 1: Retain individuals with at least 2 visits
visit_counts = my_table.groupby('PTID').size()
valid_subjects = visit_counts[visit_counts >= 2].index
my_table = my_table[my_table['PTID'].isin(valid_subjects)]

# Step 2: Remove individuals with missing values for critical fields
required_columns = ["PTGENDER", "DIAGNOSIS", "SCAN", "FIELD_STRENGTH"]
my_table = my_table.dropna(subset=required_columns)

demographics = demographics.dropna(subset=["PTDOBYY", "PTEDUCAT", "PTETHCAT"])
ab42 = ab42.dropna(subset=["ABETA42"])
genotype = genotype.dropna(subset=["GENOTYPE"])
cdr = cdr.dropna(subset=["CDGLOBAL"])


# Print the number of rows in each dataset after filtering
print(f"After filtering: \n\
        MY TABLE: {my_table.shape[0]} \n\
        AB42: {ab42.shape[0]} \n\
        CDR: {cdr.shape[0]} \n\
        DEMOGRAPHICS: {demographics.shape[0]} \n\
        GENOTYPE: {genotype.shape[0]} \n\
        ")

# Ensure at least one of the WORD columns (WORD1, WORD2, WORD3) is present
required_mmse_cols = [
    "MMYEAR", "MMDATE", "MMSEASON", "MMDAY", "MMMONTH",
    "MMSTATE", "MMCITY", "MMAREA", "MMHOSPIT", "MMFLOOR",
    "WORD1DL", "WORD2DL", "WORD3DL", "MMWATCH", "MMPENCIL", 
    "MMREPEAT", "MMHAND", "MMFOLD", "MMONFLR", "MMREAD", 
    "MMWRITE", "MMDRAW", "MMSCORE"
]

# Ensure at least one of the WORD columns (WORD1, WORD2, WORD3) is present
word_condition = my_table[["WORD1", "WORD2", "WORD3"]].notna().any(axis=1)

# Apply filtering based only on the WORD condition
my_table = my_table[word_condition]

# Condition 1: Ensure all five MMSE columns (MMD, MML, MMR, MMO, MMW) are filled
valid_mmse_condition = (
        my_table[["MMD", "MML", "MMR", "MMO", "MMW"]].notna().all(axis=1)
)

# Apply Condition 1 filtering
filtered_table = my_table[valid_mmse_condition]

# Condition 2: Ensure all MMLTR1 - MMLTR7 AND WORLDSCORE are available
alternative_condition = (
    my_table[["MMLTR1", "MMLTR2", "MMLTR3", "MMLTR4", "MMLTR5", "MMLTR6", "MMLTR7"]].notna().sum(axis=1) == 7
) & my_table["WORLDSCORE"].notna()

# Apply Condition 2 filtering to the remaining rows that did not meet Condition 1
remaining_table = my_table[~valid_mmse_condition & alternative_condition]

# Combine the results of both conditions
my_table = pd.concat([filtered_table, remaining_table])

# Print the number of rows in each dataset after filtering
print(f"After filtering MMSE: \n\
        MY TABLE: {my_table.shape[0]} \n\
        AB42: {ab42.shape[0]} \n\
        CDR: {cdr.shape[0]} \n\
        DEMOGRAPHICS: {demographics.shape[0]} \n\
        GENOTYPE: {genotype.shape[0]} \n\
        ")

# Save the filtered datasets to new CSV files data/filtered
my_table.to_csv('data/filtered/MY TABLE_Filtered.csv', index=False)
ab42.to_csv('data/filtered/AB42_Filtered.csv', index=False)
cdr.to_csv('data/filtered/CDR_Filtered.csv', index=False)
demographics.to_csv('data/filtered/DEMOGRAPHICS_Filtered.csv', index=False)
genotype.to_csv('data/filtered/GENOTYPE_Filtered.csv', index=False)