import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv("data/CUSTOM_TABLE_FILTERED.csv")

# ---------------------------
# Aggregate to Unique Patients
# ---------------------------
# Assuming that a lower "visit" value corresponds to the baseline,
# sort by 'visit' and take the first entry for each patient (subject_id)
df_unique = df.sort_values('visit').groupby('subject_id', as_index=False).first()

# Optional: Inspect the aggregated data
print("Unique patient data (first visit per patient):")
print(df_unique.head())

# ---------------------------
# 1. Demographic Stratification
# ---------------------------
# Create age groups based on 'subject_age'
age_bins = [0, 65, 70, 80, 100]  # Adjust these as appropriate for your data
age_labels = ['<65', '70-75', '75-80', '80+']
df['age_group'] = pd.cut(df['subject_age'], bins=age_bins, labels=age_labels, right=False)

# Group by the new age_group and gender (PTGENDER)
demographic_strata = df.groupby(['age_group', 'PTGENDER']).size().reset_index(name='count')

# Pivot the data so that each gender becomes a separate column
demographic_pivot = demographic_strata.pivot(index='age_group', columns='PTGENDER', values='count').fillna(0)

# Plot the demographic stratification as a bar chart
demographic_pivot.plot(kind='bar', figsize=(8,6))
plt.title("Demographic Stratification: Age Group by Gender")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ---------------------------
# 2. Clinical Stratification
# ---------------------------
# Group by clinical variables: DIAGNOSIS and CDR score (CDGLOBAL)
clinical_strata = df.groupby(['DIAGNOSIS', 'CDGLOBAL']).size().reset_index(name='count')

# For visualization, plot a bar chart for each diagnosis separately
diagnoses = clinical_strata['DIAGNOSIS'].unique()
num_diag = len(diagnoses)

# Create subplots for each diagnosis
fig, axes = plt.subplots(num_diag, 1, figsize=(8, 4 * num_diag), squeeze=False)
for ax, diag in zip(axes.flatten(), diagnoses):
    data = clinical_strata[clinical_strata['DIAGNOSIS'] == diag]
    ax.bar(data['CDGLOBAL'].astype(str), data['count'], color='skyblue')
    ax.set_title(f"Diagnosis: {diag}")
    ax.set_xlabel("CDGLOBAL")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.show()

# ---------------------------
# 3. Additional Stratification
# ---------------------------
# Stratification by GENOTYPE (APOE)
if 'GENOTYPE' in df.columns:
    genotype_strata = df.groupby('GENOTYPE').size().reset_index(name='count')
    plt.figure(figsize=(8, 6))
    plt.bar(genotype_strata['GENOTYPE'].astype(str), genotype_strata['count'], color='coral')
    plt.title("Stratification by GENOTYPE (APOE)")
    plt.xlabel("GENOTYPE")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Stratification by ABETA42 levels
if 'ABETA42' in df.columns:
    abeta42_bins = [0, 300, 600, 900, 1200]  # Example bin edges
    abeta42_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    df['ABETA42_group'] = pd.cut(df['ABETA42'], bins=abeta42_bins, labels=abeta42_labels, right=False)
    abeta42_strata = df.groupby('ABETA42_group').size().reset_index(name='count')
    
    plt.figure(figsize=(8,6))
    plt.bar(abeta42_strata['ABETA42_group'].astype(str), abeta42_strata['count'], color='limegreen')
    plt.title("Stratification by ABETA42 Levels")
    plt.xlabel("ABETA42 Group")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# ---------------------------
# 4. Stratification by MMSE Score
# ---------------------------
# Create a new column that categorizes MMSCORE into two groups: '>=24' and '<24'
# Ensure MMSCORE is numeric; if not, convert it (and handle errors as needed)
df['MMSCORE'] = pd.to_numeric(df['MMSCORE'], errors='coerce')

# Drop rows with missing MMSCORE values if necessary
df_mmse = df.dropna(subset=['MMSCORE']).copy()

# Create the MMSE category
df_mmse['MMSE_category'] = df_mmse['MMSCORE'].apply(lambda x: '>=24' if x >= 24 else '<24')

# Group by MMSE_category
mmse_strata = df_mmse.groupby('MMSE_category').size().reset_index(name='count')
print("\nMMSE Score Stratification:")
print(mmse_strata)

# Plot MMSE stratification as a bar chart
plt.figure(figsize=(6, 6))
plt.bar(mmse_strata['MMSE_category'], mmse_strata['count'], color='purple')
plt.title("Stratification by MMSE Score")
plt.xlabel("MMSE Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
