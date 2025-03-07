import pandas as pd

# Load the filtered dataset
filtered_table = pd.read_csv("data/Filtered_CUSTOM_TABLE.csv")

# Keep only the first occurrence of each unique SUBJECT_ID
filtered_table = filtered_table.drop_duplicates(subset="subject_id", keep="first")

# Function to format mean ± standard deviation
def mean_std_format(series):
    return f"{series.mean():.2f} ± {series.std():.2f}"

# Calculate values for the summary table
summary_data = {
    "N (Unique Subjects)": [filtered_table["subject_id"].nunique()],
    "Age (Mean ± SD)": [mean_std_format(filtered_table["subject_age"])],
    "Gender (% Male)": [(filtered_table["PTGENDER"].eq(1).mean() * 100)],
    "Gender (% Female)": [(filtered_table["PTGENDER"].eq(2).mean() * 100)],
    "MMSE Score (Mean ± SD)": [mean_std_format(filtered_table["MMSCORE"])],
    "CDGLOBAL (Mean ± SD)": [mean_std_format(filtered_table["CDGLOBAL"])],
    "ABETA (Mean ± SD)": [mean_std_format(filtered_table["ABETA42"])]
}

# Convert to DataFrame
summary_df = pd.DataFrame(summary_data)

# Save the table as CSV
summary_csv_path = "data/Summary_Table.csv"
summary_df.to_csv(summary_csv_path, index=False)

# Print the summary table
print(f"\n\
    Summary Statistics Table: \n\
    {summary_df.to_string(index=False)} \n\
    ")

print(f"\n\
    Summary table saved as: {summary_csv_path} \n\
    ")
