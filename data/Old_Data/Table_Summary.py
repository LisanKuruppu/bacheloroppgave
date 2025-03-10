import pandas as pd

def generate_summary_table(filtered_table_path, alzheimer_groups_path, summary_csv_path):
    """
    Generates a summary table with statistics and saves it as a CSV file.

    Parameters:
    - filtered_table_path (str): Path to the filtered dataset CSV file.
    - alzheimer_groups_path (str): Path to the Alzheimer's groups CSV file.
    - summary_csv_path (str): Path to save the summary table CSV file.
    """
    # Load the filtered dataset
    filtered_table = pd.read_csv(filtered_table_path)
    alzheimer_groups = pd.read_csv(alzheimer_groups_path)

    # Use only the subjects that are in the Alzheimer's table
    filtered_table = filtered_table[filtered_table["subject_id"].isin(alzheimer_groups["subject_id"])]
    filtered_table = filtered_table.reset_index(drop=True)

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
        "APOE (3/4)": [(filtered_table["GENOTYPE"].eq("3/4").mean() * 100)],
        "APOE (4/4)": [(filtered_table["GENOTYPE"].eq("4/4").mean() * 100)],
        "ABETA42 (Mean ± SD)": [mean_std_format(filtered_table["ABETA42"])]
    }

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save the table as CSV
    summary_df.to_csv(summary_csv_path, index=False)

    # Print the summary table
    print(f"\n\
        Summary Statistics Table: \n\
        {summary_df.to_string(index=False)} \n\
        ")

    print(f"\n\
        Summary table saved as: {summary_csv_path} \n\
        ")

def generate_train_test_summary():
    """
    Generates a summary table for Train and Test sets, summarizing key statistics.

    Parameters:
    - data_path (str): Path to the CSV file containing the full dataset with Train/Test split.
    - output_path (str): Path to save the summary CSV file.

    Returns:
    - None (Saves the summary as a CSV file).
    """

    df = pd.read_csv("data/TrainTest_Table.csv")

    # Function to format mean ± standard deviation
    def mean_std_format(series):
        return f"{series.mean():.2f} ± {series.std():.2f}"

    # Create summary statistics for Train and Test sets
    summary_data = []

    for split in ["Train", "Test"]:
        subset = df[df["Split"] == split]
        unique_subjects = subset["subject_id"].nunique()

        summary_data.append({
            "Split": split,
            "N (Unique Subjects)": unique_subjects,
            "Age (Mean ± SD)": mean_std_format(subset["subject_age"]),
            "Gender (% Male)": (subset["PTGENDER"].eq(1).mean() * 100),
            "Gender (% Female)": (subset["PTGENDER"].eq(2).mean() * 100),
            "MMSE Score (Mean ± SD)": mean_std_format(subset["MMSCORE"]),
            "CDGLOBAL (Mean ± SD)": mean_std_format(subset["CDGLOBAL"]),
            "ABETA (Mean ± SD)": mean_std_format(subset["ABETA42"])
        })     

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)   

    # Save summary table as CSV
    summary_df.to_csv("data/TrainTest_Summary.csv", index=False)

    # Print summary table
    print(f"\n\
        Train-Test Summary Table: \n\
        {summary_df.to_string(index=False)} \n\
        ")


general_summary = generate_summary_table(
    "data/Filtered_CUSTOM_TABLE.csv",
    "data/Alzheimer_Groups.csv",
    "data/Summary_Table.csv"
    )

train_test_summary = generate_train_test_summary()