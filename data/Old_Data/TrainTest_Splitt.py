import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_train_test_splitt():
    """
    Splits the subjects into train (70%) and test (30%) sets while maintaining 
    the proportion of Group 0 and Group 1 using stratification.

    Parameters:
    - data_path (str): Path to the input CSV file.
    - subject_group_path (str): Path to the CSV file containing the subject groups.
    """

    # Load the dataset
    df = pd.read_csv("data/Filtered_CUSTOM_TABLE.csv")
    groups = pd.read_csv("data/Alzheimer_Groups.csv")

    # Merge the full dataset with subject group assignments
    df = df.merge(groups, on="subject_id", how="inner")

    # Perform stratified train-test split (based on unique subjects)
    train_subjects, test_subjects = train_test_split(
        groups,
        test_size=0.3, # 30% test set
        stratify=groups["group"], # Maintain the proportion of Group 0 and Group 1
        random_state=42 # For reproducibility
    )

    # Add a new 'Split' column to indicate Train or Test
    df["Split"] = df["subject_id"].apply(lambda x: "Train" if x in train_subjects["subject_id"].values else "Test")

    # Save the dateset with the Train/Test Column
    df.to_csv("data/TrainTest_Table.csv", index=False)


stratified_train_test_splitt()