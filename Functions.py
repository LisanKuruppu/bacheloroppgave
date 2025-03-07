import pandas as pd

def split_alzheimer_groups(data_path):
    """
    Splits subjects into two groups based on their diagnosis history:
    
    - Group 0: Patients who NEVER had DIAGNOSIS = 3 in any visit (includes 1s and 2s only).
    - Group 1: Patients who had DIAGNOSIS = 1 or 2 on their first visit and later changed to DIAGNOSIS = 3 in any subsequent visit.

    Parameters:
    - data_path (str): Path to the input CSV file.
    
    Returns:
    - None (Prints the number of unique subjects in each group and unclassified subjects).
    """
    
    # Load the dataset
    df = pd.read_csv(data_path)

    # Get diagnosis history per subject (preserves visit order)
    subject_diagnosis = df.groupby("subject_id")["DIAGNOSIS"].agg(list)

    # Define Group 0: Patients who NEVER had DIAGNOSIS = 3 in any visit
    group_0_subjects = subject_diagnosis[subject_diagnosis.apply(lambda x: 3 not in x)].index

    # Define Group 1: Patients who had DIAGNOSIS = 3 in any visit
    group_1_subjects = subject_diagnosis[subject_diagnosis.apply(lambda x: 3 in x)].index

    # Count number of unique subjects 
    all_subjects = set(df["subject_id"].unique())


    # Print the number of unique subjects in each group
    print(f"\n\
          Total Subjects: {len(all_subjects)} \n\
          Group 0 (No Alzheimer): {len(group_0_subjects)} \n\
          Group 1 (Developed Alzheimer): {len(group_1_subjects)} \n\
          ")

# Example usage:
split_alzheimer_groups(data_path="data/Filtered_CUSTOM_TABLE.csv")
