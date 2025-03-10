import pandas as pd
from sklearn.model_selection import train_test_split

class AlzheimerDataProcessor:
    def __init__(self, data_path):
        """Initialize the processor with the dataset path."""
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.filter_log = []

    def log_filter_step(self, step_name):
        """Logs the number of unique subjects after a filtering step."""
        unique_subjects = self.df["subject_id"].nunique()
        self.filter_log.append({"Step": step_name, "Unique Subjects": unique_subjects})
        print(f"{step_name}: {unique_subjects} patients remaining.")

    def save_filter_summary(self, output_path):
        """Saves the filtering summary to a CSV file."""
        filter_df = pd.DataFrame(self.filter_log)
        filter_df.to_csv(output_path, index=False)
        print(f"Filtering summary saved to: {output_path} ✅")

    def filter_data(self, output_path):
        """Applies filtering criteria and saves the cleaned dataset."""
        print("Filtering data...")

        self.log_filter_step("Initial Count")

        # Function to count unique subjects
        def count_subjects(df):
            return df["subject_id"].nunique()

        # Apply filtering steps
        self.df = self.df.dropna(subset=["MMSCORE"])  # MMSE Score filter
        self.log_filter_step("MMSE Score Filter")
        
        self.df = self.df[self.df[["WORD1", "WORD2", "WORD3"]].notna().any(axis=1)]  # WORD filter
        self.log_filter_step("WORD Filter")

        mmse_condition = self.df[["MMD", "MML", "MMR", "MMO", "MMW"]].notna().sum(axis=1) == 5
        alt_condition = (self.df[["MMLTR1", "MMLTR2", "MMLTR3", "MMLTR4", "MMLTR5", "MMLTR6", "MMLTR7"]].notna().sum(axis=1) == 7) & self.df["WORLDSCORE"].notna()
        self.df = self.df[mmse_condition | alt_condition]  # MMSE completeness filter
        self.log_filter_step("MMSE Completeness Filter")

        self.df = self.df.dropna(subset=["subject_age"])
        self.log_filter_step("Missing Age Filter")

        self.df = self.df.dropna(subset=["PTGENDER"])
        self.log_filter_step("Missing Gender Filter")

        self.df = self.df.dropna(subset=["CDGLOBAL"])
        self.log_filter_step("Missing CDRGLOBAL Filter")

        self.df = self.df.dropna(subset=["GENOTYPE"])
        self.log_filter_step("Missing GENOTYPE Filter")

        self.df = self.df.dropna(subset=["DIAGNOSIS"])
        self.log_filter_step("Missing Diagnosis Filter")

        self.df = self.df.dropna(subset=["SCAN"])
        self.log_filter_step("Missing Scan Filter")

        self.df = self.df.dropna(subset=["FIELD_STRENGTH"])
        self.log_filter_step("Missing Field Strength Filter")

        # Filter subjects with at least 2 visits
        visit_counts = self.df["subject_id"].value_counts()
        valid_subjects = visit_counts[visit_counts >= 2].index
        self.df = self.df[self.df["subject_id"].isin(valid_subjects)]
        self.log_filter_step("Multiple Visits Filter")

        # Save the filtered dataset
        self.df.to_csv(output_path, index=False)
        print(f"Filtered dataset saved to: {output_path} ✅")

    def split_alzheimer_groups(self, output_path):
        """Splits subjects into Group 0 (No Alzheimer) and Group 1 (Developed Alzheimer)."""
        print("Splitting subjects into Alzheimer groups...")

        # Group subjects by diagnosis history
        subject_diagnosis = self.df.groupby("subject_id")["DIAGNOSIS"].agg(list)

        # Define groups
        group_0_subjects = subject_diagnosis[subject_diagnosis.apply(lambda x: 3 not in x)].index
        group_1_subjects = subject_diagnosis[subject_diagnosis.apply(lambda x: x[0] in [1, 2] and 3 in x[1:])].index

        # Save to CSV
        group_df = pd.DataFrame({"subject_id": list(group_0_subjects) + list(group_1_subjects),
                                 "Group": [0] * len(group_0_subjects) + [1] * len(group_1_subjects)})
        group_df.to_csv(output_path, index=False)

        # Count total subjets after splitting
        total_subjects = len(group_0_subjects) + len(group_1_subjects)

        # Log the counts
        self.filter_log.append({
        "Step": "After Alzheimer Group Splitting",
        "Unique Subjects": total_subjects,
        })

        print(f"Alzheimer Group Split:{total_subjects} patients remaining.")
        print(f"Alzheimer groups saved to: {output_path} ✅")

    def final_summary(self, output_path):
        """Saves the filtering summary including Alzheimer group splitting."""
        self.save_filter_summary(output_path)

    def stratified_train_test_split(self, group_path, output_path):
        """Splits data into Train (70%) and Test (30%) while maintaining group proportions."""
        print("Performing stratified train-test split...")

        # Load group data
        subject_groups = pd.read_csv(group_path)

        # Stratified split
        train_subjects, test_subjects = train_test_split(subject_groups, test_size=0.3, stratify=subject_groups["Group"], random_state=42)

        # Filter the dataset based on the split
        self.df = self.df[self.df["subject_id"].isin(subject_groups["subject_id"])]

        # Assign train/test labels
        self.df["Split"] = self.df["subject_id"].apply(lambda x: "Train" if x in train_subjects["subject_id"].values else "Test")

        # Save to CSV
        self.df.to_csv(output_path, index=False)
        print(f"Train-Test split dataset saved to: {output_path} ✅")

    def generate_summary_table(self, group_path, summary_output):
        """Generates a summary table with key statistics for Train and Test sets."""
        print("Generating summary table...")

        # Load Alzheimer groups
        subject_groups = pd.read_csv(group_path)
        df_summary = self.df[self.df["subject_id"].isin(subject_groups["subject_id"])].drop_duplicates(subset="subject_id")

        # Function for mean ± SD formatting
        def mean_std(series):
            return f"{series.mean():.2f} ± {series.std():.2f}"

        # Create summary
        summary_data = {
            "N (Unique Subjects)": [df_summary["subject_id"].nunique()],
            "Age (Mean ± SD)": [mean_std(df_summary["subject_age"])],
            "Gender (% Male)": [df_summary["PTGENDER"].eq(1).mean() * 100],
            "Gender (% Female)": [df_summary["PTGENDER"].eq(2).mean() * 100],
            "MMSE Score (Mean ± SD)": [mean_std(df_summary["MMSCORE"])],
            "CDGLOBAL (Mean ± SD)": [mean_std(df_summary["CDGLOBAL"])],
            "APOE4 (% 3/4)": [df_summary["GENOTYPE"].eq("3/4").mean() * 100],
            "APOE4 (% 4/4)": [df_summary["GENOTYPE"].eq("4/4").mean() * 100],
            "ABETA42 (Mean ± SD)": [mean_std(df_summary["ABETA42"])]
        }

        # Convert to DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_output, index=False)
        print(f"Summary table saved to: {summary_output} ✅")

    def generate_train_test_summary(self, summary_output):
        """Generates a summary table with key statistics for Train and Test sets."""
        print("Generating Train-Test summary table...")

        # Function for mean ± SD formatting
        def mean_std(series):
            return f"{series.mean():.2f} ± {series.std():.2f}"
        
        # Initialize summary data
        summary_data = []

        # Generate summary for Train and Test sets
        for split in ["Train", "Test"]:
            subset = self.df[self.df["Split"] == split]
            unique_subjects = subset["subject_id"].nunique()

            # Append summary data
            summary_data.append({
                "Split": split,
                "N (Unique Subjects)": unique_subjects,
                "Age (Mean ± SD)": mean_std(subset["subject_age"]),
                "Gender (% Male)": subset["PTGENDER"].eq(1).mean() * 100,
                "Gender (% Female)": subset["PTGENDER"].eq(2).mean() * 100,
                "MMSE Score (Mean ± SD)": mean_std(subset["MMSCORE"]),
                "CDGLOBAL (Mean ± SD)": mean_std(subset["CDGLOBAL"]),
                "APOE4 (% 3/4)": subset["GENOTYPE"].eq("3/4").mean() * 100,
                "APOE4 (% 4/4)": subset["GENOTYPE"].eq("4/4").mean() * 100,
                "ABETA42 (Mean ± SD)": mean_std(subset["ABETA42"])
            })

        # Convert to DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_output, index=False)
        print(f"Train-Test Summary saved to: {summary_output} ✅")

# Example usage
processor = AlzheimerDataProcessor("data/CUSTOM_TABLE.csv")

# Step 1: Filter the data
processor.filter_data("data/Filtered_CUSTOM_TABLE.csv")

# Step 2: Split Alzheimer groups
processor.split_alzheimer_groups("data/Alzheimer_Groups.csv")

# Step 3: Filtering Chart
processor.final_summary("data/Filtering_Chart.csv")

# Step 4: Train-Test Split only on subjects in Alzheimer_Groups.csv
processor.stratified_train_test_split("data/Alzheimer_Groups.csv", "data/TrainTest_Table.csv")

# Step 5: Generate Summary Table
processor.generate_summary_table("data/Alzheimer_Groups.csv", "data/Summary_Table.csv")

# Step 6: Generate Train-Test Summary Table
processor.generate_train_test_summary("data/TrainTest_Summary.csv")