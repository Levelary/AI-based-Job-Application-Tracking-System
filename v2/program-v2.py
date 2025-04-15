from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd

# Load your data
df = pd.read_csv("v2/candidate_data.csv", names=["ID", "Education_Level", "Experience_Years", "Technical_Skills"], skiprows=1)

# Map Education Level to Numeric Values
education_mapping = {
    "high school": 1,
    "bachelor's": 2,
    "master's": 3,
    "phd": 4
}
df["Education_Level"] = df["Education_Level"].astype(str).str.strip().str.lower()
df["Education_Level_Numeric"] = df["Education_Level"].map(education_mapping)

# Count the number of skills (split by commas)
df['Skills_Count'] = df['Technical_Skills'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

# Label candidates based on the number of skills
df['label'] = df['Skills_Count'].apply(lambda x: 1 if x >= 5 else 0)

# Convert Education_Level to a binary protected attribute (adjust logic to your dataset)
df['Education_Group'] = df['Education_Level_Numeric'].apply(lambda x: 1 if x >= 2 else 0)

# Drop unnecessary columns
features = ['Experience_Years', 'Education_Group']
label_name = 'label'
protected_attr = 'Education_Group'

# Ensure numeric conversion for all necessary columns
df['Experience_Years'] = pd.to_numeric(df['Experience_Years'], errors='coerce')
df['Skills_Count'] = pd.to_numeric(df['Skills_Count'], errors='coerce')

# Handle missing values (if any) by filling them with a suitable value (e.g., 0)
df.fillna(0, inplace=True)

# Check that the necessary columns are numeric
print(df[['Experience_Years', 'Skills_Count']].dtypes)

# Convert to BinaryLabelDataset
dataset = BinaryLabelDataset(df=df,
                             label_names=[label_name],
                             protected_attribute_names=[protected_attr])

# Set privileged and unprivileged group definitions
privileged_groups = [{'Education_Group': 1}]
unprivileged_groups = [{'Education_Group': 0}]

# Apply Reweighing for fairness mitigation
rw = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transformed = rw.fit_transform(dataset)

# Evaluate fairness before reweighing
metric_before = BinaryLabelDatasetMetric(dataset,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups)
print("🔍 Before Reweighing:")
print("Disparate Impact:", metric_before.disparate_impact())
print("Statistical Parity Difference:", metric_before.statistical_parity_difference())
print("Mean Difference:", metric_before.mean_difference())

# Evaluate fairness after reweighing (without changing model)
metric_after = BinaryLabelDatasetMetric(dataset_transformed,
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups)
print("\n✅ After Reweighing (Evaluation Only):")
print("Disparate Impact:", metric_after.disparate_impact())
print("Statistical Parity Difference:", metric_after.statistical_parity_difference())
print("Mean Difference:", metric_after.mean_difference())

# Check instance weights
print("\n🎯 Instance Weights (first 10):")
print(dataset_transformed.instance_weights[:10])
