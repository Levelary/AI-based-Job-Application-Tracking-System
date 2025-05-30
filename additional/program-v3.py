import pandas as pd
import numpy as np
import inFairness
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing


import tensorflow as tf


# ---------------------
# Load and Preprocess Data
# ---------------------
# import pandas as pd

# data = [
#     ["Bachelor's", 3, "Python, Java, SQL"],
#     ["Master's", 5, "ML, Python, Data Analysis, Cloud"],
#     ["High School", 1, "C++, Web Development"],
#     ["PhD", 7, "Python, ML, Data Analysis, Cloud, SQL"]
# ]

# df = pd.DataFrame(data, columns=["Education_Level", "Experience_Years", "Technical_Skills"])
# df.to_csv("candidate_data.csv", index=False, quoting=1)  # quoting=1 for QUOTE_ALL

# df = pd.read_csv("C:\\Users\\bhara\\OneDrive\\Desktop\\major_project\\v2\\candidate_data.csv")
df = pd.read_csv("v2/candidate_data.csv", names=["ID", "Education_Level", "Experience_Years", "Technical_Skills"], skiprows=1, engine='python')
# print("Rows after cleaning:", len(df))
# print(df.head())

print("Before mapping:")
print(df["Education_Level"].unique())

# Map Education to Numeric
education_mapping = {
    "high school": 1,
    "bachelor's": 2,
    "master's": 3,
    "phd": 4
}
df["Education_Level"] = df["Education_Level"].astype(str).str.strip().str.lower()
# df["Technical_Skills"] = df["Technical_Skills"].astype(str).str.lower().str.strip()

df["Education_Level"] = df["Education_Level"].map(education_mapping)

print("After mapping:")
print(df["Education_Level"].unique())

# Define list of possible skills
possible_skills = ["python", "java", "c++", "ml", "data analysis", "web development", "sql", "cloud"]

# Ensure skills column is string
# df["Technical_Skills"] = df["Technical_Skills"].astype(str).str.lower()

# Binary encode each skill
for skill in possible_skills:
    df[f"has_{skill.replace(' ', '_')}"] = df["Technical_Skills"].str.contains(skill).astype(int)

# Total skills as a derived feature
df["total_skills"] = df[[f"has_{skill.replace(' ', '_')}" for skill in possible_skills]].sum(axis=1)

# Simulate Skill Score
np.random.seed(42)
df["Skill_Score"] = np.clip(df["total_skills"] * 10 + np.random.randint(-5, 10, size=len(df)), 50, 100)

# Target variable: AI decision
df["AI_Hiring_Decision"] = (df["Skill_Score"] >= 70).astype(int)

# Derive Skill Level (protected attribute) based on total_skills
df["Skill_Level"] = pd.cut(df["total_skills"], bins=[0, 2, 5, 8], labels=[1, 2, 3])  # 1 = low, 2 = med, 3 = high

# Drop unnecessary columns
df = df[["Education_Level", "Experience_Years", "Skill_Score", "total_skills", "AI_Hiring_Decision", "Skill_Level"] +
        [f"has_{skill.replace(' ', '_')}" for skill in possible_skills]]
# df = df.apply(pd.to_numeric, errors='coerce').dropna()
# Only convert specific columns to numeric (if needed)
numeric_cols = ["Education_Level", "Experience_Years", "Skill_Score", "total_skills", "AI_Hiring_Decision"] + \
               [f"has_{skill.replace(' ', '_')}" for skill in possible_skills]

# Convert only those
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

print("Before dropna:", df.shape)
print("Null values per column:\n", df[numeric_cols].isnull().sum())

# Now drop rows with any NaN in those columns
df = df.dropna(subset=numeric_cols)

# ---------------------
# Split Data
# ---------------------
X = df.drop(columns=["AI_Hiring_Decision", "Skill_Level"])
y = df["AI_Hiring_Decision"]
protected_attr = df["Skill_Level"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Protected attr shape:", protected_attr.shape)


X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X, y, protected_attr, test_size=0.3, random_state=42
)
print(df.columns)
print(df.head())
print(df.shape)

print(X.shape, y.shape, protected_attr.shape)

# ---------------------
# Prepare AIF360 Dataset
# ---------------------
train_df = X_train.copy()
train_df["AI_Hiring_Decision"] = y_train.values
train_df["Skill_Level"] = prot_train.values

test_df = X_test.copy()
test_df["AI_Hiring_Decision"] = y_test.values
test_df["Skill_Level"] = prot_test.values

train_aif = StandardDataset(train_df,
                            label_name="AI_Hiring_Decision",
                            favorable_classes=[1],
                            protected_attribute_names=["Skill_Level"],
                            privileged_classes=[[3]])

test_aif = StandardDataset(test_df,
                           label_name="AI_Hiring_Decision",
                           favorable_classes=[1],
                           protected_attribute_names=["Skill_Level"],
                           privileged_classes=[[3]])

# ---------------------
# Train Adversarial Debiasing Model
# ---------------------
sess = tf.Session()
adv_debiaser = AdversarialDebiasing(
    privileged_groups=[{"Skill_Level": 3}],
    unprivileged_groups=[{"Skill_Level": 1}],
    scope_name='adv_debiasing',
    sess=sess,
    num_epochs=50,
    batch_size=16,
    debias=True
)
adv_debiaser.fit(train_aif)

# ---------------------
# Predict and Evaluate Fairness
# ---------------------
preds = adv_debiaser.predict(test_aif)

metric_adv = ClassificationMetric(test_aif, preds,
                                  privileged_groups=[{"Skill_Level": 3}],
                                  unprivileged_groups=[{"Skill_Level": 1}])

print("\n--- Fairness Metrics After Adversarial Debiasing ---")
print("Accuracy:", accuracy_score(test_aif.labels, preds.labels))
print("Disparate Impact:", metric_adv.disparate_impact())
print("Statistical Parity Difference:", metric_adv.statistical_parity_difference())
print("Equal Opportunity Difference:", metric_adv.equal_opportunity_difference())
