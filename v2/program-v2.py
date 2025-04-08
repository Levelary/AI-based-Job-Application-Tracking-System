import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

# Load data
df = pd.read_csv("C:\\Users\\bhara\\OneDrive\\Desktop\\major_project\\v2\\candidate_data.csv")

# ---------------------
# Feature Engineering
# ---------------------

# Encode Education
education_mapping = {
    "High School": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4
}
df["Education_Level"] = df["Education_Level"].map(education_mapping)

# Create binary flags for skills
possible_skills = ["python", "java", "c++", "ml", "data analysis", "web development", "sql", "cloud"]
for skill in possible_skills:
    df[f"has_{skill.replace(' ', '_')}"] = df["Technical_Skills"].str.lower().str.contains(skill).astype(int)

# Total skill count
df["total_skills"] = df[[f"has_{skill.replace(' ', '_')}" for skill in possible_skills]].sum(axis=1)

# Simulate skill score
if "Skill_Score" not in df.columns:
    np.random.seed(42)
    df["Skill_Score"] = np.clip(df["total_skills"] * 10 + np.random.randint(-5, 10, size=len(df)), 50, 100)

# AI Hiring Decision
df["AI_Hiring_Decision"] = (df["Skill_Score"] >= 70).astype(int)

# Define protected attribute (e.g., low vs high total skills)
df["Skill_Level"] = pd.cut(df["total_skills"], bins=[0, 2, 5, 8], labels=[1, 2, 3])

# Clean up
df = df[["Education_Level", "Experience_Years", "Skill_Score", "total_skills", "AI_Hiring_Decision", "Skill_Level"] +
        [f"has_{skill.replace(' ', '_')}" for skill in possible_skills]]
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# ---------------------
# Train/Test Split
# ---------------------
X = df.drop(columns=["AI_Hiring_Decision", "Skill_Level"])
y = df["AI_Hiring_Decision"]
protected_attr = df["Skill_Level"]

X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X, y, protected_attr, test_size=0.3, random_state=42
)

# ---------------------
# Model Training
# ---------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------
# Fairness Metrics
# ---------------------
metric_frame = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=prot_test
)

print("\n--- Fairness Metric By Skill Level Group ---")
print(metric_frame.by_group)
print("\n--- Overall Metrics ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Demographic Parity Difference:", demographic_parity_difference(y_test, y_pred, sensitive_features=prot_test))
print("Equalized Odds Difference:", equalized_odds_difference(y_test, y_pred, sensitive_features=prot_test))

# ---------------------
# AIF360 Fairness Check
# ---------------------
aif_df = df.copy()
aif_dataset = StandardDataset(
    aif_df,
    label_name="AI_Hiring_Decision",
    favorable_classes=[1],
    protected_attribute_names=["Skill_Level"],
    privileged_classes=[[3]]
)

metric = BinaryLabelDatasetMetric(aif_dataset,
                                  privileged_groups=[{"Skill_Level": 3}],
                                  unprivileged_groups=[{"Skill_Level": 1}])

print("\nAIF360 Bias Metrics:")
print("Disparate Impact:", metric.disparate_impact())
print("Statistical Parity Difference:", metric.statistical_parity_difference())

# Reweighing for mitigation
reweigher = Reweighing(privileged_groups=[{"Skill_Level": 3}],
                       unprivileged_groups=[{"Skill_Level": 1}])

fair_dataset = reweigher.fit_transform(aif_dataset)

fair_metric = BinaryLabelDatasetMetric(fair_dataset,
                                       privileged_groups=[{"Skill_Level": 3}],
                                       unprivileged_groups=[{"Skill_Level": 1}])

print("\nPost-Mitigation Fairness:")
print("Fair Disparate Impact:", fair_metric.disparate_impact())
print("Fair Statistical Parity Difference:", fair_metric.statistical_parity_difference())
