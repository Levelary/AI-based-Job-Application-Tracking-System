# pip install aif360 pandas numpy matplotlib sckit-learn

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
# from aif360.datasets import StandardDataset
# from aif360.metrics import BinaryLabelDatasetMetric
df = pd.read_csv('candidate_data.csv')

import pandas as pd
import numpy as np

# Load candidate data
df = pd.read_csv("candidate_data.csv")

# Simulating AI skill evaluation
np.random.seed(42)
df["Skill_Score"] = np.random.randint(50, 100, size=len(df))  # Random scores (50-100)

# AI Decision: Passing threshold is 70
df["AI_Hiring_Decision"] = (df["Skill_Score"] >= 70).astype(int)  

# Convert categorical fields to numerical
print(df.columns)  # Check available columns

if "Education_Level" in df.columns:
    df["Education_Level"] = df["Education_Level"].map({
        "High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4
    })

if "Technical_Skills" in df.columns:
    df["Technical_Skills"] = df["Technical_Skills"].map({
        "Low": 1, "Medium": 2, "High": 3
    })


# Save AI decision data
df.to_csv("ai_skill_bias_data.csv", index=False)

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric

# Load AI decision data into AIF360
ai_dataset = StandardDataset(
    df, 
    label_name="AI_Hiring_Decision", 
    favorable_classes=[1],  
    protected_attribute_names=["Skill_Score"],  
    privileged_classes=[[80, 100]]  # Candidates scoring above 80 are 'privileged'
)

# Check bias
metric = BinaryLabelDatasetMetric(ai_dataset, 
                                  privileged_groups=[{"Skill_Score": 80}], 
                                  unprivileged_groups=[{"Skill_Score": 70}])

print(f"Skill Bias: {metric.disparate_impact()}")
print(f"Statistical Parity Difference: {metric.statistical_parity_difference()}")

from aif360.algorithms.preprocessing import Reweighing

# Apply reweighing to adjust unfair skill scoring
reweigh = Reweighing(privileged_groups=[{"Skill_Score": 80}], 
                     unprivileged_groups=[{"Skill_Score": 70}])

fair_ai_dataset = reweigh.fit_transform(ai_dataset)

# Check bias after mitigation
fair_metric = BinaryLabelDatasetMetric(fair_ai_dataset, 
                                       privileged_groups=[{"Skill_Score": 80}], 
                                       unprivileged_groups=[{"Skill_Score": 70}])

print(f"Fair AI Disparate Impact: {fair_metric.disparate_impact()}")
print(f"Fair AI Statistical Parity Difference: {fair_metric.statistical_parity_difference()}")

