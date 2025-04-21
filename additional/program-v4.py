import pandas as pd

df = pd.read_csv("v2/candidate_data.csv", names=["ID", "Education_Level", "Experience_Years", "Technical_Skills"], skiprows=1, engine='python')
# print("Rows after cleaning:", len(df))
# print(df.head())

# print("Before mapping:")
# print(df["Education_Level"].unique())

# Map Education to Numeric
education_mapping = {
    "high school": 1,
    "bachelor's": 2,
    "master's": 3,
    "phd": 4
}
df["Education_Level"] = df["Education_Level"].astype(str).str.strip().str.lower()
df["Technical_Skills"] = df["Technical_Skills"].astype(str).str.lower().str.strip()

df["Education_Level_Numeric"] = df["Education_Level"].map(education_mapping)


# Count number of skills
df["Skills_Count"] = df["Technical_Skills"].apply(lambda x: len(x.split(',')))

# Create new column for protected group (e.g. whether education level is above high school)
df["Education_Group"] = df["Education_Level_Numeric"].apply(lambda x: 0 if x == 1 else 1)
# 0 = unprivileged (high school), 1 = privileged (bachelor’s and above)
df["Education_Group"] = df["Education_Group"].astype(int)
from aif360.datasets import StandardDataset

features = ["Experience_Years", "Skills_Count"]
target = "Education_Level_Numeric"
# setting up aif360 dataset
dataset = StandardDataset(
    df,
    label_name=target,
    favorable_classes=[3, 4],
    protected_attribute_names=["Education_Group"],
    privileged_classes=[[1]],
    features_to_drop=["ID", "Education_Level", "Technical_Skills"]
)

# Show the dataset summary
# Check dataset content
print("Features:", dataset.features)
print("Labels:", dataset.labels)






######################################
# Training model for ranking

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Prepare features and target variables
X = df[features]
y = df[target]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get the predictions (this will be used for ranking)
y_pred = model.predict(X_test)

# Rank students based on predicted values
ranked_students = pd.DataFrame({'ID': df['ID'], 'Predicted_Rank': model.predict(df[features])})
ranked_students = ranked_students.sort_values(by='Predicted_Rank', ascending=False)

# Display the ranked students
print(ranked_students)




######################################
# Bias evaluation and mitigation

#checking bias
from aif360.metrics import BinaryLabelDatasetMetric

# Convert dataset to AIF360 format
aif_dataset = dataset.copy()

# Calculate disparity for 'Education_Level' as protected attribute
metric = BinaryLabelDatasetMetric(dataset,
                                  privileged_groups=[{'Education_Group': 1}],
                                  unprivileged_groups=[{'Education_Group': 0}])
print("Disparate Impact:", metric.disparate_impact())
print("Statistical Parity Difference:", metric.statistical_parity_difference())
print("Mean Difference:", metric.mean_difference())

#mitigating bias using preprocessing
from aif360.algorithms.preprocessing import Reweighing

# Apply Reweighing to reduce bias
rw = Reweighing(unprivileged_groups=[{'Education_Group': 0}], privileged_groups=[{'Education_Group': 1}])

dataset_transformed = rw.fit_transform(aif_dataset)

# Get metrics again after preprocessing
metric_after_reweighting = BinaryLabelDatasetMetric(dataset_transformed,
    privileged_groups=[{"Education_Group": 1}],
    unprivileged_groups=[{"Education_Group": 0}]
)
# Disparate Impact = Pr(Y=1 | unprivileged) / Pr(Y=1 | privileged)
print('Disparate Impact After Reweighting:', metric_after_reweighting.disparate_impact())

# ranking after mitigation
# Train a new model with transformed dataset if required, and rerank
# For simplicity, we assume the model was retrained on the reweighted dataset
# Predict new ranks after transformation (assuming the reweighing works with the model)
ranked_students_after_mitigation = pd.DataFrame({'ID': df['ID'], 'Predicted_Rank': model.predict(df[features])})
ranked_students_after_mitigation = ranked_students_after_mitigation.sort_values(by='Predicted_Rank', ascending=False)

# Display the new ranked students
print(ranked_students_after_mitigation)


# Convert reweighted AIF360 dataset back to DataFrame
df_transformed = dataset_transformed.convert_to_dataframe()[0]
X_rw = df_transformed[features]
y_rw = df_transformed[target]

model_rw = RandomForestRegressor(n_estimators=100, random_state=42)
# weights = df_transformed['instance_weights'] for reducing bias
model_rw.fit(X_rw, y_rw)#, sample_weight=weights)

ranked_students_rw = pd.DataFrame({'ID': df['ID'], 'Predicted_Rank': model_rw.predict(df[features])})
ranked_students_rw = ranked_students_rw.sort_values(by='Predicted_Rank', ascending=False)
print(ranked_students_rw) 
print('Disparate Impact After Reweighting:', metric_after_reweighting.disparate_impact())

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.bar(ranked_students_rw["ID"].astype(str), ranked_students_rw["Predicted_Rank"], color="skyblue")
plt.xlabel("Candidate ID")
plt.ylabel("Predicted Capability Score")
plt.title("Student Ranking (Fairness-Aware)")
plt.tight_layout()
plt.show()
