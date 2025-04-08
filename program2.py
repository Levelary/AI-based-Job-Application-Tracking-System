# pip install aif360 fairness-indicators transformers tensorflow torch pandas numpy scikit-learn
# pip install 'aif360[Reductions]'
# pip install 'aif360[inFairness]'
# pip install fairlearn
# pip install tensorflow-model-analysis


import pandas as pd
import numpy as np
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference


# Load candidate data
df = pd.read_csv("C:\\Users\\bhara\\OneDrive\\Desktop\\major_project\\v2\\candidate_data.csv")

# Simulate AI skill evaluation
np.random.seed(42)
if "Skill_Score" not in df.columns:
    df["Skill_Score"] = np.random.randint(50, 100, size=len(df))  

# AI Decision: Passing threshold is 70
df["AI_Hiring_Decision"] = (df["Skill_Score"] >= 70).astype(int)

# Convert categorical attributes
# if "Education_Level" in df.columns:
#     df["Education_Level"] = df["Education_Level"].map({
#         "High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4
#     })

# if "Technical_Skills" in df.columns:
#     df["Technical_Skills"] = df["Technical_Skills"].map({
#         "Low": 1, "Medium": 2, "High": 3
#     })
education_mapping = {
    "High School": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4
}
skills_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}


df["Education_Level"] = df["Education_Level"].map(education_mapping)
df["Technical_Skills"] = df["Technical_Skills"].map(skills_mapping)

# Define skill levels as protected groups
# df["Skill_Level"] = pd.cut(df["Skill_Score"], bins=[50, 69, 89, 100], labels=["Low", "Medium", "High"])
df["Skill_Level"] = pd.cut(df["Skill_Score"], bins=[50, 69, 89, 100], labels=[1, 2, 3])



# Ensure all columns have proper numerical values
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()  # Remove rows with invalid conversions


# ------------------------------
# 5. Split features and target
# ------------------------------
X = df[["Education_Level", "Technical_Skills", "Experience_Years", "Skill_Score"]]
y = df["AI_Hiring_Decision"]
protected_attr = df["Skill_Level"]

X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
    X, y, protected_attr, test_size=0.3, random_state=42
)

# ------------------------------
# 6. Train Logistic Regression model
# ------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------------
# 7. Fairness Evaluation
# ------------------------------
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate
    },
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

# --------------------------

# Prepare dataset for AIF360
ai_dataset = StandardDataset(
    df, 
    label_name="AI_Hiring_Decision", 
    favorable_classes=[1],  
    protected_attribute_names=["Skill_Level"],  
    privileged_classes=[[3]]  #"High"
)

# Check bias
metric = BinaryLabelDatasetMetric(ai_dataset, 
                                  privileged_groups=[{"Skill_Level": 3}],  #"High"
                                  unprivileged_groups=[{"Skill_Level": 1}]) #"Low"

print(f"Skill Bias: {metric.disparate_impact()}")
print(f"Statistical Parity Difference: {metric.statistical_parity_difference()}")

# Apply bias mitigation (Reweighing)
reweigh = Reweighing(privileged_groups=[{"Skill_Level": 3}], #"High"
                     unprivileged_groups=[{"Skill_Level": 1}]) #"Low"

fair_ai_dataset = reweigh.fit_transform(ai_dataset)

# Verify fairness improvement
fair_metric = BinaryLabelDatasetMetric(fair_ai_dataset, 
                                       privileged_groups=[{"Skill_Level": 3}], #"High"
                                       unprivileged_groups=[{"Skill_Level": 1}]) #"Low"

print(f"Fair AI Disparate Impact: {fair_metric.disparate_impact()}")
print(f"Fair AI Statistical Parity Difference: {fair_metric.statistical_parity_difference()}")


#
#
# tf processing
#
#


# import tensorflow as tf
# import tensorflow_model_analysis as tfma

# # Save dataset in TFRecord format
# df.to_csv("fairness_data.csv", index=False)

# # Convert CSV to TF Dataset
# def parse_csv(line):
#     """Convert CSV line into a dictionary"""
#     feature_names = list(df.columns)
#     defaults = [tf.float32] * len(feature_names)  # Convert all to float
#     parsed_line = tf.io.decode_csv(line, record_defaults=defaults)
#     return dict(zip(feature_names, parsed_line))

# # Load data as TensorFlow Dataset
# dataset = tf.data.experimental.make_csv_dataset("fairness_data.csv", batch_size=1, num_epochs=1)

# # Define fairness evaluation config
# eval_config = tfma.EvalConfig(
#     model_specs=[tfma.ModelSpec(label_key="AI_Hiring_Decision")],
#     slicing_specs=[tfma.SlicingSpec()],
#     metrics_specs=[
#         tfma.MetricsSpec(metrics=[
#             tfma.MetricConfig(class_name="FairnessIndicators"),
#         ])
#     ]
# )

# # Run evaluation
# eval_result = tfma.run_model_analysis(eval_config=eval_config, data_location="fairness_data.csv")
# tfma.view.render_fairness_indicator(eval_result)


# #
# #
# # hugging face model
# #
# #


# from transformers import pipeline

# # Load NLP model for sentiment/skill assessment
# soft_skill_analyzer = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")

# # Example candidate responses
# responses = [
#     "I am a great team player with excellent leadership skills.",
#     "I work well under pressure and adapt to challenges easily.",
#     "I have deep technical expertise in machine learning."
# ]

# # Analyze soft skills
# for response in responses:
#     analysis = soft_skill_analyzer(response)
#     print(f"Response: {response}\nSoft Skill Score: {analysis}\n")
