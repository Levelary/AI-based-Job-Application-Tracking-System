# candidate_ranker.py
import pandas as pd
from aif360.datasets import StandardDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

def get_ranked_candidates():
    df = pd.read_csv("candidate_profiles.csv", names=["ID", "Education_Level", "Experience_Years", "Technical_Skills"], skiprows=1, engine='python')

    education_mapping = {
        "high school": 1,
        "bachelor's": 2,
        "master's": 3,
        "phd": 4
    }

    df["Education_Level"] = df["Education_Level"].astype(str).str.strip().str.lower()
    df["Technical_Skills"] = df["Technical_Skills"].astype(str).str.strip().str.lower()
    
    df["Education_Level_Numeric"] = df["Education_Level"].map(education_mapping)
    df["Education_Group"] = df["Education_Level_Numeric"].apply(lambda x: 0 if x == 1 else 1).astype(int)
    df["Skills_Count"] = df["Technical_Skills"].apply(lambda x: len(x.split(',')))
    
    features = ["Experience_Years", "Skills_Count"]
    target = "Education_Level_Numeric"

    dataset = StandardDataset(
        df,
        label_name=target,
        favorable_classes=[3, 4],
        protected_attribute_names=["Education_Group"],
        privileged_classes=[[1]],
        features_to_drop=["ID", "Education_Level", "Technical_Skills"]
    )
    
    aif_dataset = dataset.copy()

    rw = Reweighing(unprivileged_groups=[{'Education_Group': 0}], privileged_groups=[{'Education_Group': 1}])
    dataset_transformed = rw.fit_transform(dataset)

    df_transformed = dataset_transformed.convert_to_dataframe()[0]
    X = df_transformed[features]
    y = df_transformed[target]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    predictions = model.predict(X)
    ranked_df = pd.DataFrame({
        "ID": df["ID"],
        "Predicted_Rank": predictions
    })

    ranked_df = ranked_df.sort_values(by="Predicted_Rank", ascending=False).reset_index(drop=True)

    return ranked_df.to_dict(orient="records")
