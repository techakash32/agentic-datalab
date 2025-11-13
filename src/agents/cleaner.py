import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

class DataCleanerAgent:
    def run(self, df):
        df = df.drop_duplicates()
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("MISSING")
        encoders = {}
        for col in df.select_dtypes(include=["object", "category"]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df, encoders
