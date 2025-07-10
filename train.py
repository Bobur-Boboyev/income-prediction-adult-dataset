import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

def data_preparation(path):
    data = pd.read_csv(path)
    data = data.replace('?', np.nan)
    cleaned_data = data.dropna().copy()

    encoders = {}
    for col in cleaned_data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        cleaned_data[col] = le.fit_transform(cleaned_data[col])
        encoders[col] = le

    x = cleaned_data.drop("income", axis=1)
    y = cleaned_data["income"]

    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)
    scaled_x_df = pd.DataFrame(scaled_x, columns=x.columns)

    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(list(x.columns), 'columns.pkl')

    return train_test_split(scaled_x_df, y, test_size=0.2, random_state=42)
