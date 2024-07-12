import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv('./../data/BeijingHousingPrices_processed_modified.csv')

# Replace 'Unknown' with NaN
df.replace('Unknown', np.nan, inplace=True)

# Fill missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

# Define categorical columns
categorical_columns = ['floor_num', 'renovationCondition', 'buildingStructure', 'fiveYearsProperty']

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Columns of interest for the correlation matrix
base_columns_of_interest = [
    'followers', 'price', 'floor_type', 'constructionTime', 
    'ladderRatio', 'district', 'communityAverage', 'elevator', 'subway'
]

# Extend columns_of_interest with one-hot encoded columns
one_hot_encoded_columns = [col for col in df.columns if any(prefix in col for prefix in categorical_columns)]
columns_of_interest = base_columns_of_interest + one_hot_encoded_columns

# Scale the data
scaler = MinMaxScaler()
df[columns_of_interest] = scaler.fit_transform(df[columns_of_interest])

# Calculate the correlation matrix
correlation_matrix = df[columns_of_interest].corr()

# Save the correlation matrix to a CSV file
correlation_matrix.to_csv('./../data/correlation_matrix.csv')

print("Correlation matrix calculation is done! Check the 'correlation_matrix.csv' file.")
