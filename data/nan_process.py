import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('./BeijingHousingPrices_processed_modified.csv')

df.replace('Unknown', np.nan, inplace=True)
df.replace('Nan', np.nan, inplace=True)

for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

# Save the modified DataFrame back to a CSV file
df.to_csv('BeijingHousingPrices_processed_modified_nan.csv', index=False)