import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('./BeijingHousingPrices_processed.csv')

# Replace values in 'floor_type' column
df['floor_type'] = df['floor_type'].replace({'Top': 4, 'High': 3, 'Middle': 2, 'Low': 1})

# Delete abnormal data in each column by normal distribution and treat them as NaN
for column in ['square', 'ladderRatio']: 
    mean = df[column].mean()
    std = df[column].std()
    upper_limit = mean + 3 * std
    lower_limit = mean - 3 * std
    df[column] = df[column].apply(lambda x: np.nan if x > upper_limit or x < lower_limit else x)

# Save the modified DataFrame back to a CSV file
df.to_csv('BeijingHousingPrices_processed_modified.csv', index=False)
print("Data processing is done! Check the 'BeijingHousingPrices_processed_modified.csv' file.")