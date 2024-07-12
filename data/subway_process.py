import pandas as pd
import numpy as np

df = pd.read_csv('./BeijingHousingPrices_processed_modified_nan_sample.csv')

# Add subway_type column if subway column is 1 then subway_type is true else false
df['subway_type'] = df['subway'].apply(lambda x: True if x == 1 else False)

# if subway is 0, price and totalPrice
# column turns to -price and -totalPrice
df['price'] = df.apply(lambda x: -x['price'] if x['subway'] == 0 else x['price'], axis=1)
df['totalPrice'] = df.apply(lambda x: -x['totalPrice'] if x['subway'] == 0 else x['totalPrice'], axis=1)

# Save the modified DataFrame back to a CSV file
df.to_csv('BeijingHousingPrices_processed_modified_nan_sampled_subway.csv', index=False)
