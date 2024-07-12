import pandas as pd
import numpy as np

df = pd.read_csv('./BeijingHousingPrices_processed_modified_nan.csv')
# AC_subway=communityAverage if subway is 1 else 0
df['AC_subway'] = df.apply(lambda x: x['communityAverage'] if x['subway'] == 1 else 0, axis=1)
df['AC_subway_nan'] = df.apply(lambda x: x['communityAverage'] if x['subway'] == 0 else 0, axis=1)

# column turns to -price and -totalPrice
df['price_subway'] = df.apply(lambda x: -x['price'] if x['subway'] == 0 else x['price'], axis=1)
df['totalPrice_subway'] = df.apply(lambda x: -x['totalPrice'] if x['subway'] == 0 else x['totalPrice'], axis=1)

# Save the modified DataFrame back to a CSV file
df.to_csv('BeijingHousingPrices_processed_modified_nan_subway.csv', index=False)