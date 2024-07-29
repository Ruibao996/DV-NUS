import pandas as pd
import numpy as np

data_path = 'BeijingHousingPrices_processed_modified_nan.csv'

# Specify low_memory=False to avoid warning
df = pd.read_csv(data_path, low_memory=False)

df['tradeTime'] = pd.to_datetime(df['tradeTime'], format='%m/%d/%Y')

df = df[df['tradeTime'] >= '2010-01-01']

df.set_index('tradeTime', inplace=True)

df.sort_values(by='tradeTime', inplace=True)

# Ensure only numeric columns are included for mean calculation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

monthly_avg = df[numeric_cols].resample('ME').mean()

monthly_avg.reset_index(inplace=True)

# Assuming 'totalPrice' is a numeric column, if not, ensure it is converted to numeric
monthly_avg = monthly_avg[['tradeTime', 'totalPrice']]
monthly_avg.rename(columns={'tradeTime': 'month', 'totalPrice': 'averageTotalPrice'}, inplace=True)

print(monthly_avg.head())

monthly_avg.to_csv('./../result/monthly_total_prices.csv', index=False)