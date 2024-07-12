import pandas as pd 
import numpy as np

# sample 300 rows
df = pd.read_csv('./BeijingHousingPrices_processed_modified_nan.csv')
df = df.sample(300)
df.to_csv('BeijingHousingPrices_processed_modified_nan_sample.csv', index=False)