import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./../data/BeijingHousingPrices_processed_modified.csv')

df.replace('Unknown', np.nan, inplace=True)

# use the mode for categorical columns and the mean for numerical columns to fill missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].mean(), inplace=True)

categorical_columns = ['floor_type', 'renovationCondition', 'buildingStructure', 'elevator', 'fiveYearsProperty', 'subway']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

columns_of_interest = [
    'followers', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom',
    'floor_num', 'constructionTime', 'ladderRatio', 'district', 'communityAverage'
]

columns_of_interest.extend([col for col in df.columns if any(prefix in col for prefix in categorical_columns)])

scaler = MinMaxScaler()
df[columns_of_interest + ['price']] = scaler.fit_transform(df[columns_of_interest + ['price']])

correlation_dict = {}
for column in columns_of_interest:
    correlation_dict[column] = df['price'].corr(df[column])

correlation_df = pd.DataFrame(list(correlation_dict.items()), columns=['type', 'correlation'])

correlation_df.to_csv('./../result/data/correlation_with_price.csv', index=False)

print("Correlation calculation is done! Check the 'correlation_with_price.csv' file.")
