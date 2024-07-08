import pandas as pd

# Load the CSV file
df = pd.read_csv('BeijingHousingPrices_processed_modified_nan.csv')

# Dictionary mapping district codes to district names
district_mapping = {
    1: 'Dongcheng District',
    2: 'Fengtai District',
    3: 'Tongzhou District',
    4: 'Daxing District',
    5: 'Fangshan District',
    6: 'Changping District',
    7: 'Chaoyang District',
    8: 'Haidian District',
    9: 'Shijingshan District',
    10: 'Xicheng District',
    11: 'Pinggu District',
    12: 'Mentougou District',
    13: 'Shunyi District'
}

# Replace the district column using the mapping
df['district'] = df['district'].map(district_mapping)

# Save the modified DataFrame back to CSV
df.to_csv('BeijingHousingPrices_processed_modified_region.csv', index=False)