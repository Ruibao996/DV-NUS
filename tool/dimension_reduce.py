import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def pca_dimension_reduction(data, components=2):
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

def mse_dimension_reduction(data, components=2):
    # Assuming MSE refers to multidimensional scaling (MDS)
    from sklearn.manifold import MDS
    mds = MDS(n_components=components, dissimilarity='euclidean', random_state=0)
    reduced_data = mds.fit_transform(data)
    return reduced_data

def tsne_dimension_reduction(data, components=2):
    tsne = TSNE(n_components=components, random_state=0)
    reduced_data = tsne.fit_transform(data)
    return reduced_data

# Random sample of the data 1/1000
def random_sample(data, sample_size=0.005):
    # Calculate the number of samples to select
    num_samples = int(data.shape[0] * sample_size)
    
    # Generate random indices
    random_indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    
    # Use the indices to select rows
    sample = data[random_indices, :]
    
    return sample

def DR_main(input_path, output_path, type='PCA'):
    df = pd.read_csv(input_path, dtype={'id': str})

    df.replace('Unknown', np.nan, inplace=True)
    df.replace('Nan', np.nan, inplace=True)

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna(df[column].mode().iloc[0])
        else:
            df[column] = df[column].fillna(df[column].mean())

    # Try to convert constructionTime to float
    try:
        df['constructionTime'] = df['constructionTime'].astype(float)
    except ValueError:
        print("Error converting 'constructionTime' to float. Check data for non-numeric values.")
        return

    categorical_columns = ['buildingStructure', 'fiveYearsProperty', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'district']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    columns_of_interest = [
        'id', 'floor_num', 'elevator', 'subway', 'floor_type', 'renovationCondition',
        'constructionTime', 'ladderRatio', 'communityAverage', 'totalPrice'
    ]

    # Extend columns_of_interest with generated dummy variable columns
    columns_of_interest.extend([col for col in df.columns if any(prefix in col for prefix in categorical_columns)])

    # Check which columns are missing
    missing_columns = [col for col in columns_of_interest if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing from the DataFrame: {missing_columns}")
        columns_of_interest = [col for col in columns_of_interest if col in df.columns]

    data = df[columns_of_interest]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop('id', axis=1))  # Exclude 'id' from scaling

    original_data = df[columns_of_interest]

    if type == 'PCA':
        # PCA
        pca_result = pca_dimension_reduction(data_scaled)
        pca_df = pd.DataFrame(pca_result, index=df.index, columns=['PCA1', 'PCA2'])
        pca_df = pd.concat([original_data.reset_index(drop=True), pca_df], axis=1)
        pca_df.to_csv(output_path, index=False)
        print('PCA reduction is done! Check the "pca_result.csv" file.')
    elif type == 'MSE':
        # MSE
        data_scaled_mse = random_sample(data_scaled)
        # Create column names for the sampled data
        columns = data.drop('id', axis=1).columns  # Use the columns of the original scaled data
        data_scaled_mse_df = pd.DataFrame(data_scaled_mse, columns=columns)
        data_scaled_mse_df.to_csv('./result/data/mse_sampled_data.csv', index=False)
        mse_result = mse_dimension_reduction(data_scaled_mse)
        mse_df = pd.DataFrame(mse_result, columns=['MSE1', 'MSE2'])
        mse_df = pd.concat([original_data.reset_index(drop=True).iloc[:len(mse_df)], mse_df], axis=1)
        mse_df.to_csv(output_path, index=False)
        print('MSE reduction is done! Check the "mse_result.csv" file.')
    else:
        # t-SNE
        data_scaled_tsne = random_sample(data_scaled)
        # Create column names for the sampled data
        columns = data.drop('id', axis=1).columns  # Use the columns of the original scaled data
        data_scaled_tsne_df = pd.DataFrame(data_scaled_tsne, columns=columns)
        data_scaled_tsne_df.to_csv('./result/data/tsne_sampled_data.csv', index=False)
        tsne_result = tsne_dimension_reduction(data_scaled_tsne, 3)
        tsne_df = pd.DataFrame(tsne_result, columns=['tSNE1', 'tSNE2', 'tSNE3'])
        tsne_df = pd.concat([original_data.reset_index(drop=True).iloc[:len(tsne_df)], tsne_df], axis=1)
        tsne_df.to_csv(output_path, index=False)
        print('t-SNE reduction is done! Check the "tsne_result.csv" file.')

if __name__ == "__main__":
    DR_main(input_path='input.csv', output_path='output.csv', type='PCA')
