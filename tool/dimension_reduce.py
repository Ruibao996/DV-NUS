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

# Random sample of the data 1/10
def random_sample(data, sample_size=0.05):
    # Calculate the number of samples to select
    num_samples = int(data.shape[0] * sample_size)
    
    # Generate random indices
    random_indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    
    # Use the indices to select rows
    sample = data[random_indices, :]
    
    return sample

def DR_main(input_path, output_path):
    df = pd.read_csv(input_path)

    df.replace('Unknown', np.nan, inplace=True)
    df.replace('Nan', np.nan, inplace=True)

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)

    categorical_columns = ['floor_type', 'renovationCondition', 'buildingStructure', 'elevator', 'fiveYearsProperty', 'subway']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    columns_of_interest = [
        'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'floor_num', 
        'constructionTime', 'ladderRatio', 'district', 'communityAverage'
    ]

    columns_of_interest.extend([col for col in df.columns if any(prefix in col for prefix in categorical_columns)])

    data = df[columns_of_interest]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    original_data = df[['id'] + columns_of_interest]

    # PCA
    pca_result = pca_dimension_reduction(data_scaled)
    pca_df = pd.DataFrame(pca_result, index=df.index, columns=['PCA1', 'PCA2'])
    pca_df = pd.concat([original_data.reset_index(drop=True), pca_df], axis=1)
    pca_df.to_csv(output_path, index=False)
    print('PCA reduction is done! Check the "pca_result.csv" file.')

    # MSE
    # data_scaled_mse = random_sample(data_scaled)
    # mse_result = mse_dimension_reduction(data_scaled_mse)
    # mse_df = pd.DataFrame(mse_result, index=df.index, columns=['MSE1', 'MSE2'])
    # mse_df = pd.concat([original_data.reset_index(drop=True), mse_df], axis=1)
    # mse_df.to_csv('./../result/data/mse_result.csv', index=False)
    # print('MSE reduction is done! Check the "mse_result.csv" file.')

    # t-SNE
    # tsne_result = tsne_dimension_reduction(data_scaled)
    # tsne_df = pd.DataFrame(tsne_result, index=df.index, columns=['tSNE1', 'tSNE2'])
    # tsne_df = pd.concat([original_data.reset_index(drop=True), tsne_df], axis=1)
    # tsne_df.to_csv('./../result/data/tsne_result.csv', index=False)
    # print('t-SNE reduction is done! Check the "tsne_result.csv" file.')


if __name__ == "__main__":
    DR_main()