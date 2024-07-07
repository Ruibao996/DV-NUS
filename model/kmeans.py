import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def perform_kmeans_clustering(data, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def KM_main(KM_DR_input_path, KM_ori_input_path, KM_cluster_centroids_path, KM_pca_clustered_result_path, KM_pca_cluster_visualization_path):
    # Load the PCA result and original data
    pca_df = pd.read_csv(KM_DR_input_path)
    original_df = pd.read_csv(KM_ori_input_path)
    
    # Ensure the original dataframe has the id column as string to merge correctly
    original_df['id'] = original_df['id'].astype(str)
    pca_df['id'] = pca_df['id'].astype(str)

    # Merge the PCA result with the original data to get the total_price column
    merged_df = pd.merge(pca_df, original_df[['id', 'totalPrice']], on='id')

    pca_data = merged_df[['PCA1', 'PCA2']]

    num_clusters = 4
    clusters, kmeans = perform_kmeans_clustering(pca_data, num_clusters)

    merged_df['Cluster'] = clusters

    # Calculate the centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Calculate num and Incoming for each cluster
    cluster_summary = merged_df.groupby('Cluster').agg(
        num=('Cluster', 'size'),
        Incoming=('totalPrice', 'sum'),
        livingRoom=('livingRoom', 'mean'),
        drawingRoom=('drawingRoom', 'mean'),
        kitchen=('kitchen', 'mean'),
        bathRoom=('bathRoom', 'mean'),
        floor_num=('floor_num', 'mean'),
        constructionTime=('constructionTime', 'mean'),
        ladderRatio=('ladderRatio', 'mean'),
        district=('district', 'mean'),
        communityAverage=('communityAverage', 'mean'),
    ).reset_index()

    # Add the centroids to the cluster summary
    centroids_df = pd.DataFrame(centroids, columns=['PCA1', 'PCA2'])
    cluster_summary = pd.concat([cluster_summary, centroids_df], axis=1)

    # Save the centroids to a CSV file
    cluster_summary.to_csv(KM_cluster_centroids_path, index=False)

    merged_df.to_csv(KM_pca_clustered_result_path, index=False)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='viridis', data=merged_df, legend='full')
    plt.title('PCA Cluster Visualization')
    plt.savefig(KM_pca_cluster_visualization_path)
    plt.show()

    print("Clustering is done! Check the 'pca_clustered_result.csv', 'cluster_centroids.csv', and 'pca_cluster_visualization_4.png' files.")

if __name__ == "__main__":
    KM_main()
