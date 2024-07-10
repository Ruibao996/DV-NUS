import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def perform_kmeans_clustering(data, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

def find_optimal_clusters(data, max_clusters=10):
    iters = range(2, max_clusters + 1, 1)
    
    sse = []
    silhouette_scores = []
    
    for k in iters:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(iters, sse, marker='o')
    ax1.set_xlabel('Cluster Number')
    ax1.set_ylabel('SSE')
    ax1.set_title('Elbow Method')
    
    ax2.plot(iters, silhouette_scores, marker='o')
    ax2.set_xlabel('Cluster Number')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Method')
    
    plt.show()

    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters

def KM_main(KM_DR_input_path, KM_ori_input_path, KM_cluster_centroids_path, KM_pca_clustered_result_path, KM_pca_cluster_visualization_path, type='PCA'):
    # Load the PCA result and original data
    dr_df = pd.read_csv(KM_DR_input_path)
    original_df = pd.read_csv(KM_ori_input_path)

    # Ensure the original dataframe has the id column as string to merge correctly
    original_df['id'] = original_df['id'].astype(str)
    dr_df['id'] = dr_df['id'].astype(str)

    if type == 'PCA':
        # Merge the PCA result with the original data to get the totalPrice column
        merged_df = pd.merge(dr_df, original_df[['id', 'totalPrice']], on='id')
    else:
        merged_df = original_df

    if type == 'PCA':
        data = merged_df[['PCA1', 'PCA2']]
    elif type == 'MSE':
        data = merged_df[['MSE1', 'MSE2']]
    else:
        data = merged_df[['tSNE1', 'tSNE2', 'tSNE3']]

    optimal_clusters = find_optimal_clusters(data)
    print(f"Optimal number of clusters: {optimal_clusters}")

    clusters, kmeans = perform_kmeans_clustering(data, optimal_clusters)

    merged_df['Cluster'] = clusters

    # Calculate the centroids of the clusters
    centroids = kmeans.cluster_centers_

    # Calculate num and Incoming for each cluster
    cluster_summary = merged_df.groupby('Cluster').agg(
        num=('Cluster', 'size'),
        Incoming=('totalPrice', 'sum'),
        # livingRoom=('livingRoom', 'mean'),
        # drawingRoom=('drawingRoom', 'mean'),
        # kitchen=('kitchen', 'mean'),
        # bathRoom=('bathRoom', 'mean'),
        floor_type=('floor_type', 'mean'),
        renovationCondition=('renovationCondition', 'mean'),
        elevator = ('elevator', 'mean'),
        subway = ('subway', 'mean'),
        floor_num=('floor_num', 'mean'),
        constructionTime=('constructionTime', 'mean'),
        ladderRatio=('ladderRatio', 'mean'),
        # district=('district', 'mean'),
        communityAverage=('communityAverage', 'mean'),
    ).reset_index()

    # Add the centroids to the cluster summary
    if type == 'PCA':
        centroids_df = pd.DataFrame(centroids, columns=['PCA1', 'PCA2'])
    elif type == 'MSE':
        centroids_df = pd.DataFrame(centroids, columns=['MSE1', 'MSE2'])
    else:
        centroids_df = pd.DataFrame(centroids, columns=['tSNE1', 'tSNE2', 'tSNE3'])

    cluster_summary = pd.concat([cluster_summary, centroids_df], axis=1)

    # Save the centroids to a CSV file
    cluster_summary.to_csv(KM_cluster_centroids_path, index=False)

    merged_df.to_csv(KM_pca_clustered_result_path, index=False)

    plt.figure(figsize=(12, 10))
    if type == 'PCA':
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Spectral', data=merged_df, legend='full')
        plt.title('PCA Cluster Visualization', fontsize=15)
        plt.xlabel('PCA1', fontsize=12)
        plt.ylabel('PCA2', fontsize=12)
    elif type == 'MSE':
        sns.scatterplot(x='MSE1', y='MSE2', hue='Cluster', palette='Spectral', data=merged_df, legend='full')
        plt.title('MSE Cluster Visualization', fontsize=15)
        plt.xlabel('MSE1', fontsize=12)
        plt.ylabel('MSE2', fontsize=12)
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(merged_df['tSNE1'], merged_df['tSNE2'], merged_df['tSNE3'], c=merged_df['Cluster'], cmap='viridis', s=60, alpha=0.8)
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        ax.set_xlabel('tSNE1', fontsize=12)
        ax.set_ylabel('tSNE2', fontsize=12)
        ax.set_zlabel('tSNE3', fontsize=12)
        ax.set_title('tSNE Cluster Visualization', fontsize=15)
    
    plt.savefig(KM_pca_cluster_visualization_path)
    plt.show()

    print("Clustering is done! Check the clustered result and visualization files.")

if __name__ == "__main__":
    KM_main(
        KM_DR_input_path='path_to_dr_input.csv',
        KM_ori_input_path='path_to_ori_input.csv',
        KM_cluster_centroids_path='path_to_centroids.csv',
        KM_pca_clustered_result_path='path_to_clustered_result.csv',
        KM_pca_cluster_visualization_path='path_to_cluster_visualization.png',
        type='PCA'  # or 'MSE' or 'tSNE' based on your requirement
    )
