from tool import *
from model import *

DR_input_path = './data/BeijingHousingPrices_processed_modified.csv'
DR_output_path_PCA = './result/data/pca_result.csv'
DR_output_path_MSE = './result/data/MSE_result.csv'
DR_output_path_TSNE = './result/data/tsne_result.csv'

# DR_main(DR_input_path, DR_output_path_PCA)
# DR_main(DR_input_path, DR_output_path_MSE, type='MSE')
DR_main(DR_input_path, DR_output_path_TSNE, type='TSNE')

KM_DR_PCA_input_path = './result/data/pca_result.csv'
KM_DR_MSE_input_path = './result/data/MSE_result.csv'
KM_DR_TSNE_input_path = './result/data/tsne_result.csv'

KM_ori_input_path = './data/BeijingHousingPrices_processed.csv'
KM_mse_input_path = './result/data/MSE_result.csv'
KM_tsne_input_path = './result/data/tsne_result.csv'

KM_PCA_cluster_centroids_path = './result/data/pca_cluster_centroids.csv'
KM_MSE_cluster_centroids_path = './result/data/MSE_cluster_centroids.csv'
KM_TSNE_cluster_centroids_path = './result/data/tsne_cluster_centroids.csv'

KM_pca_clustered_result_path = './result/data/pca_clustered_result.csv'
KM_mse_clustered_result_path = './result/data/mse_clustered_result.csv'
KM_tsne_clustered_result_path = './result/data/tsne_clustered_result.csv'

KM_pca_cluster_visualization_path = './result/picture/pca_cluster_visualization_4.png'
KM_mse_cluster_visualization_path = './result/picture/mse_cluster_visualization_4.png'
KM_tsne_cluster_visualization_path = './result/picture/tsne_cluster_visualization_4.png'

# KM_main(KM_DR_PCA_input_path, KM_ori_input_path, KM_PCA_cluster_centroids_path, KM_pca_clustered_result_path, KM_pca_cluster_visualization_path)
# KM_main(KM_DR_MSE_input_path, KM_mse_input_path, KM_MSE_cluster_centroids_path, KM_mse_clustered_result_path, KM_mse_cluster_visualization_path,type='MSE')
KM_main(KM_DR_TSNE_input_path, KM_tsne_input_path, KM_TSNE_cluster_centroids_path, KM_tsne_clustered_result_path, KM_tsne_cluster_visualization_path,type='TSNE')
