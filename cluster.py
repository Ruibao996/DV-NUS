from tool import *
from model import *

DR_input_path = './data/BeijingHousingPrices_processed_modified.csv'
DR_output_path = './result/data/pca_result.csv'

DR_main(DR_input_path, DR_output_path)

KM_DR_input_path = './result/data/pca_result.csv'
KM_ori_input_path = './data/BeijingHousingPrices_processed.csv'
KM_cluster_centroids_path = './result/data/cluster_centroids.csv'
KM_pca_clustered_result_path = './result/data/pca_clustered_result.csv'
KM_pca_cluster_visualization_path = './result/picture/pca_cluster_visualization_4.png'

KM_main(KM_DR_input_path, KM_ori_input_path, KM_cluster_centroids_path, KM_pca_clustered_result_path, KM_pca_cluster_visualization_path)
