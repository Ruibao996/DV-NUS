from model import *
from tool import *


train_input_path = './data/BeijingHousingPrices_processed_modified.csv'
train_model_save_path = './result/model/lstm_model_month.h5'
train_scaler_followers_save_path = './result/model/scaler_followers_month.npy'
train_scaler_price_save_path = './result/model/scaler_price_month.npy'
train_scaler_total_price_save_path = './result/model/scaler_total_price_month.npy'
train_scaler_num_save_path = './result/model/scaler_num_month.npy'


train_main_month(train_input_path, train_model_save_path, train_scaler_followers_save_path, train_scaler_price_save_path, train_scaler_total_price_save_path, train_scaler_num_save_path)
