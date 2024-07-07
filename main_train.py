from model import *
from tool import *

train_input_path = './data/BeijingHousingPrices_processed_modified.csv'
train_model_save_path = './result/model/lstm_model_average.h5'
train_scaler_followers_save_path = './result/model/scaler_followers_average.npy'
train_scaler_price_save_path = './result/model/scaler_price_average.npy'

train_main(train_input_path, train_model_save_path, train_scaler_followers_save_path, train_scaler_price_save_path)