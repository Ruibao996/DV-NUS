from model import *
from tool import *

prediction_model_path = './result/model/lstm_model_month.h5'
prediction_followers_scaled_path = './result/model/scaler_followers_month.npy'
prediction_price_scaled_path = './result/model/scaler_price_month.npy'
prediction_total_price_scaled_path = './result/model/scaler_total_price_month.npy'
prediction_num_scaled_path = './result/model/scaler_num_month.npy'
predict_input_path = './data/BeijingHousingPrices_processed_modified.csv'
monthly_price_save_path = './result/data/monthly_prices.csv'
monthly_follower_save_path = './result/data/monthly_followers.csv'
monthly_total_price_save_path = './result/data/monthly_total_prices.csv'
monthly_square_save_path = './result/data/monthly_square.csv'
monthly_num_save_path = './result/data/monthly_num.csv'
price_predictions_save_path = './result/data/price_predictions_month.csv'
total_price_predictions_save_path = './result/data/total_price_predictions_month.csv'
num_predictions_save_path = './result/data/num_predictions_month.csv'
future_price_predictions_save_path = './result/data/future_price_predictions_month.csv'
future_total_price_predictions_save_path = './result/data/future_total_price_predictions_month.csv'
future_num_predictions_save_path = './result/data/future_num_predictions_month.csv'
price_predictions_picture_save_path = './result/picture/price_prediction_month.png'
total_price_predictions_picture_save_path = './result/picture/total_price_prediction_month.png'
num_predictions_picture_save_path = './result/picture/num_prediction_month.png'
price_predictions_picture_2017_2019_save_path = './result/picture/price_prediction_month_2017_2019.png'
total_price_predictions_picture_2017_2019_save_path = './result/picture/total_price_prediction_month_2017_2019.png'
num_predictions_picture_2017_2019_save_path = './result/picture/num_prediction_month_2017_2019.png'

predict_main_month(predict_input_path, monthly_price_save_path, monthly_follower_save_path, monthly_total_price_save_path, monthly_square_save_path, monthly_num_save_path, 
                   price_predictions_save_path, total_price_predictions_save_path, num_predictions_save_path, 
                   prediction_followers_scaled_path, prediction_price_scaled_path, prediction_total_price_scaled_path, prediction_num_scaled_path, prediction_model_path, 
                   future_price_predictions_save_path, future_total_price_predictions_save_path, future_num_predictions_save_path,
                   price_predictions_picture_save_path, total_price_predictions_picture_save_path, num_predictions_picture_save_path, 
                   price_predictions_picture_2017_2019_save_path, total_price_predictions_picture_2017_2019_save_path, num_predictions_picture_2017_2019_save_path)
