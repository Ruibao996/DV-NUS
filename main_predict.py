from model import *
from tool import *

prediction_model_path = './result/model/lstm_model_average.h5'
prediction_followers_scaled_path = './result/model/scaler_followers_average.npy'
prediction_price_scaled_path = './result/model/scaler_price_average.npy'
predict_input_path = './data/BeijingHousingPrices_processed_modified.csv'
predict_daily_prices_save_path = './result/data/daily_prices.csv'
predict_follower_save_path = './result/data/daily_followers.csv'
price_predictions_save_path = './result/data/price_predictions_average.csv'
future_price_predictions_save_path = './result/data/future_price_predictions.csv'
price_predictions_picture_save_path = './result/picture/price_prediction_average.png'

predict_main(predict_input_path, predict_daily_prices_save_path, predict_follower_save_path, price_predictions_save_path, prediction_followers_scaled_path, prediction_price_scaled_path, prediction_model_path, future_price_predictions_save_path, price_predictions_picture_save_path)