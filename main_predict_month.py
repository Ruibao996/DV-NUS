from model import *
from tool import *

# prediction_model_path = './result/model/lstm_model_month.h5'
# prediction_followers_scaled_path = './result/model/scaler_followers_month.npy'
# prediction_price_scaled_path = './result/model/scaler_price_month.npy'
# prediction_total_price_scaled_path = './result/model/scaler_total_price_month.npy'
# prediction_square_scaled_path = './result/model/scaler_square_month.npy'
# predict_input_path = './data/BeijingHousingPrices_processed_modified.csv'
# monthly_price_save_path = './result/data/monthly_prices.csv'
# monthly_follower_save_path = './result/data/monthly_followers.csv'
# monthly_total_price_save_path = './result/data/monthly_total_prices.csv'
# monthly_square_save_path = './result/data/monthly_square.csv'
# monthly_square_save_path = './result/data/monthly_square.csv'
# price_predictions_save_path = './result/data/price_predictions_month.csv'
# total_price_predictions_save_path = './result/data/total_price_predictions_month.csv'
# square_predictions_save_path = './result/data/square_predictions_month.csv'
# future_price_predictions_save_path = './result/data/future_price_predictions_month.csv'
# future_total_price_predictions_save_path = './result/data/future_total_price_predictions_month.csv'
# future_square_predictions_save_path = './result/data/future_square_predictions_month.csv'
# price_predictions_picture_save_path = './result/picture/price_prediction_month.png'
# total_price_predictions_picture_save_path = './result/picture/total_price_prediction_month.png'
# square_predictions_picture_save_path = './result/picture/square_prediction_month.png'
# price_predictions_picture_2017_2019_save_path = './result/picture/price_prediction_month_2017_2019.png'
# total_price_predictions_picture_2017_2019_save_path = './result/picture/total_price_prediction_month_2017_2019.png'
# square_predictions_picture_2017_2019_save_path = './result/picture/square_prediction_month_2017_2019.png'

# predict_main_month(predict_input_path, 
#                    monthly_price_save_path, 
#                    monthly_follower_save_path, 
#                    monthly_total_price_save_path, 
#                    monthly_square_save_path,  
#                    price_predictions_save_path, 
#                    total_price_predictions_save_path, 
#                    square_predictions_save_path, 
#                    prediction_followers_scaled_path, 
#                    prediction_price_scaled_path, 
#                    prediction_total_price_scaled_path, 
#                    prediction_square_scaled_path, 
#                    prediction_square_scaled_path, 
#                    prediction_model_path, 
#                    future_price_predictions_save_path, 
#                    future_total_price_predictions_save_path, 
#                    future_square_predictions_save_path, 
#                    price_predictions_picture_save_path, 
#                    total_price_predictions_picture_save_path, 
#                    square_predictions_picture_save_path, 
#                    price_predictions_picture_2017_2019_save_path, 
#                    total_price_predictions_picture_2017_2019_save_path, 
#                    square_predictions_picture_2017_2019_save_path)


predict_main_month('./data/BeijingHousingPrices_processed_modified.csv', 
                       './result/data/monthly_prices.csv', 
                       './result/data/monthly_followers.csv', 
                       './result/data/monthly_total_prices.csv', 
                       './result/data/monthly_square.csv',  
                       './result/data/price_predictions_month.csv', 
                       './result/data/total_price_predictions_month.csv', 
                       './result/data/square_predictions_month.csv', 
                       './result/model/scaler_followers_month.npy', 
                       './result/model/scaler_price_month.npy', 
                       './result/model/scaler_total_price_month.npy', 
                       './result/model/scaler_square_month.npy', 
                       './result/model/lstm_model_month.h5', 
                       './result/data/future_price_predictions_month.csv', 
                       './result/data/future_total_price_predictions_month.csv', 
                       './result/data/future_square_predictions_month.csv',
                    #    './result/total_price_predictions_save_month.csv', 
                       './result/picture/price_prediction_month.png', 
                       './result/picture/total_price_prediction_month.png', 
                       './result/picture/square_prediction_month.png', 
                       './result/picture/price_prediction_month_2017_2019.png', 
                       './result/picture/total_price_prediction_month_2017_2019.png', 
                       './result/picture/square_prediction_month_2017_2019.png')