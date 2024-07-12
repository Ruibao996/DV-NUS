import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step)]
        X.append(a)
    return np.array(X)

def predict_main_month(predict_input_path, monthly_price_save_path, monthly_follower_save_path, monthly_total_price_save_path, monthly_square_save_path, monthly_num_save_path,
                       price_predictions_save_path, total_price_predictions_save_path, num_predictions_save_path,
                       prediction_followers_scaled_path, prediction_price_scaled_path, prediction_total_price_scaled_path, prediction_square_scaled_path, prediction_num_scaled_path,
                       prediction_model_path, future_price_predictions_save_path, future_total_price_predictions_save_path, future_num_predictions_save_path,
                       price_predictions_picture_save_path, total_price_predictions_picture_save_path, num_predictions_picture_save_path, 
                       price_predictions_picture_2017_2019_save_path, total_price_predictions_picture_2017_2019_save_path, num_predictions_picture_2017_2019_save_path):
    # Load the data
    df = pd.read_csv(predict_input_path)

    # Convert tradeTime to datetime
    df['tradeTime'] = pd.to_datetime(df['tradeTime'], format='%m/%d/%Y')

    # Filter data from 2010 onwards
    df = df[df['tradeTime'] >= '2010-01-01']

    # Ensure the data is sorted by tradeTime
    df.sort_values(by='tradeTime', inplace=True)

    # Group by tradeTime and calculate the mean price, followers, totalPrice, square, and num for each month
    df['month'] = df['tradeTime'].dt.to_period('M')
    df_grouped = df.groupby('month').agg({'price': 'mean', 'followers': 'mean', 'totalPrice': 'mean', 'square': 'mean', 'num': 'mean'}).reset_index()

    # Convert the period to a datetime object
    df_grouped['month'] = df_grouped['month'].dt.to_timestamp()

    # Preprocess the data
    df_grouped['followers'] = pd.to_numeric(df_grouped['followers'], errors='coerce')
    df_grouped['price'] = pd.to_numeric(df_grouped['price'], errors='coerce')
    df_grouped['totalPrice'] = pd.to_numeric(df_grouped['totalPrice'], errors='coerce')
    df_grouped['square'] = pd.to_numeric(df_grouped['square'], errors='coerce')
    df_grouped['num'] = pd.to_numeric(df_grouped['num'], errors='coerce')

    df_grouped['followers'].fillna(df_grouped['followers'].mean(), inplace=True)
    df_grouped['price'].fillna(df_grouped['price'].mean(), inplace=True)
    df_grouped['totalPrice'].fillna(df_grouped['totalPrice'].mean(), inplace=True)
    df_grouped['square'].fillna(df_grouped['square'].mean(), inplace=True)
    df_grouped['num'].fillna(df_grouped['num'].mean(), inplace=True)

    # Save the monthly data to CSV
    df_grouped.to_csv(monthly_price_save_path, index=False)

    # Load the scalers
    scaler_followers = np.load(prediction_followers_scaled_path, allow_pickle=True).item()
    scaler_price = np.load(prediction_price_scaled_path, allow_pickle=True).item()
    scaler_total_price = np.load(prediction_total_price_scaled_path, allow_pickle=True).item()
    scaler_square = np.load(prediction_square_scaled_path, allow_pickle=True).item()
    scaler_num = np.load(prediction_num_scaled_path, allow_pickle=True).item()

    followers_scaled = scaler_followers.transform(np.array(df_grouped['followers']).reshape(-1, 1))
    price_scaled = scaler_price.transform(np.array(df_grouped['price']).reshape(-1, 1))
    total_price_scaled = scaler_total_price.transform(np.array(df_grouped['totalPrice']).reshape(-1, 1))
    square_scaled = scaler_square.transform(np.array(df_grouped['square']).reshape(-1, 1))
    num_scaled = scaler_num.transform(np.array(df_grouped['num']).reshape(-1, 1))

    # Create dataset for prediction
    time_step = 12  # One year of monthly data
    X_followers = create_dataset(followers_scaled, time_step)
    X_price = create_dataset(price_scaled, time_step)
    X_total_price = create_dataset(total_price_scaled, time_step)
    X_square = create_dataset(square_scaled, time_step)
    X_num = create_dataset(num_scaled, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_followers = X_followers.reshape(X_followers.shape[0], X_followers.shape[1], 1)
    X_price = X_price.reshape(X_price.shape[0], X_price.shape[1], 1)
    X_total_price = X_total_price.reshape(X_total_price.shape[0], X_total_price.shape[1], 1)
    X_square = X_square.reshape(X_square.shape[0], X_square.shape[1], 1)
    X_num = X_num.reshape(X_num.shape[0], X_num.shape[1], 1)

    # Load the model
    model = load_model(prediction_model_path)

    # Make initial predictions
    predictions = model.predict([X_followers, X_price, X_total_price, X_square, X_num])
    price_predictions = scaler_price.inverse_transform(predictions[0])
    total_price_predictions = scaler_total_price.inverse_transform(predictions[1])
    num_predictions = scaler_num.inverse_transform(predictions[2])

    # Prepare to make future predictions for 12 months
    future_price_predictions = []
    future_total_price_predictions = []
    future_num_predictions = []

    last_followers_data = followers_scaled[-time_step:]
    last_price_data = price_scaled[-time_step:]
    last_total_price_data = total_price_scaled[-time_step:]
    last_square_data = square_scaled[-time_step:]
    last_num_data = num_scaled[-time_step:]

    for _ in range(12):
        next_followers = model.predict([last_followers_data.reshape(1, time_step, 1), last_price_data.reshape(1, time_step, 1), last_total_price_data.reshape(1, time_step, 1), last_square_data.reshape(1, time_step, 1), last_num_data.reshape(1, time_step, 1)])
        next_price = model.predict([last_followers_data.reshape(1, time_step, 1), last_price_data.reshape(1, time_step, 1), last_total_price_data.reshape(1, time_step, 1), last_square_data.reshape(1, time_step, 1), last_num_data.reshape(1, time_step, 1)])

        # Append the next prediction
        future_price_predictions.append(next_price[0, 0])
        future_total_price_predictions.append(next_price[1, 0])
        future_num_predictions.append(next_price[2, 0])

        # Update the data for the next step
        last_followers_data = np.append(last_followers_data[1:], next_followers[0]).reshape(time_step, 1)
        last_price_data = np.append(last_price_data[1:], next_price[0]).reshape(time_step, 1)
        last_total_price_data = np.append(last_total_price_data[1:], next_price[1]).reshape(time_step, 1)
        last_square_data = np.append(last_square_data[1:], next_price[2]).reshape(time_step, 1)
        last_num_data = np.append(last_num_data[1:], next_price[3]).reshape(time_step, 1)

    # Inverse transform the future predictions
    future_price_predictions = scaler_price.inverse_transform(np.array(future_price_predictions).reshape(-1, 1))
    future_total_price_predictions = scaler_total_price.inverse_transform(np.array(future_total_price_predictions).reshape(-1, 1))
    future_num_predictions = scaler_num.inverse_transform(np.array(future_num_predictions).reshape(-1, 1))

    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame({'Predicted_Price': price_predictions.flatten(), 'Predicted_Total_Price': total_price_predictions.flatten(), 'Predicted_Num': num_predictions.flatten()})
    predictions_df['tradeTime'] = df_grouped['month'][time_step:].reset_index(drop=True)

    # Create a DataFrame for the future predictions
    last_date = df_grouped['month'].iloc[-1]
    future_dates = pd.date_range(last_date, periods=12 + 1, freq='M')[1:]
    future_predictions_df = pd.DataFrame({'Predicted_Price': future_price_predictions.flatten(), 'Predicted_Total_Price': future_total_price_predictions.flatten(), 'Predicted_Num': future_num_predictions.flatten()})
    future_predictions_df['tradeTime'] = future_dates

    # Save the predictions
    predictions_df.to_csv(price_predictions_save_path, index=False)
    future_predictions_df.to_csv(future_price_predictions_save_path, index=False)

    # Visualize the price prediction
    plt.figure(figsize=(16, 8))
    plt.title("Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    ax = plt.gca()  
    ax.set_facecolor('whitesmoke') 
    plt.plot(df_grouped['month'], df_grouped['price'], label='Original Price', linewidth=2)
    plt.plot(predictions_df['tradeTime'], predictions_df['Predicted_Price'], label='Predicted Price', alpha=0.7, linewidth=2)
    plt.plot(future_predictions_df['tradeTime'], future_predictions_df['Predicted_Price'], label='Future Predicted Price', alpha=0.7, linewidth=2, color='red')
    plt.legend(loc="lower right")
    plt.savefig(price_predictions_picture_save_path)
    plt.show()

    # Visualize the total price prediction
    plt.figure(figsize=(16, 8))
    plt.title("Total Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Total Price")
    ax = plt.gca()  
    ax.set_facecolor('whitesmoke') 
    plt.plot(df_grouped['month'], df_grouped['totalPrice'], label='Original Total Price', linewidth=2)
    plt.plot(predictions_df['tradeTime'], predictions_df['Predicted_Total_Price'], label='Predicted Total Price', alpha=0.7, linewidth=2)
    plt.plot(future_predictions_df['tradeTime'], future_predictions_df['Predicted_Total_Price'], label='Future Predicted Total Price', alpha=0.7, linewidth=2, color='red')
    plt.legend(loc="lower right")
    plt.savefig(total_price_predictions_picture_save_path)
    plt.show()

    # Visualize the num prediction
    plt.figure(figsize=(16, 8))
    plt.title("Num Prediction")
    plt.xlabel("Time")
    plt.ylabel("Num")
    ax = plt.gca()  
    ax.set_facecolor('whitesmoke') 
    plt.plot(df_grouped['month'], df_grouped['num'], label='Original Num', linewidth=2)
    plt.plot(predictions_df['tradeTime'], predictions_df['Predicted_Num'], label='Predicted Num', alpha=0.7, linewidth=2)
    plt.plot(future_predictions_df['tradeTime'], future_predictions_df['Predicted_Num'], label='Future Predicted Num', alpha=0.7, linewidth=2, color='red')
    plt.legend(loc="lower right")
    plt.savefig(num_predictions_picture_save_path)
    plt.show()

    # Visualize the price prediction for 2017-2019
    plt.figure(figsize=(16, 8))
    plt.title("Price Prediction (2017-2019)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    ax = plt.gca()  
    ax.set_facecolor('whitesmoke') 
    plt.plot(df_grouped['month'], df_grouped['price'], label='Original Price', linewidth=4)
    plt.plot(predictions_df['tradeTime'], predictions_df['Predicted_Price'], label='Predicted Price', alpha=0.7, linewidth=4)
    plt.plot(future_predictions_df['tradeTime'], future_predictions_df['Predicted_Price'], label='Future Predicted Price', alpha=0.7, linewidth=3, color='red', linestyle='dashed')
    plt.xlim(pd.to_datetime("2017-01-01"), pd.to_datetime("2019-01-01"))
    plt.ylim(55000, 75000)
    plt.legend(loc="lower right")
    plt.savefig(price_predictions_picture_2017_2019_save_path)
    plt.show()

    # Visualize the total price prediction for 2017-2019
    plt.figure(figsize=(16, 8))
    plt.title("Total Price Prediction (2017-2019)")
    plt.xlabel("Time")
    plt.ylabel("Total Price")
    ax = plt.gca()  
    ax.set_facecolor('whitesmoke') 
    plt.plot(df_grouped['month'], df_grouped['totalPrice'], label='Original Total Price', linewidth=4)
    plt.plot(predictions_df['tradeTime'], predictions_df['Predicted_Total_Price'], label='Predicted Total Price', alpha=0.7, linewidth=4)
    plt.plot(future_predictions_df['tradeTime'], future_predictions_df['Predicted_Total_Price'], label='Future Predicted Total Price', alpha=0.7, linewidth=3, color='red', linestyle='dashed')
    plt.xlim(pd.to_datetime("2017-01-01"), pd.to_datetime("2019-01-01"))
    plt.ylim(55000, 75000)
    plt.legend(loc="lower right")
    plt.savefig(total_price_predictions_picture_2017_2019_save_path)
    plt.show()

    # Visualize the num prediction for 2017-2019
    plt.figure(figsize=(16, 8))
    plt.title("Num Prediction (2017-2019)")
    plt.xlabel("Time")
    plt.ylabel("Num")
    ax = plt.gca()  
    ax.set_facecolor('whitesmoke') 
    plt.plot(df_grouped['month'], df_grouped['num'], label='Original Num', linewidth=4)
    plt.plot(predictions_df['tradeTime'], predictions_df['Predicted_Num'], label='Predicted Num', alpha=0.7, linewidth=4)
    plt.plot(future_predictions_df['tradeTime'], future_predictions_df['Predicted_Num'], label='Future Predicted Num', alpha=0.7, linewidth=3, color='red', linestyle='dashed')
    plt.xlim(pd.to_datetime("2017-01-01"), pd.to_datetime("2019-01-01"))
    plt.ylim(55000, 75000)
    plt.legend(loc="lower right")
    plt.savefig(num_predictions_picture_2017_2019_save_path)
    plt.show()

if __name__ == "__main__":
    predict_main_month(predict_input_path='predict_input.csv',
                       monthly_price_save_path='monthly_price.csv',
                       monthly_follower_save_path='monthly_follower.csv',
                       monthly_total_price_save_path='monthly_total_price.csv',
                       monthly_square_save_path='monthly_square.csv',
                       monthly_num_save_path='monthly_num.csv',
                       price_predictions_save_path='monthly_price_predictions.csv',
                       total_price_predictions_save_path='monthly_total_price_predictions.csv',
                       num_predictions_save_path='monthly_num_predictions.csv',
                       prediction_followers_scaled_path='followers_scaler_month.npy',
                       prediction_price_scaled_path='price_scaler_month.npy',
                       prediction_total_price_scaled_path='total_price_scaler_month.npy',
                       prediction_square_scaled_path='square_scaler_month.npy',
                       prediction_num_scaled_path='num_scaler_month.npy',
                       prediction_model_path='model_month.h5',
                       future_price_predictions_save_path='future_monthly_price_predictions.csv',
                       future_total_price_predictions_save_path='future_monthly_total_price_predictions.csv',
                       future_num_predictions_save_path='future_monthly_num_predictions.csv',
                       price_predictions_picture_save_path='monthly_price_predictions.png',
                       total_price_predictions_picture_save_path='monthly_total_price_predictions.png',
                       num_predictions_picture_save_path='monthly_num_predictions.png',
                       price_predictions_picture_2017_2019_save_path='monthly_price_predictions_2017_2019.png',
                       total_price_predictions_picture_2017_2019_save_path='monthly_total_price_predictions_2017_2019.png',
                       num_predictions_picture_2017_2019_save_path='monthly_num_predictions_2017_2019.png')
