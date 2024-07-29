import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib

def create_dataset(data, time_step=1):
    X = []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step)]
        X.append(a)
    return np.array(X)

def predict_main_month(predict_input_path, monthly_price_save_path, monthly_follower_save_path, monthly_total_price_save_path, monthly_square_save_path, 
                       price_predictions_save_path, total_price_predictions_save_path, square_predictions_save_path, 
                       scaler_followers_path, scaler_price_path, scaler_total_price_path, scaler_square_path, model_path, 
                       future_price_predictions_save_path, future_total_price_predictions_save_path, future_square_predictions_save_path, 
                       price_predictions_picture_save_path, total_price_predictions_picture_save_path, square_predictions_picture_save_path, 
                       price_predictions_picture_2017_2019_save_path, total_price_predictions_picture_2017_2019_save_path, square_predictions_picture_2017_2019_save_path):
    # Load the data
    df = pd.read_csv(predict_input_path)

    # Convert tradeTime to datetime
    df['tradeTime'] = pd.to_datetime(df['tradeTime'], format='%m/%d/%Y')

    # Filter data from 2010 onwards
    df = df[df['tradeTime'] >= '2010-01-01']

    # Ensure the data is sorted by tradeTime
    df.sort_values(by='tradeTime', inplace=True)

    # Group by tradeTime and calculate the mean price, followers, totalPrice, and square for each month
    df['month'] = df['tradeTime'].dt.to_period('M')
    df_grouped = df.groupby('month').agg({'price': 'mean', 'followers': 'mean', 'totalPrice': 'mean', 'square': 'mean'}).reset_index()

    # Convert the period to a datetime object
    df_grouped['month'] = df_grouped['month'].dt.to_timestamp()

    # Preprocess the data
    df_grouped['followers'] = pd.to_numeric(df_grouped['followers'], errors='coerce')
    df_grouped['price'] = pd.to_numeric(df_grouped['price'], errors='coerce')
    df_grouped['totalPrice'] = pd.to_numeric(df_grouped['totalPrice'], errors='coerce')
    df_grouped['square'] = pd.to_numeric(df_grouped['square'], errors='coerce')

    df_grouped['followers'].fillna(df_grouped['followers'].mean(), inplace=True)
    df_grouped['price'].fillna(df_grouped['price'].mean(), inplace=True)
    df_grouped['totalPrice'].fillna(df_grouped['totalPrice'].mean(), inplace=True)
    df_grouped['square'].fillna(df_grouped['square'].mean(), inplace=True)

    # Save preprocessed data to CSV files
    df_grouped.to_csv(monthly_price_save_path, columns=['month', 'price'], index=False)
    df_grouped.to_csv(monthly_follower_save_path, columns=['month', 'followers'], index=False)
    df_grouped.to_csv(monthly_total_price_save_path, columns=['month', 'totalPrice'], index=False)
    df_grouped.to_csv(monthly_square_save_path, columns=['month', 'square'], index=False)

    # Load the scalers
    scaler_followers = joblib.load(scaler_followers_path)
    scaler_price = joblib.load(scaler_price_path)
    scaler_total_price = joblib.load(scaler_total_price_path)
    scaler_square = joblib.load(scaler_square_path)

    # Scale the data
    followers_scaled = scaler_followers.transform(np.array(df_grouped['followers']).reshape(-1, 1))
    price_scaled = scaler_price.transform(np.array(df_grouped['price']).reshape(-1, 1))
    total_price_scaled = scaler_total_price.transform(np.array(df_grouped['totalPrice']).reshape(-1, 1))
    square_scaled = scaler_square.transform(np.array(df_grouped['square']).reshape(-1, 1))

    # Create dataset for LSTM
    time_step = 12  # One year of monthly data
    X_followers = create_dataset(followers_scaled, time_step)
    X_price = create_dataset(price_scaled, time_step)
    X_total_price = create_dataset(total_price_scaled, time_step)
    X_square = create_dataset(square_scaled, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_followers = X_followers.reshape(X_followers.shape[0], X_followers.shape[1], 1)
    X_price = X_price.reshape(X_price.shape[0], X_price.shape[1], 1)
    X_total_price = X_total_price.reshape(X_total_price.shape[0], X_total_price.shape[1], 1)
    X_square = X_square.reshape(X_square.shape[0], X_square.shape[1], 1)

    # Load the model
    model = load_model(model_path)

    # Make initial predictions
    predictions_price = model.predict([X_followers, X_price, X_total_price, X_square])
    predictions_total_price = model.predict([X_followers, X_price, X_total_price, X_square])  # Initialize predictions for total price
    predictions_square = model.predict([X_followers, X_price, X_total_price, X_square])      # Initialize predictions for square

    # Inverse transform the predictions to get actual values
    predictions_price = scaler_price.inverse_transform(predictions_price)
    predictions_total_price = scaler_total_price.inverse_transform(predictions_total_price)
    predictions_square = scaler_square.inverse_transform(predictions_square)

    # Save initial predictions to CSV files
    df_predictions_price = pd.DataFrame(predictions_price, columns=['PredictedPrice'])
    df_predictions_price['month'] = df_grouped['month'][time_step:].values
    df_predictions_price.to_csv(price_predictions_save_path, index=False)

    df_predictions_total_price = pd.DataFrame(predictions_total_price, columns=['PredictedTotalPrice'])
    df_predictions_total_price['month'] = df_grouped['month'][time_step:].values
    df_predictions_total_price.to_csv(total_price_predictions_save_path, index=False)

    df_predictions_square = pd.DataFrame(predictions_square, columns=['PredictedSquare'])
    df_predictions_square['month'] = df_grouped['month'][time_step:].values
    df_predictions_square.to_csv(square_predictions_save_path, index=False)


    # Prepare to make future predictions for 12 months
    future_price_predictions = []
    future_total_price_predictions = []
    future_square_predictions = []

    last_followers_data = followers_scaled[-time_step:]
    last_price_data = price_scaled[-time_step:]
    last_total_price_data = total_price_scaled[-time_step:]
    last_square_data = square_scaled[-time_step:]

    for _ in range(12):
        next_followers = model.predict([last_followers_data.reshape(1, time_step, 1), last_price_data.reshape(1, time_step, 1), last_total_price_data.reshape(1, time_step, 1), last_square_data.reshape(1, time_step, 1)])
        next_price = model.predict([last_followers_data.reshape(1, time_step, 1), last_price_data.reshape(1, time_step, 1), last_total_price_data.reshape(1, time_step, 1), last_square_data.reshape(1, time_step, 1)])
        next_total_price = model.predict([last_followers_data.reshape(1, time_step, 1), last_price_data.reshape(1, time_step, 1), last_total_price_data.reshape(1, time_step, 1), last_square_data.reshape(1, time_step, 1)])
        next_square = model.predict([last_followers_data.reshape(1, time_step, 1), last_price_data.reshape(1, time_step, 1), last_total_price_data.reshape(1, time_step, 1), last_square_data.reshape(1, time_step, 1)])

        # Append the next prediction
        future_price_predictions.append(next_price[0, 0])
        future_total_price_predictions.append(next_total_price[0, 0])
        future_square_predictions.append(next_square[0, 0])

        # Update the data for the next step
        last_followers_data = np.append(last_followers_data[1:], next_followers[0]).reshape(time_step, 1)
        last_price_data = np.append(last_price_data[1:], next_price[0]).reshape(time_step, 1)
        last_total_price_data = np.append(last_total_price_data[1:], next_total_price[0]).reshape(time_step, 1)
        last_square_data = np.append(last_square_data[1:], next_square[0]).reshape(time_step, 1)

    # Inverse transform the future predictions
    future_price_predictions = scaler_price.inverse_transform(np.array(future_price_predictions).reshape(-1, 1))
    future_total_price_predictions = scaler_total_price.inverse_transform(np.array(future_total_price_predictions).reshape(-1, 1))
    future_square_predictions = scaler_square.inverse_transform(np.array(future_square_predictions).reshape(-1, 1))

    # Create DataFrames for the future predictions
    last_date = df_grouped['month'].iloc[-1]
    future_dates = pd.date_range(last_date, periods=12 + 1, freq='M')[1:]
    future_price_predictions_df = pd.DataFrame(future_price_predictions, columns=['PredictedPrice'])
    future_total_price_predictions_df = pd.DataFrame(future_total_price_predictions, columns=['PredictedTotalPrice'])
    future_square_predictions_df = pd.DataFrame(future_square_predictions, columns=['PredictedSquare'])

    future_price_predictions_df['month'] = future_dates
    future_total_price_predictions_df['month'] = future_dates
    future_square_predictions_df['month'] = future_dates

    # Save the future predictions
    future_price_predictions_df.to_csv(future_price_predictions_save_path, index=False)
    future_total_price_predictions_df.to_csv(future_total_price_predictions_save_path, index=False)
    future_square_predictions_df.to_csv(future_square_predictions_save_path, index=False)


    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['month'], df_grouped['price'], label='Actual Price')
    plt.plot(df_predictions_price['month'], df_predictions_price['PredictedPrice'], label='Predicted Price')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Price')
    plt.legend()
    plt.savefig(price_predictions_picture_save_path)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['month'], df_grouped['totalPrice'], label='Actual Total Price')
    plt.plot(future_total_price_predictions_df['month'], future_total_price_predictions_df['PredictedTotalPrice'], label='Predicted Total Price')
    plt.xlabel('Month')
    plt.ylabel('Total Price')
    plt.title('Actual vs Predicted Total Price')
    plt.legend()
    plt.savefig(total_price_predictions_picture_save_path)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['month'], df_grouped['square'], label='Actual Square')
    plt.plot(future_square_predictions_df['month'], future_square_predictions_df['PredictedSquare'], label='Predicted Square')
    plt.xlabel('Month')
    plt.ylabel('Square')
    plt.title('Actual vs Predicted Square')
    plt.legend()
    plt.savefig(square_predictions_picture_save_path)
    plt.show()

    # Plot the predictions for 2017-2019
    df_predictions_price_2017_2019 = df_predictions_price[(df_predictions_price['month'] >= '2017-01-01') & (df_predictions_price['month'] <= '2019-12-31')]
    df_predictions_total_price_2017_2019 = future_total_price_predictions_df[(future_total_price_predictions_df['month'] >= '2017-01-01') & (future_total_price_predictions_df['month'] <= '2019-12-31')]
    df_predictions_square_2017_2019 = future_square_predictions_df[(future_square_predictions_df['month'] >= '2017-01-01') & (future_square_predictions_df['month'] <= '2019-12-31')]

    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['month'], df_grouped['price'], label='Actual Price')
    plt.plot(df_predictions_price_2017_2019['month'], df_predictions_price_2017_2019['PredictedPrice'], label='Predicted Price (2017-2019)')
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Price (2017-2019)')
    plt.legend()
    plt.savefig(price_predictions_picture_2017_2019_save_path)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['month'], df_grouped['totalPrice'], label='Actual Total Price')
    plt.plot(df_predictions_total_price_2017_2019['month'], df_predictions_total_price_2017_2019['PredictedTotalPrice'], label='Predicted Total Price (2017-2019)')
    plt.xlabel('Month')
    plt.ylabel('Total Price')
    plt.title('Actual vs Predicted Total Price (2017-2019)')
    plt.legend()
    plt.savefig(total_price_predictions_picture_2017_2019_save_path)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped['month'], df_grouped['square'], label='Actual Square')
    plt.plot(df_predictions_square_2017_2019['month'], df_predictions_square_2017_2019['PredictedSquare'], label='Predicted Square (2017-2019)')
    plt.xlabel('Month')
    plt.ylabel('Square')
    plt.title('Actual vs Predicted Square (2017-2019)')
    plt.legend()
    plt.savefig(square_predictions_picture_2017_2019_save_path)
    plt.show()

if __name__ == "__main__":
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
                       './result/total_price_predictions_save_month.csv', 
                       './result/picture/price_prediction_month.png', 
                       './result/picture/total_price_prediction_month.png', 
                       './result/picture/square_prediction_month.png', 
                       './result/picture/price_prediction_month_2017_2019.png', 
                       './result/picture/total_price_prediction_month_2017_2019.png', 
                       './result/picture/square_prediction_month_2017_2019.png')
