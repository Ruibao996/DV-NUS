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

def predict_main(predict_input_path, daily_price_save_path, price_predictions_save_path, prediction_followers_scaled_path, prediction_price_scaled_path, prediction_model_path, future_price_predictions_save_path, price_predictions_picture_save_path):
    # Load the data
    df = pd.read_csv(predict_input_path)

    # Convert tradeTime to datetime
    df['tradeTime'] = pd.to_datetime(df['tradeTime'], format='%m/%d/%Y')

    # Filter data from 2010 onwards
    df = df[df['tradeTime'] >= '2010-01-01']

    # Ensure the data is sorted by tradeTime
    df.sort_values(by='tradeTime', inplace=True)

    # Group by tradeTime and calculate the mean price and followers
    df_grouped = df.groupby('tradeTime').agg({'price': 'mean', 'followers': 'mean'}).reset_index()

    # Preprocess the data
    df_grouped['followers'] = pd.to_numeric(df_grouped['followers'], errors='coerce')
    df_grouped['price'] = pd.to_numeric(df_grouped['price'], errors='coerce')

    df_grouped['followers'].fillna(df_grouped['followers'].mean(), inplace=True)
    df_grouped['price'].fillna(df_grouped['price'].mean(), inplace=True)

    # Calculate rolling mean to smooth the data
    window_size = 7  # 7-day rolling average
    df_grouped['price_rolling'] = df_grouped['price'].rolling(window=window_size).mean()
    df_grouped['followers_rolling'] = df_grouped['followers'].rolling(window=window_size).mean()

    # Drop NaN values created by rolling mean
    df_grouped.dropna(inplace=True)

    # Save the daily price_mean and price_rolling to CSV
    df_daily_prices = df_grouped[['tradeTime', 'price', 'price_rolling']].copy()
    df_daily_prices.set_index('tradeTime', inplace=True)
    df_daily_prices.to_csv(daily_price_save_path)

    # Load the scalers
    scaler_followers = np.load(prediction_followers_scaled_path, allow_pickle=True).item()
    scaler_price = np.load(prediction_price_scaled_path, allow_pickle=True).item()

    followers_scaled = scaler_followers.transform(np.array(df_grouped['followers_rolling']).reshape(-1, 1))
    price_scaled = scaler_price.transform(np.array(df_grouped['price_rolling']).reshape(-1, 1))

    # Create dataset for prediction
    time_step = 365  # One year of daily data
    X_followers = create_dataset(followers_scaled, time_step)
    X_price = create_dataset(price_scaled, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_followers = X_followers.reshape(X_followers.shape[0], X_followers.shape[1], 1)
    X_price = X_price.reshape(X_price.shape[0], X_price.shape[1], 1)

    # Load the model
    model = load_model(prediction_model_path)

    # Make initial predictions
    predictions = model.predict([X_followers, X_price])
    predictions = scaler_price.inverse_transform(predictions)

    # Prepare to make future predictions for 365 days
    future_predictions = []
    last_followers_data = followers_scaled[-time_step:]
    last_price_data = price_scaled[-time_step:]

    for _ in range(365):
        next_followers = model.predict([last_followers_data.reshape(1, time_step, 1), last_price_data.reshape(1, time_step, 1)])
        next_price = model.predict([last_followers_data.reshape(1, time_step, 1), last_price_data.reshape(1, time_step, 1)])

        # Append the next prediction
        future_predictions.append(next_price[0, 0])

        # Update the data for the next step
        last_followers_data = np.append(last_followers_data[1:], next_followers[0]).reshape(time_step, 1)
        last_price_data = np.append(last_price_data[1:], next_price[0]).reshape(time_step, 1)

    # Inverse transform the future predictions
    future_predictions = scaler_price.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Create a DataFrame for the predictions
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Price'])
    predictions_df['tradeTime'] = df_grouped['tradeTime'][time_step:].reset_index(drop=True)

    # Create a DataFrame for the future predictions
    last_date = df_grouped['tradeTime'].iloc[-1]
    future_dates = pd.date_range(last_date, periods=365 + 1)[1:]
    future_predictions_df = pd.DataFrame(future_predictions, columns=['Predicted_Price'])
    future_predictions_df['tradeTime'] = future_dates

    # Save the predictions
    predictions_df.to_csv(price_predictions_save_path, index=False)
    future_predictions_df.to_csv(future_price_predictions_save_path, index=False)

    # Visualize the price prediction
    plt.figure(figsize=(10, 6))
    plt.plot(df_grouped['tradeTime'], df_grouped['price_rolling'], label='Original Price (Rolling)')
    plt.plot(predictions_df['tradeTime'], predictions_df['Predicted_Price'], label='Predicted Price', linestyle='--')
    plt.plot(future_predictions_df['tradeTime'], future_predictions_df['Predicted_Price'], label='Future Predicted Price', linestyle='--')
    plt.title('Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(price_predictions_picture_save_path)
    plt.show()

if __name__ == "__main__":
    predict_main()
