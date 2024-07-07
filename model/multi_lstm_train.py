import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step)]
        X.append(a)
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

def train_main(train_path, model_path, scaler_followers_path, scaler_price_path):
    # Check if GPU is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("GPUs Available: ", physical_devices)
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
    else:
        print("No GPUs found. Using CPU...")

    # Load the data
    df = pd.read_csv(train_path)

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
    df_grouped['followers'] = df_grouped['followers'].rolling(window=window_size).mean()
    df_grouped['price'] = df_grouped['price'].rolling(window=window_size).mean()

    # Drop NaN values created by rolling mean
    df_grouped.dropna(inplace=True)

    # Scale the data
    scaler_followers = MinMaxScaler(feature_range=(0, 1))
    followers_scaled = scaler_followers.fit_transform(np.array(df_grouped['followers']).reshape(-1, 1))

    scaler_price = MinMaxScaler(feature_range=(0, 1))
    price_scaled = scaler_price.fit_transform(np.array(df_grouped['price']).reshape(-1, 1))

    # Create dataset for LSTM
    time_step = 365  # One year of daily data
    X_followers, Y_followers = create_dataset(followers_scaled, time_step)
    X_price, Y_price = create_dataset(price_scaled, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_followers = X_followers.reshape(X_followers.shape[0], X_followers.shape[1], 1)
    X_price = X_price.reshape(X_price.shape[0], X_price.shape[1], 1)

    # Split the data into train and test sets
    train_size = int(len(X_followers) * 0.8)
    X_train_followers, X_test_followers = X_followers[:train_size], X_followers[train_size:]
    X_train_price, X_test_price = X_price[:train_size], X_price[train_size:]
    Y_train_price, Y_test_price = Y_price[:train_size], Y_price[train_size:]

    # Build the LSTM model for followers prediction
    followers_input = Input(shape=(time_step, 1))
    followers_lstm = LSTM(50, return_sequences=True)(followers_input)
    followers_lstm = LSTM(50, return_sequences=False)(followers_lstm)
    followers_dense = Dense(25)(followers_lstm)
    followers_output = Dense(1)(followers_dense)

    # Build the LSTM model for price prediction
    price_input = Input(shape=(time_step, 1))
    price_lstm = LSTM(50, return_sequences=True)(price_input)
    price_lstm = LSTM(50, return_sequences=False)(price_lstm)
    price_dense = Dense(25)(price_lstm)


    # Concatenate the outputs of both LSTM models
    concatenated = Concatenate()([followers_output, price_dense])
    concatenated_dense = Dense(25)(concatenated)
    final_output = Dense(1)(concatenated_dense)

    # Combine the models
    model = Model(inputs=[followers_input, price_input], outputs=final_output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit([X_train_followers, X_train_price], Y_train_price, batch_size=48, epochs=15, validation_split=0.2)

    # Save the model
    model.save(model_path)

    # Save the scalers
    np.save(scaler_followers_path, scaler_followers)
    np.save(scaler_price_path, scaler_price)

if __name__ == "__main__":
    train_main()
