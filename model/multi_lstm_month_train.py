import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
import tensorflow as tf
import joblib

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step)]
        X.append(a)
        Y.append(data[i + time_step])
    return np.array(X), np.array(Y)

def train_main_month(train_path, model_path, scaler_followers_path, scaler_price_path, scaler_total_price_path, scaler_square_path):
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

    # Scale the data
    scaler_followers = MinMaxScaler(feature_range=(0, 1))
    followers_scaled = scaler_followers.fit_transform(np.array(df_grouped['followers']).reshape(-1, 1))

    scaler_price = MinMaxScaler(feature_range=(0, 1))
    price_scaled = scaler_price.fit_transform(np.array(df_grouped['price']).reshape(-1, 1))

    scaler_total_price = MinMaxScaler(feature_range=(0, 1))
    total_price_scaled = scaler_total_price.fit_transform(np.array(df_grouped['totalPrice']).reshape(-1, 1))

    scaler_square = MinMaxScaler(feature_range=(0, 1))
    square_scaled = scaler_square.fit_transform(np.array(df_grouped['square']).reshape(-1, 1))

    # Create dataset for LSTM
    time_step = 12  # One year of monthly data
    X_followers, Y_followers = create_dataset(followers_scaled, time_step)
    X_price, Y_price = create_dataset(price_scaled, time_step)
    X_total_price, Y_total_price = create_dataset(total_price_scaled, time_step)
    X_square, Y_square = create_dataset(square_scaled, time_step)

    # Reshape input to be [samples, time steps, features] which is required for LSTM
    X_followers = X_followers.reshape(X_followers.shape[0], X_followers.shape[1], 1)
    X_price = X_price.reshape(X_price.shape[0], X_price.shape[1], 1)
    X_total_price = X_total_price.reshape(X_total_price.shape[0], X_total_price.shape[1], 1)
    X_square = X_square.reshape(X_square.shape[0], X_square.shape[1], 1)

    # Split the data into train and test sets
    train_size = int(len(X_followers) * 0.8)
    X_train_followers, X_test_followers = X_followers[:train_size], X_followers[train_size:]
    X_train_price, X_test_price = X_price[:train_size], X_price[train_size:]
    X_train_total_price, X_test_total_price = X_total_price[:train_size], X_total_price[train_size:]
    X_train_square, X_test_square = X_square[:train_size], X_square[train_size:]
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
    price_output = Dense(1)(price_dense)

    # Build the LSTM model for total price prediction
    total_price_input = Input(shape=(time_step, 1))
    total_price_lstm = LSTM(50, return_sequences=True)(total_price_input)
    total_price_lstm = LSTM(50, return_sequences=False)(total_price_lstm)
    total_price_dense = Dense(25)(total_price_lstm)
    total_price_output = Dense(1)(total_price_dense)

    # Build the LSTM model for square prediction
    square_input = Input(shape=(time_step, 1))
    square_lstm = LSTM(50, return_sequences=True)(square_input)
    square_lstm = LSTM(50, return_sequences=False)(square_lstm)
    square_dense = Dense(25)(square_lstm)
    square_output = Dense(1)(square_dense)

    # Concatenate the outputs of all LSTM models
    concatenated = Concatenate()([followers_output, price_output, total_price_output, square_output])
    concatenated_dense = Dense(25)(concatenated)
    final_output = Dense(1)(concatenated_dense)

    # Combine the models
    model = Model(inputs=[followers_input, price_input, total_price_input, square_input], outputs=final_output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit([X_train_followers, X_train_price, X_train_total_price, X_train_square], Y_train_price, batch_size=3, epochs=5, validation_split=0.2)

    # Save the model
    model.save(model_path)

    # Save the scalers
    joblib.dump(scaler_followers, scaler_followers_path)
    joblib.dump(scaler_price, scaler_price_path)
    joblib.dump(scaler_total_price, scaler_total_price_path)
    joblib.dump(scaler_square, scaler_square_path)

if __name__ == "__main__":
    train_main_month('path/to/your/data.csv', 'model_month.h5', 'scaler_followers_month.pkl', 'scaler_price_month.pkl', 'scaler_total_price_month.pkl', 'scaler_square_month.pkl')
