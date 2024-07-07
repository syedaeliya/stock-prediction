import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = 'your_secret_key_here'

# Path to upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess and predict using LSTM model
def preprocess_and_predict(file_path):
    try:
        # Load data
        data = pd.read_csv(file_path)

        print(f"Original Data:\n{data.head()}")  # Print loaded data for debugging

        # Clean the dataset
        data = clean_data(data)

        print(f"Cleaned Data:\n{data.head()}")  # Print cleaned data for debugging

        if data.empty:
            raise ValueError("No valid data after cleaning")

        # Use multiple features for prediction
        features = ['close', 'volume', 'open', 'high', 'low']  # Adjusted to lowercase consistent with cleaned data
        prices = data[features].values

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # Create training and test sets
        train_size = int(len(scaled_prices) * 0.8)
        train_data = scaled_prices[:train_size]
        test_data = scaled_prices[train_size:]

        # Prepare data for LSTM
        look_back = 5
        X_train, y_train = create_dataset(train_data, look_back)
        X_test, y_test = create_dataset(test_data, look_back)

        # Ensure there is enough data
        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            raise ValueError("Not enough data to create the specified look_back dataset.")

        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, len(features))))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Implement early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

        # Train the model
        history = model.fit(X_train, y_train, batch_size=1, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stop])

        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], len(features)-1))), axis=1))[:,0]

        # Prepare data for plotting
        test_dates = data['date'][train_size + look_back + 1:]

        return test_dates, data['close'][train_size + look_back + 1:], predictions, os.path.basename(file_path)

    except Exception as e:
        print(f"Error in preprocess_and_predict: {e}")
        return None, None, None, None

# Function to clean the dataset
def clean_data(data):
    # Convert column headers to lowercase
    data.columns = data.columns.str.lower()

    # Drop rows with missing or invalid values
    data.dropna(inplace=True)

    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')

    # Drop rows with invalid dates
    data.dropna(subset=['date'], inplace=True)

    # Convert numeric columns to numeric
    numeric_cols = ['open', 'high', 'low', 'close']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Clean volume column
    data['volume'] = data['volume'].replace({',': ''}, regex=True)
    data['volume'] = pd.to_numeric(data['volume'], errors='coerce')

    # Drop any remaining rows with missing values
    data.dropna(inplace=True)

    return data

# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 0])  # Predicting 'close' price
    return np.array(X), np.array(Y)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No selected file")

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            test_dates, actual_prices, predicted_prices, file_name = preprocess_and_predict(file_path)

            if test_dates is None or actual_prices is None or predicted_prices is None or file_name is None:
                return render_template('index.html', error="Error processing file")

            # Calculate accuracy metrics
            accuracy = calculate_accuracy(actual_prices, predicted_prices)

            # Plot results
            plt.figure(figsize=(20, 10), facecolor=(0,0,0,0))  
            plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.05)  # Adjust margins
            plt.grid(color='white', linestyle='--', linewidth=0.5)
            ax = plt.gca()  # Get the current Axes instance
            # Set axis background color
            ax.set_facecolor((0, 0, 0, 0.3))
            plt.plot(test_dates, actual_prices, label='Actual Prices', color="blue")
            plt.plot(test_dates, predicted_prices, label='Predicted Prices', color="green")
            ax.tick_params(axis='x', colors='white')  # Set x-axis tick color
            ax.tick_params(axis='y', colors='white')  # Set y-axis tick color
            plt.xlabel('Date',color='white')
            plt.ylabel('Stock Price',color='white')
            plt.legend()
            plot_path = os.path.join('static', 'plot.png')
            plt.savefig(plot_path)

            return render_template('index.html', plot=plot_path, file_name=file_name, accuracy=accuracy)

    except Exception as e:
        print(f"Error in upload_file: {e}")
        return render_template('index.html', error="Error uploading file")

def calculate_accuracy(actual, predicted):
    return (1 - np.mean(np.abs((actual - predicted) / actual))) * 100

if __name__ == '__main__':
    app.run(debug=True)
