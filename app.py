import numpy as np
from flask import Flask, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf

app = Flask(__name__)

# Function to get historical stock prices
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print(stock_data)
    return stock_data

# Function to prepare data for training
def prepare_data(data, target_column, window_size):
    data['Target'] = data[target_column].shift(-1)  # Shift the target column by one day
    data = data.dropna()  # Drop NaN values
    features = []
    targets = []

    for i in range(len(data) - window_size):
        features.append(data[target_column].values[i:i+window_size])
        targets.append(data['Target'].values[i+window_size])

    return np.array(features), np.array(targets)

# Function to train the model
def train_model(features, targets):
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Function to make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

# Function to evaluate the model
def evaluate_model(y_test, predictions):
    mse = np.mean((predictions - y_test)**2)
    return mse

# API route to get predictions
@app.route('/predict_stock', methods=['POST'])
def predict_stock():
    content = request.get_json()
    ticker = content['ticker']
    start_date = content['start_date']
    end_date = content['end_date']
    window_size = content['window_size']

    print("ticker", ticker)
    # Get historical stock data
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Prepare data
    features, targets = prepare_data(stock_data, 'Close', window_size)
    print("features and targets length", len(features), len(targets))

    # Train the model
    model, X_test, y_test = train_model(features, targets)

    # Make predictions
    predictions = make_predictions(model, X_test)

    # Evaluate the model
    mse = evaluate_model(y_test, predictions)

    # Convert NumPy arrays to Python lists
    y_test_list = y_test.tolist()
    predictions_list = predictions.tolist()

    # Create a response dictionary
    response = {
        'actual_values': y_test_list,
        'predicted_values': predictions_list,
        # 'mse': mse
    }

    return jsonify(response)


def get_last_traded_price(data):
    # Check if data is not empty
    if not data.empty:
        last_traded_price = data['Close'].iloc[-1]
        return last_traded_price
    else:
        return None

# Function to get basic information about a stock
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d')
    last_traded_price = get_last_traded_price(data)
    info = stock.info
    print(stock.info)
    return {
        'symbol': info.get('symbol', ''),
        'company_name': info.get('longName', ''),
        'last_price': last_traded_price,
        'currency': info.get('currency', ''),
        'info': info
    }

# API route to get basic stock information
@app.route('/stock_info/<ticker>', methods=['GET'])
def stock_info(ticker):
    try:
        stock_data = get_stock_info(ticker)
        return jsonify(stock_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# API route to get history of stock
@app.route('/stock_history', methods=['POST'])
def stock_history():
    try:
        content = request.get_json()
        ticker = content['ticker']
        start_date = content['start_date']
        end_date = content['end_date']
        print(ticker, start_date, end_date)
        stock_data = get_stock_data(ticker, start_date, end_date)
        stock_info = get_stock_info(ticker)
        stock_history = stock_data.to_dict('records')
        print(stock_history)
        responseData = {"stock": stock_history, "info": stock_info}
        print(responseData)
        return jsonify(responseData)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)