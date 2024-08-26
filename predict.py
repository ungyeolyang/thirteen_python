import os
import pickle
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

base_path = os.path.abspath(os.path.dirname(__file__))
data_dict_path = os.path.join(base_path, 'data_dict.pkl')
scalers_path = os.path.join(base_path, 'scalers.pkl')

# 파일에서 데이터 로드
with open(data_dict_path, 'rb') as f:
    data_dict = pickle.load(f)

with open(scalers_path, 'rb') as f:
    scalers = pickle.load(f)
# 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 시드를 설정합니다.
set_seed()

cache = {}

# 데이터셋을 생성하는 함수
def create_dataset(data, look_back=60):
    X, y = [], []
    data = np.array(data)
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 3일 후 예측을 위한 데이터셋 생성 함수
def create_future_dataset(model, data, look_back=60, future_days=3):
    future_X = data[-look_back:]
    future_X = np.expand_dims(future_X, axis=0)
    predictions = []
    for _ in range(future_days):
        next_pred = model.predict(future_X, verbose=0)[0]
        next_pred = np.expand_dims(next_pred, axis=-1)
        future_X = np.append(future_X[:, 1:, :], np.expand_dims(next_pred, axis=1), axis=1)
        predictions.append(next_pred)
    return np.array(predictions)

# 입력받은 주식번호에 대해 예측하는 함수
def predict_for_stock_number(stock_number, look_back=60, future_days=3):
    # 캐시에서 검색된 주식번호 확인
    if stock_number in cache:
        print(f"캐시에서 {stock_number}에 대한 예측 결과를 가져옵니다.")
        predicted_prices = cache[stock_number]
        # 최신 가격을 함께 반환하기 위해 데이터 프레임에서 가져오기
        df = data_dict.get(stock_number)
        if df is None:
            print(f"Data for stock number {stock_number} not found in cache.")
            return None, None
        latest_prices = df['종가'].values[-60:]
        return latest_prices.tolist(), predicted_prices.tolist()

    else:
        df = data_dict.get(stock_number)
        scaler = scalers.get(stock_number)

        if df is None or scaler is None:
            print(f"Data for stock number {stock_number} not found.")
            return None, None

        data = df[['종가_scaled']].values
        train_data_len = int(np.ceil(len(data) * 0.8))
        train_data = data[:train_data_len]
        test_data = data[train_data_len - look_back:]

        if len(train_data) < look_back or len(test_data) < look_back:
            print(f"Stock number {stock_number}에 대한 데이터가 충분하지 않아 건너뜁니다...")
            return None, None

        X_train, y_train = create_dataset(train_data, look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(30, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(30, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(15))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks=[early_stop])

        # Debugging: Check the training history
        print("Training history:", history.history)

        try:
            predicted_stock_price = create_future_dataset(model, test_data, look_back, future_days)

            predicted_stock_price = np.squeeze(predicted_stock_price, axis=1)
            predicted_stock_price_full = np.concatenate(
                [predicted_stock_price, np.zeros((predicted_stock_price.shape[0], 1))], axis=1)
            predicted_stock_price_original = scaler.inverse_transform(predicted_stock_price_full)[:, 0]

            # 캐시에 저장
            print(f"캐시에 {stock_number}에 대한 예측 결과를 저장합니다.")
            cache[stock_number] = predicted_stock_price_original
            predicted_prices = predicted_stock_price_original

            # 예측 결과와 마지막 60일 가격 반환
            latest_prices = df['종가'].values[-60:]
            return latest_prices.tolist(), predicted_prices.tolist()

        except Exception as e:
            print(f"예측 생성 중 오류 발생: {e}")
            return None, None

    return None, None

if __name__ == "__main__":
    base_path = os.path.abspath(os.path.dirname(__file__))
    data_dict_path = os.path.join(base_path, 'data_dict.pkl')
    scalers_path = os.path.join(base_path, 'scalers.pkl')
    ticker_names_path = os.path.join(base_path, 'ticker_names.pkl')

    try:
        with open(data_dict_path, 'rb') as f:
            data_dict = pickle.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {data_dict_path}")
        exit(1)
    except Exception as e:
        print(f"data_dict.pkl을 로드하는 동안 오류가 발생했습니다: {e}")
        exit(1)

    try:
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {scalers_path}")
        exit(1)
    except Exception as e:
        print(f"scalers.pkl을 로드하는 동안 오류가 발생했습니다: {e}")
        exit(1)

    try:
        with open(ticker_names_path, 'rb') as f:
            ticker_names = pickle.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {ticker_names_path}")
        exit(1)
    except Exception as e:
        print(f"ticker_names.pkl을 로드하는 동안 오류가 발생했습니다: {e}")
        exit(1)

    input_stock_number = input("Enter the stock name to predict: ")
    predict_for_stock_number(input_stock_number, data_dict, scalers, ticker_names)