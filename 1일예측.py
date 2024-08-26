import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pykrx import stock

# 모든 주식 티커를 가져오는 함수
def get_all_stock_tickers():
    kospi_tickers = stock.get_market_ticker_list(market="KOSPI")
    tickers = kospi_tickers
    return tickers

# 특정 티커의 주식 데이터를 가져오는 함수
def get_stock_data(ticker, start_date, end_date):
    df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)
    return df

# 티커와 이름 매핑 함수
def get_ticker_names():
    tickers = get_all_stock_tickers()
    ticker_names = {ticker: stock.get_market_ticker_name(ticker) for ticker in tickers}
    return ticker_names

# 데이터셋을 생성하는 함수
def create_dataset(data, look_back=60):
    X, y = [], []
    data = np.array(data)
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 1일 후 예측을 위한 데이터셋 생성 함수
def create_future_dataset(model, data, look_back=60):
    future_X = data[-look_back:]
    future_X = np.expand_dims(future_X, axis=0)
    next_pred = model.predict(future_X, verbose=0)[0]
    return next_pred

# 데이터 학습 및 예측 함수
def train_and_predict_for_ticker(ticker, df, scaler, look_back=60):
    data = df[['종가_scaled']].values
    train_data_len = int(np.ceil(len(data) * 0.8))
    train_data = data[:train_data_len]
    test_data = data[train_data_len - look_back:]

    if len(train_data) < look_back or len(test_data) < look_back:
        print(f"티커 {ticker}에 대한 데이터가 충분하지 않아 건너뜁니다...")
        return ticker, None, None

    X_train, y_train = create_dataset(train_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(20, return_sequences=True))  # LSTM 유닛 수 조정
    model.add(Dropout(0.2))  # 드롭아웃 비율 조정
    model.add(LSTM(20, return_sequences=False))  # LSTM 유닛 수 조정
    model.add(Dropout(0.2))  # 드롭아웃 비율 조정
    model.add(Dense(10))  # Dense 레이어 크기 축소
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # patience 축소
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras', monitor='val_loss', save_best_only=True, verbose=1
    )
    model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2,
              callbacks=[early_stop, model_checkpoint])  # 배치 크기 증가

    predicted_stock_price = create_future_dataset(model, test_data, look_back)

    predicted_stock_price_full = np.concatenate(
        [predicted_stock_price.reshape(-1, 1), np.zeros((predicted_stock_price.shape[0], 3))], axis=1)
    predicted_stock_price_original = scaler.inverse_transform(predicted_stock_price_full)[:, 0]

    latest_price = df['종가'].values[-1]
    return ticker, predicted_stock_price_original[0], latest_price

# 데이터 학습 및 예측 결과 저장 함수
def train_and_save_predictions(data_dict, scalers, look_back=60):
    predictions = {}
    max_threads = 12  # CPU 스레드 수에 맞춰 병렬화
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(
                train_and_predict_for_ticker,
                ticker,
                data_dict[ticker],
                scalers[ticker],
                look_back
            ) for ticker in data_dict.keys()
        ]
        for future in futures:
            ticker, predicted_stock_price, latest_price = future.result()
            if predicted_stock_price is not None:
                predictions[ticker] = (predicted_stock_price, latest_price)
                print(f"Saved prediction for {ticker}: Predicted Price: {predicted_stock_price}, Latest Price: {latest_price}")

    base_path = os.path.abspath(os.path.dirname(__file__))
    predictions_path = os.path.join(base_path, 'predictions.pkl')
    with open(predictions_path, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    start_time = datetime.now()  # 시작 시간 기록
    start_date = "2024-01-06"
    end_date = datetime.today().strftime('%Y-%m-%d')  # 오늘 날짜로 설정

    # 데이터 수집 및 전처리 수행
    tickers = get_all_stock_tickers()
    data_dict = {}
    scalers = {}

    for i, ticker in enumerate(tickers):  # 각 티커에 대해
        try:
            df = get_stock_data(ticker, start_date, end_date)
            df.reset_index(inplace=True)

            # 데이터가 비어 있는 경우 건너뛰기
            if df.empty:
                print(f"No data for ticker {ticker}")
                continue

            data_dict[ticker] = df

            # 정규화
            data = df['종가'].values.reshape(-1, 1)  # 종가 데이터를 numpy 배열로 변환
            if len(data) == 0:  # 데이터가 비어 있는 경우 건너뛰기
                continue
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)  # 데이터 정규화
            scalers[ticker] = scaler
            df['종가_scaled'] = scaled_data  # 데이터 프레임에 '종가_scaled' 열 추가

            # 진행 상황 출력
            print(f"Processed {i + 1}/{len(tickers)}: {ticker}")
        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")

    base_path = os.path.abspath(os.path.dirname(__file__))
    data_dict_path = os.path.join(base_path, 'data_dict.pkl')
    scalers_path = os.path.join(base_path, 'scalers.pkl')

    try:
        with open(data_dict_path, 'wb') as f:
            pickle.dump(data_dict, f)  # data_dict를 data_dict.pkl 파일로 저장
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)  # scalers를 scalers.pkl 파일로 저장
        print(f"Data saved to {data_dict_path} and {scalers_path}")
    except Exception as e:
        print(f"An error occurred while saving files: {e}")

    # 티커 이름 매핑 파일 저장
    ticker_names = get_ticker_names()
    ticker_names_path = os.path.join(base_path, 'ticker_names.pkl')

    try:
        with open(ticker_names_path, 'wb') as f:
            pickle.dump(ticker_names, f)  # 티커 이름 매핑을 ticker_names.pkl 파일로 저장
        print(f"Ticker names saved to {ticker_names_path}")
    except Exception as e:
        print(f"An error occurred while saving ticker_names.pkl: {e}")

    train_and_save_predictions(data_dict, scalers)

    end_time = datetime.now()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 총 소요 시간 계산

    print(f"Program started at: {start_time}")
    print(f"Program ended at: {end_time}")

    print(f"Total elapsed time: {elapsed_time}")