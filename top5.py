import os
import pickle
import numpy as np

base_path = os.path.abspath(os.path.dirname(__file__))
predictions_path = os.path.join(base_path, 'predictions.pkl')
ticker_names_path = os.path.join(base_path, 'ticker_names.pkl')

with open(predictions_path, 'rb') as f:
    predictions = pickle.load(f)

with open(ticker_names_path, 'rb') as f:
    ticker_names = pickle.load(f)
def find_closest_stocks(input_price):
    closest_stocks = []
    min_price = input_price * 0.9  # 입력 가격의 -10%

    for ticker, (pred, latest_price) in predictions.items():
        if min_price <= latest_price <= input_price:
            closest_stocks.append((ticker, pred, latest_price))

    closest_stocks.sort(key=lambda x: abs(x[2] - input_price))
    return closest_stocks


def print_recommendations(closest_stocks):
    recommendations = []
    for ticker, pred, latest_price in closest_stocks:
        change_rate = ((pred - latest_price) / latest_price) * 100
        if isinstance(change_rate, float) and 1 <= change_rate <= 4:
            recommendations.append({
                'ticker': ticker,
                'name': ticker_names.get(ticker, ticker),
                'latest_price': float(latest_price),
                'predicted_price': float(pred),
                'change_rate': change_rate
            })

    recommendations.sort(key=lambda x: x['change_rate'], reverse=True)
    return recommendations[:5]


if __name__ == "__main__":
    # 이 부분은 직접 실행할 때만 사용됩니다.
    input_price = float(input("Enter the stock price to find closest prediction: "))
    closest_stocks = find_closest_stocks(input_price, predictions)
    recommendations = print_recommendations(closest_stocks, ticker_names)

    if not recommendations:
        print("추천 종목 없음")
    else:
        print("추천 종목 (상위 5개):")
        for rec in recommendations:
            print(f"티커 {rec['ticker']} ({rec['name']})")
            print(f"최신 종가: {rec['latest_price']}")
            print(f"예측된 1일차 가격: {rec['predicted_price']} (등락률: {rec['change_rate']:.2f}%)")
            print()