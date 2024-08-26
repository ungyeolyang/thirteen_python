from flask import Flask, jsonify, request
from flask_cors import CORS
from 엘라스틱서치 import get_stock_from_elasticsearch, send_to_elasticsearch
from 카드엘라스틱서치 import get_card_from_elasticsearch,search_top_cards
from top5 import find_closest_stocks, print_recommendations
from predict import predict_for_stock_number
import os
import sys
from werkzeug.utils import secure_filename
from ex import process_excel

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])


@app.route('/')
def hello():
    return 'Hello, Flask!'


@app.route('/api/save', methods=['GET'])
def save_stock():
    try:
        send_to_elasticsearch()
        return jsonify({"status": "success", "message": "Data sent to Elasticsearch"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/stock', methods=['GET'])
def search_stock():
    query = request.args.get('query', '')  # URL 쿼리 매개변수로부터 검색어를 가져옵니다.

    # Elasticsearch에서 검색
    search_results = get_stock_from_elasticsearch(query if query else None)

    if search_results is None:
        return jsonify({"error": "검색 중 오류가 발생했습니다."}), 500

    return jsonify(search_results)

@app.route('/api/card', methods=['GET'])
def search_card():
    query = request.args.get('query', '')  # URL 쿼리 매개변수로부터 검색어를 가져옵니다.

    # Elasticsearch에서 검색
    search_results = get_card_from_elasticsearch(query if query else None)

    if search_results is None:
        return jsonify({"error": "검색 중 오류가 발생했습니다."}), 500

    return jsonify(search_results)

sys.path.append(os.path.join(os.path.dirname(__file__), 'routes'))

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/data', methods=['GET'])
def get_data():
    # 업로드된 파일이 있는지 확인
    uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not uploaded_files:
        return jsonify({'error': 'No file uploaded'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_files[0])

    try:
        category_totals = process_excel(file_path)
        return jsonify({'category_totals': category_totals})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        category_totals = process_excel(file_path)
        return jsonify({'category_totals': category_totals})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/topcard', methods=['GET'])
def get_top_cards():
    # 쿼리 매개변수에서 'categories'를 가져옵니다.
    categories_param = request.args.get('categories', '')
    if not categories_param:
        return jsonify({"error": "카테고리를 제공해야 합니다."}), 400

    # 카테고리를 쉼표로 구분하여 리스트로 변환합니다.
    categories = [cat.strip() for cat in categories_param.split(',')]

    # 초기 카드 검색
    card_scores = search_top_cards(categories)

    # 점수를 기준으로 내림차순 정렬
    sorted_cards = sorted(card_scores, key=lambda x: x[1], reverse=True)

    # 상위 5개 카드 선택
    top_5_cards = sorted_cards[:5]

    # 결과 반환
    return jsonify([card[0] for card in top_5_cards])


@app.route('/api/top5', methods=['GET'])
def get_recommendations():
    # 쿼리 파라미터에서 입력 가격 받기
    input_price = request.args.get('price', type=float)

    if input_price is None:
        return jsonify({'error': 'Price parameter is required'}), 400

    closest_stocks = find_closest_stocks(input_price)
    recommendations = print_recommendations(closest_stocks)

    return jsonify(recommendations)

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    # 쿼리 파라미터에서 주식번호 받기
    stock_number = request.args.get('stock', type=str)

    if stock_number is None:
        return jsonify({'error': 'Stock number parameter is required'}), 400

    latest_prices, predicted_prices = predict_for_stock_number(stock_number)

    if latest_prices is None:
        return jsonify({'error': f'Prediction for stock number {stock_number} could not be generated'}), 404

    return jsonify({
        'stock_number': stock_number,
        'latest_prices': latest_prices,
        'predicted_prices': predicted_prices
    })


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
