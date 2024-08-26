import requests
import pandas as pd
import re

# Google Places API 키 (본인의 API 키를 입력하세요)
api_key = 'AIzaSyCdt5Kz-mFlomO-KzYlUqjBLOsveD9SE-0'

# API 호출 결과 캐싱
place_type_cache = {}

def get_place_type(place_name, location='37.7749,-122.4194', radius='1500'):
    """Google Places API를 사용하여 가맹점명의 업종을 가져옵니다."""
    if place_name in place_type_cache:
        return place_type_cache[place_name]

    place_name_encoded = requests.utils.quote(place_name)
    url = f'https://maps.googleapis.com/maps/api/place/textsearch/json?query={place_name_encoded}&location={location}&radius={radius}&key={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        places_data = response.json()

        if 'results' in places_data and len(places_data['results']) > 0:
            place_types = places_data['results'][0]['types']
            place_type_cache[place_name] = place_types
            return place_types
        else:
            return ['정보 없음']
    except requests.RequestException as e:
        print(f"API 호출 실패: {e}")
        return ['정보 없음']

def preprocess_text(text):
    """텍스트를 소문자로 변환하고, 특수 문자를 제거합니다."""
    return re.sub(r'\s+', ' ', text.lower().strip())

def classify_place_types(place_types, place_name=None):
    """업종 정보를 사용자 정의 카테고리로 변환합니다."""
    type_mapping = {
        'cafe': '식비',
        'restaurant': '식비',
        'meal_delivery': '식비',
        'meal_takeaway': '식비',
        'food': '식비',
        'store': '쇼핑',
        'shopping_mall': '쇼핑',
        'subway_station': '교통',
        'transit_station': '교통',
        'supermarket': '마트/편의점',
        'shoe_store': '쇼핑',
        'grocery_or_supermarket': '마트/편의점',
        'bakery': '카페.간식',
        'bus_station': '교통',
        'movie_theater': '취미.여가',
        'hospital': '병원',
        'convenience_store': '마트/편의점',
        'doctor': '병원',
        'pharmacy': '병원',
        'book_store': '서점',
        'beauty_salon': '미용',
        'hair_care': '미용',
        'home_goods_store': '쇼핑',
        'bar': '취미.여가',
        'electronics_store': '쇼핑',
        'gym': '운동',
        'embassy': '공공기관',
        'insurance_agency': '공공기관',
        'airport': '교통',
        'amusement_park': '취미.여가',
        'aquarium': '취미.여가',
        'art_gallery': '취미.여가',
        'atm': '은행',
        'bank': '은행',
        'bicycle_store': '쇼핑',
        'bowling_alley': '취미.여가',
        'car_dealer': '쇼핑',
        'car_rental': '교통',
        'car_repair': '기타',
        'car_wash': '기타',
        'casino': '취미.여가',
        'cemetery': '기타',
        'church': '기타',
        'city_hall': '공공기관',
        'clothing_store': '쇼핑',
        'courthouse': '공공기관',
        'dentist': '병원',
        'department_store': '쇼핑',
        'drugstore': '병원',
        'electrician': '기타',
        'fire_station': '공공기관',
        'florist': '쇼핑',
        'funeral_home': '기타',
        'gas_station': '교통',
        'hardware_store': '쇼핑',
        'hindu_temple': '기타',
        'jewelry_store': '쇼핑',
        'laundry': '기타',
        'lawyer': '기타',
        'library': '취미.여가',
        'light_rail_station': '교통',
        'liquor_store': '취미.여가',
        'local_government_office': '공공기관',
        'locksmith': '기타',
        'lodging': '숙소',
        'mosque': '기타',
        'movie_rental': '취미.여가',
        'moving_company': '취미.여가',
        'museum': '취미.여가',
        'night_club': '취미.여가',
        'painter': '기타',
        'park': '취미.여가',
        'parking': '교통',
        'pet_store': '쇼핑',
        'physiotherapist': '병원',
        'plumber': '기타',
        'police': '공공기관',
        'post_office': '공공기관',
        'primary_school': '교육',
        'real_estate_agency': '기타',
        'roofing_contractor': '기타',
        'rv_park': '숙소',
        'school': '교육',
        'secondary_school': '교육',
        'spa': '미용',
        'stadium': '운동',
        'storage': '기타',
        'synagogue': '기타',
        'taxi_stand': '교통',
        'tourist_attraction': '취미.여가',
        'train_station': '교통',
        'travel_agency': '기타',
        'university': '교육',
        'veterinary_care': '병원',
        'zoo': '취미.여가'
    }

    # 업종 분류
    for place_type in place_types:
        if place_type in type_mapping:
            return type_mapping[place_type]

    # 가맹점명을 기반으로 업종 분류 보강
    if place_name:
        place_name_preprocessed = preprocess_text(place_name)
        korean_keywords = {
            '식비': ['식당', '밥', '버거', '치킨','통닭','집','찌개','휴게소','맥도날드','롯데리아','kfc','돼지','무한리필'],
            '병원': ['의원', '외과', '내과', '인과', '외과', '피부과', '후과', '안과','약국','의학과'],
            '마트/편의점': ['마트', '할인점', '마켓', 'gs25', '씨유','CU', '스톱', '일레븐', '몰','지에스','슈퍼','식자재'],
            '쇼핑': ['쿠팡','수입','세외','인터넷'],
            '보험': ['보험', '화재','생명'],
            '카페.간식': ['카페', '커피', '다방', '스타벅스', '떡', '할인점', '브레드', 'bread','바게뜨','뚜레쥬르','프레소','더벤티','아이스크림','배스킨라빈스'],
            '회사': ['주식회사', '(주)', '기업', '유통', '전자'],
            '운동': ['짐', '헬스'],
            '취미.여가': ['노래', '장'],
            '미용': ['헤어','미용실'],
            '서점':['문고','문구'],
            '전자/통신':['전자','Apple','LG U+','SKT','KT'],
            '숙소':['여기어때','야놀자','Agoda','AIRBNB'],
            '이체.페이': ['페이', 'kcp'],
            'pc': ['pc', '피시방', '피시', '피씨'],
            '교통': ['주휴소', '택시', '지하철', '항공', '교통', '카', '버스','공항'],
        }

        for category, keywords in korean_keywords.items():
            if any(preprocess_text(keyword) in place_name_preprocessed for keyword in keywords):
                return category

    return '기타'


def process_excel(file_path):
    try:
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path, header=None)

        # 헤더 찾기
        header_row = None
        for i, row in df.iterrows():
            if '가맹점명' in row.values or '이용하신곳' in row.values:
                header_row = i
                break

        if header_row is None:
            raise ValueError("'가맹점명' 열이 데이터에 없습니다.")

        # 헤더를 이용하여 다시 데이터프레임 로드
        df = pd.read_excel(file_path, skiprows=header_row)

        # '가맹점명' 열 확인
        if '가맹점명' not in df.columns and '이용하신곳' not in df.columns:
            raise ValueError("'가맹점명' 또는 '이용하신곳' 열이 데이터에 없습니다.")

        # 업종 분류
        df['업종'] = df['가맹점명'].map(
            lambda x: classify_place_types(get_place_type(x), place_name=x) if pd.notna(x) else '기타')

        # '매출금액', '거래금액', '승인금액' 열 확인
        if '매출금액' in df.columns:
            amount_col = '매출금액'
        elif '거래금액' in df.columns:
            amount_col = '거래금액'
        elif '승인금액' in df.columns:
            amount_col = '승인금액'
        elif '이용금액' in df.columns:
            amount_col = '이용금액'
        elif '원화사용금액' in df.columns:
            amount_col = '원화사용금액'
        elif '사용금액' in df.columns:
            amount_col = '사용금액'
        else:
            raise ValueError("매출금액, 거래금액, 이용금액, 또는 승인금액 열이 데이터에 없습니다.")

        # 금액 열을 숫자로 변환
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')

        # '기타' 업종 데이터 확인
        기타_df = df[df['업종'] == '기타']
        for idx, row in 기타_df.iterrows():
            print(f"가맹점명/이용하신곳: {row['가맹점명']}, 업종: {row['업종']}, {amount_col}: {row[amount_col]}")

        # 카테고리별 총합 계산
        category_totals = df.groupby('업종')[amount_col].sum().to_dict()
        return category_totals

    except Exception as e:
        print(f"Error processing file: {e}")
        raise
