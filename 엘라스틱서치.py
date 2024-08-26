from elasticsearch import Elasticsearch
from pykrx import stock

# 모든 주식 티커와 회사명을 가져오는 함수
def get_all_stock_tickers():
    tickers = stock.get_market_ticker_list(market="KOSPI")
    ticker_dict = {}

    for ticker in tickers:
        # 주식 코드로부터 회사명 가져오기
        company_name = stock.get_market_ticker_name(ticker)
        ticker_dict[ticker] = f"{company_name}({ticker})"

    return ticker_dict


# Elasticsearch 인덱스를 생성하는 함수 (nori 분석기 포함)
def create_index_with_nori_analyzer():
    es = Elasticsearch("http://localhost:9200")

    index_name = "stock_tickers"

    # 인덱스가 존재하는지 확인
    if es.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists.")
        return

    # nori 분석기를 사용한 인덱스 생성
    index_body = {
        "settings": {
            "analysis": {
                "tokenizer": {
                    "nori_tokenizer": {
                        "type": "nori_tokenizer",
                        "decompound_mode": "discard"
                    }
                },
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "filter": ["lowercase"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "ticker": {
                    "type": "keyword"  # 티커는 그대로 저장되므로 키워드 타입 사용
                },
                "company": {
                    "type": "text",
                    "analyzer": "nori_analyzer"
                }
            }
        }
    }

    es.indices.create(index=index_name, body=index_body)
    print(f"Index '{index_name}' created with nori analyzer.")


# Elasticsearch에 데이터를 전송하는 함수
def send_to_elasticsearch():
    es = Elasticsearch("http://localhost:9200")

    # 인덱스 생성 (nori 분석기 포함)
    create_index_with_nori_analyzer()

    index_name = "stock_tickers"
    stock_data = get_all_stock_tickers()

    for ticker, company in stock_data.items():
        document = {
            "ticker": ticker,
            "company": company
        }
        response = es.index(index=index_name, body=document)
        print(response)

    print("Data sent to Elasticsearch")

from apscheduler.schedulers.background import BackgroundScheduler

# 스케줄러 생성
scheduler = BackgroundScheduler()

# send_to_elasticsearch 함수를 매일 9시에 실행하도록 설정
scheduler.add_job(
    func=send_to_elasticsearch,
    trigger="cron",
    hour=9,
    minute=0,
    id="send_to_elasticsearch_daily"
)

# 스케줄러 시작
scheduler.start()

import requests
import json

# Elasticsearch 서버 URL 및 인덱스 이름
es_url = "http://localhost:9200"
index_name = "stock_tickers"

def get_stock_from_elasticsearch(query=None):
    if query:
        # Match 쿼리 및 Wildcard 쿼리를 사용하여 부분 일치 검색
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "company": {
                                    "query": query,
                                    "analyzer": "nori_analyzer",  # nori_analyzer를 사용하여 분석합니다.
                                    "fuzziness": "AUTO"  # 자동으로 적절한 퍼지 값을 적용합니다.
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "company": {
                                    "value": f"*{query}*",
                                    "boost": 2.0  # Wildcard 쿼리의 중요도를 설정합니다.
                                }
                            }
                        },
                        {
                            "prefix": {
                                "company": {
                                    "value": query,
                                    "boost": 1.5  # Prefix 쿼리의 중요도를 설정합니다.
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1  # should 쿼리에서 하나 이상의 조건이 맞아야 합니다.
                }
            },
            "size": 1000
        }
    else:
        # 전체 검색 쿼리
        search_body = {
            "query": {
                "match_all": {}
            },
            "size": 1000
        }

    # Elasticsearch 쿼리 URL 생성
    query_url = f"{es_url}/{index_name}/_search"

    # GET 요청을 보내기
    response = requests.get(query_url, headers={"Content-Type": "application/json"}, data=json.dumps(search_body))

    # 응답 처리
    if response.status_code == 200:
        response_json = response.json()
        hits = response_json.get("hits", {}).get("hits", [])
        result = [{"company": hit["_source"]["company"]} for hit in hits]
        return result
    else:
        print("검색 실패:", response.status_code, response.text)
        return None