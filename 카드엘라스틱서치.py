import logging
from elasticsearch import Elasticsearch, helpers
import json

logging.basicConfig(level=logging.INFO)

# Elasticsearch 클라이언트 생성
es = Elasticsearch("http://localhost:9200")

def index_cards():
    # Elasticsearch 인덱스 이름 설정
    index_name = "card"

    # 인덱스 설정 및 매핑 정의
    index_body = {
        "settings": {
            "analysis": {
                "tokenizer": {
                    "nori_user_dict_tokenizer": {
                        "type": "nori_tokenizer",
                        "decompound_mode": "mixed",
                        "discard_punctuation": "false"
                    }
                },
                "filter": {
                    "korean_stop": {
                        "type": "stop",
                        "stopwords_path": "analysis/stopwords/korean_stopwords.txt"
                    },
                    "nori_filter": {
                      "type": "nori_part_of_speech",
                      "stoptags": [
                        "E", "IC", "J", "MAG", "MAJ", "MM", "SP", "SSC", "SSO", "SC", "SE", "XPN", "XSA", "XSN", "XSV",
                        "UNA", "NA", "VSV", "NP"
                      ]
                    },
                    "ngram_filter": {
                        "type": "ngram",
                        "min_gram": 2,
                        "max_gram": 3
                    },
                    "english_ngram_filter": {
                        "type": "ngram",
                        "min_gram": 2,
                        "max_gram": 3
                    },
                },
                "analyzer": {
                    "nori_analyzer_with_stopwords": {
                        "type": "custom",
                        "tokenizer": "nori_user_dict_tokenizer",
                        "filter": ["nori_readingform", "korean_stop", "nori_filter", "trim"]
                    },
                    "nori_ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_user_dict_tokenizer",
                        "filter": ["nori_readingform", "ngram_filter", "trim"]
                    },
                    "english_ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "english_ngram_filter", "trim"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "ccname": {
                    "type": "text",
                    "analyzer": "nori_analyzer_with_stopwords",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "nori_ngram_analyzer"
                        }
                    }
                },
                "cname": {
                    "type": "text",
                    "analyzer": "nori_analyzer_with_stopwords",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "nori_ngram_analyzer"
                        }
                    }
                },
                "cimage": {"type": "text"},  # 이미지 URL은 텍스트로 처리
                "performance": {"type": "text"},  # 성과는 텍스트로 처리
                "benefits": {
                    "type": "text",
                    "analyzer": "nori_analyzer_with_stopwords",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "analyzer": "nori_ngram_analyzer"
                        }
                    }
                },
                "annualfee": {"type": "text"},
            }
        }
    }

    try:
        # 인덱스가 존재하지 않으면 정의된 매핑과 함께 인덱스 생성
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=index_body)
            logging.info(f"Index '{index_name}' created.")

        # JSON 파일에서 카드 데이터를 가져옴
        with open('카드.json', 'r', encoding='utf-8') as file:
            card_list = json.load(file)

        # 벌크 인덱싱 준비
        actions = [
            {
                "_index": index_name,
                "_source": card
            }
            for card in card_list
        ]

        # helpers.bulk() 를 사용해 벌크 인덱싱 수행
        helpers.bulk(es, actions)

        # 인덱스 새로고침 (색인을 즉시 사용 가능하게 함)
        es.indices.refresh(index=index_name)

        # Elasticsearch에서 모든 카드를 검색하여 총 카드 수 출력
        response = es.search(index=index_name, body={"query": {"match_all": {}}})
        print(f"Total cards indexed: {response['hits']['total']['value']}")

    except Exception as e:
        logging.error("An error occurred during indexing:", exc_info=True)

if __name__ == "__main__":
    index_cards()

import requests
import json

# Elasticsearch 서버 URL 및 인덱스 이름
es_url = "http://localhost:9200"
index_name = "card"

def get_card_from_elasticsearch(query=None):
    if query:
        # Match 쿼리 및 Wildcard 쿼리를 사용하여 부분 일치 검색
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["benefits", "cname", "ccname"],
                                "analyzer": "nori_analyzer_with_stopwords",
                                "fuzziness": "AUTO"
                            }
                        },
                        {
                            "wildcard": {
                                "benefits": {
                                    "value": f"*{query}*",
                                    "boost": 2.0
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "cname": {
                                    "value": f"*{query}*",
                                    "boost": 2.0
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "ccname": {
                                    "value": f"*{query}*",
                                    "boost": 2.0
                                }
                            }
                        },
                        {
                            "prefix": {
                                "benefits": {
                                    "value": query,
                                    "boost": 1.5
                                }
                            }
                        },
                        {
                            "prefix": {
                                "cname": {
                                    "value": query,
                                    "boost": 1.5
                                }
                            }
                        },
                        {
                            "prefix": {
                                "ccname": {
                                    "value": query,
                                    "boost": 1.5
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
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
            "size": 400
        }

    # Elasticsearch 쿼리 URL 생성
    query_url = f"{es_url}/{index_name}/_search"

    # GET 요청을 보내기
    response = requests.get(query_url, headers={"Content-Type": "application/json"}, data=json.dumps(search_body))

    # 응답 처리
    if response.status_code == 200:
        response_json = response.json()
        hits = response_json.get("hits", {}).get("hits", [])
        result = [{"ccname": hit["_source"]["ccname"],
                   "cname": hit["_source"]["cname"],
                   "cimage": hit["_source"]["cimage"],
                   "performance": hit["_source"]["performance"],
                   "benefits": hit["_source"]["benefits"],
                   "annualfee": hit["_source"]["annualfee"]} for hit in hits]
        return result
    else:
        print("검색 실패:", response.status_code, response.text)
        return None


def calculate_match_score(benefits, categories):
    category_weights = {category: 5 - index for index, category in enumerate(categories)}
    score = 0
    for category in categories:
        if category in benefits:
            score += category_weights.get(category, 0)  # 가중치를 점수에 추가
    return score

def search_top_cards(categories):
    # 초기 검색 쿼리
    query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "benefits": {
                                "query": category,
                                "boost": 1.0  # 기본 boost 값, 필요에 따라 조정
                            }
                        }
                    } for category in categories
                ]
            }
        }
    }

    # 쿼리 실행
    response = es.search(index="card", body=query, size=400)

    # 카드 정보와 일치도 계산
    card_scores = []
    for hit in response['hits']['hits']:
        card = hit['_source']
        benefits = card.get('benefits', '')
        score = calculate_match_score(benefits, categories)
        card_scores.append((card, score))

    return card_scores
