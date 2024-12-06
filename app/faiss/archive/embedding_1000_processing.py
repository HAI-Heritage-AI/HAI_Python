import os
import psycopg2
import re
import kss
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. PostgreSQL에서 특정 데이터 가져오기
try:
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT ccbaAsno, content FROM national_heritage WHERE ccbaAsno = 340000 LIMIT 1")  # 특정 ID의 데이터만 가져오기
    row = cursor.fetchone()
    conn.close()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 2. 데이터 전처리
if row:
    original_id, text = row

    # 3. 전처리 단계 - 한국어 특성에 맞는 전처리 적용
    # 문장 단위로 분리 (KSS 사용)
    sentences = kss.split_sentences(text)

    # 각 문장에서 구두점 제거
    processed_sentences = [re.sub(r'[\.,!?]', '', sentence) for sentence in sentences]

    # 임베딩 모델 및 토크나이저 로드
    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 슬라이딩 윈도우 방식으로 문장을 추가하며 512 토큰을 넘지 않도록 유지
    max_tokens = 512
    current_tokens = 0
    current_text = ""
    embeddings = []

    for sentence in processed_sentences:
        # 현재 문장을 추가했을 때 토큰 수 확인
        tokens = tokenizer.tokenize(current_text + " " + sentence)
        token_count = len(tokens)

        if token_count <= max_tokens:
            # 토큰 수가 최대치를 넘지 않으면 문장을 추가
            current_text += " " + sentence
            current_tokens = token_count
        else:
            # 토큰 수가 최대치를 넘으면 현재 텍스트를 임베딩하고 초기화
            if current_text:
                embedding = model.encode(current_text.strip())
                embeddings.append(embedding)

            # 새로운 문장으로 초기화
            current_text = sentence
            current_tokens = len(tokenizer.tokenize(sentence))

    # 마지막 남은 텍스트도 임베딩
    if current_text:
        embedding = model.encode(current_text.strip())
        embeddings.append(embedding)

    # 전처리 및 임베딩된 결과 출력
    print(f"ccbaAsno: {original_id}")
    print("임베딩된 벡터의 수:", len(embeddings))
    for i, embedding in enumerate(embeddings):
        print(f"벡터 {i + 1}: {embedding[:10]}...")  # 벡터의 앞 10개 요소만 출력
else:
    print("데이터를 가져오지 못했습니다.")
