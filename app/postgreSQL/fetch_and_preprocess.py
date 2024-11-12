import os
import psycopg2
import re
import kss

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

    # 전처리된 결과 출력
    print(f"ccbaAsno: {original_id}")
    print("전처리된 텍스트:")
    for sentence in processed_sentences:
        print(sentence)
else:
    print("데이터를 가져오지 못했습니다.")
