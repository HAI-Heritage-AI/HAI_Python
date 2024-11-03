import psycopg2
from konlpy.tag import Okt
import re

# 1. PostgreSQL에서 데이터 가져오기
try:
    # PostgreSQL 연결 설정
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="your_password"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM national_heritage LIMIT 1")  # 데이터 하나 가져오기
    row = cursor.fetchone()
    conn.close()
    
    if row:
        text = row[0]
        print(f"원본 데이터: {text}")
    else:
        print("데이터를 가져오지 못했습니다.")
        exit()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 2. 한국어 텍스트 전처리
okt = Okt()

# 소문자 변환은 한국어에서는 불필요, 구두점 제거
text_cleaned = re.sub(r"[^\w\s]", "", text)  # 구두점 및 특수 문자 제거

# 형태소 분석 및 명사, 동사, 형용사 추출 (불용어 제거 대용)
tokens = okt.pos(text_cleaned, norm=True, stem=True)
filtered_tokens = [word for word, tag in tokens if tag in ['Noun', 'Verb', 'Adjective']]

# 전처리 결과 확인
print(f"전처리된 데이터: {' '.join(filtered_tokens)}")
