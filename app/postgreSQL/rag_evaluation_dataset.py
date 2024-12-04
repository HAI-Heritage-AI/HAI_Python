import os
import re
import kss
import csv
import psycopg2

# 1. PostgreSQL에서 데이터 가져오기
try:
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM national_heritage")  # 테이블에서 content와 고유 식별자를 가져오기
    rows = cursor.fetchall()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 2. 데이터 전처리 및 문장별 분리 후 저장
document_segments = []

for row in rows:
    id, text = row

    # 문장 단위로 분리 (KSS 사용)
    sentences = kss.split_sentences(text)

    # 각 문장에서 구두점 제거 및 저장할 데이터 구성
    for sentence_id, sentence in enumerate(sentences):
        processed_sentence = re.sub(r'[\.,!?]', '', sentence)
        document_segments.append((id, sentence_id, processed_sentence))

# 3. 전처리된 문장별 데이터를 파일로 저장
output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "national_heritage_sentences.csv")

try:
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "sentence_id", "sentence"])
        writer.writerows(document_segments)
    print(f"문장 데이터를 '{output_file}' 파일로 저장했습니다.")
except Exception as e:
    print("문장 데이터 파일 저장 실패:", e)
