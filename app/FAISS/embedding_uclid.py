import os 
import psycopg2
import re
import kss
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# 1. PostgreSQL에서 1000개의 데이터 가져오기
try:
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT ccbaAsno, content FROM national_heritage")  # 1000개의 데이터 가져오기
    rows = cursor.fetchall()
    conn.close()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 2. 모델 및 토크나이저 로드
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

embeddings = []
metadata = []

# 3. 데이터 전처리 및 슬라이딩 윈도우 방식 적용
max_tokens = 512
stride = 256
context_overlap = 2  # 문맥 보존을 위해 중첩할 문장 수

for row in rows:
    original_id, text = row

    # 문장 단위로 분리 (KSS 사용)
    sentences = kss.split_sentences(text)

    # 각 문장에서 구두점 제거
    processed_sentences = [re.sub(r'[\.,!?]', '', sentence) for sentence in sentences]

    current_text = ""
    segment_id = 0

    for i in range(0, len(processed_sentences), stride):
        # 현재 슬라이딩 윈도우 범위의 문장들
        current_sentences = processed_sentences[i:i + stride]

        # 이전 문장 일부를 포함해 문맥 보존
        if i > 0:
            overlap_sentences = processed_sentences[max(0, i - context_overlap):i]
            current_sentences = overlap_sentences + current_sentences

        # 문장들을 하나의 텍스트로 결합
        current_text = " ".join(current_sentences)

        # 임베딩 생성
        embedding = model.encode(current_text)
        embeddings.append(embedding)

        metadata_entry = {
            "original_id": original_id,
            "segment_id": segment_id,
            "text_segment": current_text
        }
        metadata.append(metadata_entry)
        segment_id += 1

# 4. 유클리디안 거리(L2 거리)방식을 사용해서 FAISS 인덱스 생성 및 데이터 추가
embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)

# FAISS는 numpy 배열로 데이터를 다루므로 리스트를 numpy로 변환
embeddings_np = np.array(embeddings).astype('float32')

# 인덱스에 벡터 추가
index.add(embeddings_np)

# 5. FAISS 인덱스 및 메타데이터 파일 저장
base_dir = os.path.dirname(os.path.realpath(__file__))
faiss_index_file = os.path.join(base_dir, "../FAISS/faiss_index_1000_uclid.bin")
faiss.write_index(index, faiss_index_file)
print(f"FAISS 인덱스를 '{faiss_index_file}' 파일로 저장했습니다.")

metadata_file = os.path.join(base_dir, "../FAISS/faiss_metadata_1000_uclid.pkl")
with open(metadata_file, "wb") as f:
    pickle.dump(metadata, f)
print(f"메타데이터를 '{metadata_file}' 파일로 저장했습니다.")
