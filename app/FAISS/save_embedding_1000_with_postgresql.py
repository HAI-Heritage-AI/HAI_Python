import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import psycopg2
import pickle
from transformers import GPT2Tokenizer

# 1. PostgreSQL에서 데이터 가져오기
try:
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT ccbaAsno, content FROM national_heritage LIMIT 1000")  # 모든 데이터 가져오기
    rows = cursor.fetchall()
    conn.close()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 2. 모델 로드 및 데이터 임베딩
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # 토큰 개수를 세기 위해 GPT-2 토크나이저 사용

embeddings = []
metadata = []

for row in rows:
    original_id, text = row
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    print(f"데이터 ID {original_id}의 토큰 수: {token_count}")

    # 슬라이딩 윈도우 적용 (최대 512 토큰씩)
    max_tokens = 512
    stride = 256

    start_idx = 0
    segment_id = 0
    while start_idx < token_count:
        end_idx = min(start_idx + max_tokens, token_count)
        window_tokens = tokens[start_idx:end_idx]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        embedding = model.encode(window_text)

        embeddings.append(embedding)

        # 각 윈도우 구간에도 세그먼트 ID 추가하여 메타데이터 생성
        metadata_entry = {
            "original_id": original_id,      # 원본 데이터 ID
            "segment_id": segment_id,        # 세그먼트 번호
            "text_segment": window_text      # 텍스트 세그먼트
        }
        metadata.append(metadata_entry)

        start_idx += stride
        segment_id += 1

# 3. FAISS 인덱스 생성 및 데이터 추가
embedding_dim = len(embeddings[0])
index = faiss.IndexFlatL2(embedding_dim)

# FAISS는 numpy 배열로 데이터를 다루므로 리스트를 numpy로 변환
embeddings_np = np.array(embeddings).astype('float32')

# 인덱스에 벡터 추가
index.add(embeddings_np)

# 4. FAISS 인덱스 및 메타데이터 파일 저장
base_dir = os.path.dirname(os.path.realpath(__file__))
faiss_index_file = os.path.join(base_dir, "../FAISS/faiss_index_1000_02.bin")
faiss.write_index(index, faiss_index_file)
print(f"FAISS 인덱스를 '{faiss_index_file}' 파일로 저장했습니다.")

metadata_file = os.path.join(base_dir, "../FAISS/faiss_metadata_1000_02.pkl")
with open(metadata_file, "wb") as f:
    pickle.dump(metadata, f)
print(f"메타데이터를 '{metadata_file}' 파일로 저장했습니다.")
