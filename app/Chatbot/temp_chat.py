import os
import faiss
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer
import pickle

# 1. 현재 파일의 디렉토리를 기준으로 상대 경로 설정
base_dir = os.path.dirname(os.path.realpath(__file__))  # 현재 파일의 경로를 가져옵니다.
index_file = os.path.join(base_dir, "../FAISS/faiss_index_1000.bin")  # 상위 폴더의 FAISS 디렉토리 기준 상대 경로
metadata_file = os.path.join(base_dir, "../FAISS/faiss_metadata_1000.pkl")

# 2. FAISS 인덱스 불러오기
try:
    index = faiss.read_index(index_file)
    print(f"FAISS 인덱스 '{index_file}'을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"FAISS 인덱스를 불러오는 데 실패했습니다: {e}")
    exit()

# 3. 메타데이터 불러오기
try:
    with open(metadata_file, "rb") as f:
        original_ids = pickle.load(f)
    print(f"메타데이터 '{metadata_file}'을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"메타데이터를 불러오는 데 실패했습니다: {e}")
    exit()

# 4. 모델 로드
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # 토큰 개수를 세기 위해 GPT-2 토크나이저 사용

# 5. 사용자의 임시 질문을 입력받아 임베딩
user_question = "서울에있는 도자기 문화재를 찾아줘"  # 사용자의 임시 질문
tokens = tokenizer.tokenize(user_question)
token_count = len(tokens)
print(f"사용자 질문의 토큰 수: {token_count}")

# 질문이 512 토큰 이상이면 슬라이딩 윈도우 적용
max_tokens = 512
stride = 256
start_idx = 0

query_embeddings = []

while start_idx < token_count:
    end_idx = min(start_idx + max_tokens, token_count)
    window_tokens = tokens[start_idx:end_idx]
    window_text = tokenizer.convert_tokens_to_string(window_tokens)
    embedding = model.encode(window_text)
    query_embeddings.append(embedding)
    start_idx += stride

# 만약 슬라이딩 윈도우를 통해 여러 개로 나눠졌다면, 평균 벡터를 사용
if len(query_embeddings) > 1:
    query_vector = np.mean(query_embeddings, axis=0).astype('float32').reshape(1, -1)
else:
    query_vector = np.array(query_embeddings[0]).astype('float32').reshape(1, -1)

# 6. FAISS 인덱스에서 유사한 벡터 검색
D, I = index.search(query_vector, k=5)  # 가장 유사한 5개 검색

print(f"가장 유사한 벡터들의 인덱스: {I}")
print(f"각 유사한 벡터와의 거리: {D}")

# 7. 검색 결과와 원본 ID 매핑 및 PostgreSQL에서 원본 데이터 가져오기
try:
    # PostgreSQL 연결 설정
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()

    for idx in I[0]:
        original_id_segment = original_ids[idx]
        # 원본 데이터의 ID 추출 (예: "2600000_segment_5"에서 "2600000" 추출)
        original_id = original_id_segment.split('_')[0]

        # PostgreSQL에서 해당 ID의 데이터를 가져오기
        cursor.execute("SELECT * FROM national_heritage WHERE ccbaAsno = %s", (original_id,))
        row = cursor.fetchone()

        if row:
            print(f"원본 데이터 ID {original_id}의 내용: {row}")
        else:
            print(f"원본 데이터 ID {original_id}을(를) 찾을 수 없습니다.")

    conn.close()
except Exception as e:
    print(f"PostgreSQL 연결 또는 데이터 조회에 실패했습니다: {e}")
    exit()
