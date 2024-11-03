import faiss
import psycopg2
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer
import numpy as np

# 1. PostgreSQL에서 데이터 가져오기
try:
    # PostgreSQL 연결 설정
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT ccbaAsno, content FROM national_heritage LIMIT 10")  # 테스트로 10개의 데이터만 가져오기

    rows = cursor.fetchall()
    conn.close()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 2. 모델 로드 및 데이터 임베딩
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # 토큰 개수를 세기 위해 GPT-2 토크나이저 사용

embeddings = []
original_ids = []

for row in rows:
    original_id, text = row
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    print(f"데이터 ID {original_id}의 토큰 수: {token_count}")

    # 슬라이딩 윈도우 적용 (최대 512 토큰씩)
    max_tokens = 512
    stride = 256  # 겹치는 부분을 고려한 슬라이딩 윈도우

    start_idx = 0
    segment_id = 0
    while start_idx < token_count:
        end_idx = min(start_idx + max_tokens, token_count)
        window_tokens = tokens[start_idx:end_idx]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        embedding = model.encode(window_text)

        embeddings.append(embedding)
        # 각 윈도우 구간에도 세그먼트 ID 추가
        original_ids.append(f"{original_id}_segment_{segment_id}")
        
        start_idx += stride
        segment_id += 1

# 3. FAISS 인덱스 생성 및 데이터 추가
# 임베딩의 차원을 확인합니다.
embedding_dim = len(embeddings[0])

# FAISS 인덱스 생성 (IndexFlatL2를 사용하여 L2 거리로 유사도 검색)
index = faiss.IndexFlatL2(embedding_dim)

# FAISS는 numpy 배열로 데이터를 다루므로 리스트를 numpy로 변환
embeddings_np = np.array(embeddings).astype('float32')

# 인덱스에 벡터 추가
index.add(embeddings_np)

# 4. 검색 테스트
# 예시로 첫 번째 벡터를 사용하여 유사한 벡터 검색
query_vector = embeddings_np[0].reshape(1, -1)
D, I = index.search(query_vector, k=5)  # 가장 유사한 5개 검색

print(f"가장 유사한 벡터들의 인덱스: {I}")
print(f"각 유사한 벡터와의 거리: {D}")

# 검색 결과와 원본 ID 매핑
for idx in I[0]:
    print(f"원본 데이터 ID: {original_ids[idx]}")
