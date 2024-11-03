from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
import psycopg2
from transformers import GPT2Tokenizer

# 1. Milvus에 연결하기
connections.connect(host='localhost', port='19530')

# 2. 컬렉션 스키마 정의
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="postgresql_id", dtype=DataType.INT64)  # 원래 데이터를 식별할 수 있는 필드 추가
]
schema = CollectionSchema(fields, description="Cultural Heritage Embeddings")

# 3. 컬렉션 생성 (이미 생성된 경우 생략)
collection_name = "heritage_embeddings"
if collection_name not in Collection.list():
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# 4. PostgreSQL에서 데이터 가져오기
try:
    # PostgreSQL 연결 설정
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM your_table LIMIT 10")  # 테스트로 10개의 데이터만 가져오기

    rows = cursor.fetchall()
    conn.close()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 5. 모델 로드 및 데이터 임베딩
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # 토큰 개수를 세기 위해 GPT-2 토크나이저 사용

embeddings = []
postgresql_ids = []

for row in rows:
    original_id, text = row
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    print(f"데이터 ID {postgresql_id}의 토큰 수: {token_count}")

    # 슬라이딩 윈도우 적용 (최대 512 토큰씩)
    max_tokens = 512
    stride = 256  # 겹치는 부분을 고려한 슬라이딩 윈도우

    start_idx = 0
    while start_idx < token_count:
        end_idx = min(start_idx + max_tokens, token_count)
        window_tokens = tokens[start_idx:end_idx]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        embedding = model.encode(window_text)
        embeddings.append(embedding)
        postgresql_ids.append(original_id)  # 각 임베딩에 원래 데이터의 ID 추가
        start_idx += stride

# 6. Milvus에 데이터 삽입하기
# 컬렉션에 인서트하기 위해 임베딩을 리스트 형식으로 준비
collection.insert([embeddings, postgresql_ids])

print("임베딩 데이터를 Milvus에 성공적으로 저장했습니다.")
