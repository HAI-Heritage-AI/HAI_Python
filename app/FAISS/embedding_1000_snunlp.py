import os
import psycopg2
import re
import kss
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.preprocessing import normalize

# 1. PostgreSQL에서 1000개의 데이터 가져오기
try:
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT ccbaAsno, content FROM national_heritage LIMIT 1000")  # 1000개의 데이터 가져오기
    rows = cursor.fetchall()
    conn.close()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 2. 모델 및 토크나이저 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')  # 한국어 SBERT 모델
tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

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

    i = 0
    while i < len(processed_sentences):
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

        # stride 만큼 이동
        i += stride - context_overlap

# FAISS 인덱스 저장 경로 설정
base_dir = os.path.dirname(os.path.realpath(__file__))
metadata_file = os.path.join(base_dir, "../FAISS/snunlp_metadata_1000.pkl")
with open(metadata_file, "wb") as f:
    pickle.dump(metadata, f)
print(f"메타데이터를 '{metadata_file}' 파일로 저장했습니다.")

# 4. FAISS 인덱스 생성 및 저장 (내적 방식)
embedding_dim = len(embeddings[0])
embeddings_np = np.array(embeddings).astype('float32')

# 내적 방식
index_dot_product = faiss.IndexFlatIP(embedding_dim)
index_dot_product.add(embeddings_np)
faiss_index_file_dot = os.path.join(base_dir, "../FAISS/snunlp_dotProduct_index_1000.bin")
faiss.write_index(index_dot_product, faiss_index_file_dot)
print(f"내적 기반 FAISS 인덱스를 '{faiss_index_file_dot}' 파일로 저장했습니다.")

# 코사인 유사도 방식 (벡터 정규화 후 내적 방식 활용)
index_cosine = faiss.IndexFlatIP(embedding_dim)
embeddings_cosine = normalize(embeddings_np, norm='l2')  # 정규화하여 코사인 유사도 계산
index_cosine.add(embeddings_cosine)
faiss_index_file_cosine = os.path.join(base_dir, "../FAISS/snunlp_cosine_index_1000.bin")
faiss.write_index(index_cosine, faiss_index_file_cosine)
print(f"코사인 유사도 기반 FAISS 인덱스를 '{faiss_index_file_cosine}' 파일로 저장했습니다.")

# 유클리드 거리 방식
index_euclidean = faiss.IndexFlatL2(embedding_dim)
index_euclidean.add(embeddings_np)
faiss_index_file_euclidean = os.path.join(base_dir, "../FAISS/snunlp_euclidean_index_1000.bin")
faiss.write_index(index_euclidean, faiss_index_file_euclidean)
print(f"유클리드 거리 기반 FAISS 인덱스를 '{faiss_index_file_euclidean}' 파일로 저장했습니다.")
