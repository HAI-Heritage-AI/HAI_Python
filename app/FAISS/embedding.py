import os
import psycopg2
import re
import kss
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# 1. PostgreSQL에서 모든 데이터 가져오기
try:
    conn = psycopg2.connect(
        host="localhost",
        database="heritage_db",
        user="postgres",
        password="iam@123"
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, ccbaMnm1, ccbaMnm2, ccmaName, ccbaCtcdNm, ccsiName, ccceName, content, ccbaLcad
        FROM national_heritage
    """)  
    rows = cursor.fetchall()
    conn.close()
except Exception as e:
    print("PostgreSQL 연결 실패:", e)
    exit()

# 2. 모델 및 토크나이저 로드
model = SentenceTransformer(' /ko-sroberta-multitask')
embeddings = []
metadata = []

# 3. 데이터 전처리 및 슬라이딩 윈도우 방식 적용
max_tokens = 512
stride = 256
context_overlap = 2  # 문맥 보존을 위해 중첩할 문장 수

for row in tqdm(rows, desc="Processing rows"):
    primary_id, ccbaMnm1, ccbaMnm2, ccmaName, ccbaCtcdNm, ccsiName, ccceName, content, ccbaLcad = row

    # 모든 정보를 하나의 텍스트로 결합
    combined_text = f"국가유산명_국문: {ccbaMnm1}, 국가유산명_한자: {ccbaMnm2}, 국가유산종목: {ccmaName}, 시도명: {ccbaCtcdNm}, 시군구명: {ccsiName}, 시대: {ccceName}, 내용: {content}, 소재지상세: {ccbaLcad}"

    # 문장 단위로 분리 (KSS 사용)
    sentences = kss.split_sentences(combined_text)

    # 각 문장에서 구두점 제거
    processed_sentences = [re.sub(r'[\.,!?]', '', sentence) for sentence in sentences]

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
            "primary_id": primary_id,
            "국가유산명_국문": ccbaMnm1,
            "국가유산명_한자": ccbaMnm2,
            "국가유산종목": ccmaName,
            "시도명": ccbaCtcdNm,
            "시군구명": ccsiName,
            "시대": ccceName,
            "내용": current_text,
            "소재지상세": ccbaLcad,
            "segment_id": segment_id  # 슬라이딩 윈도우 구간
        }
        metadata.append(metadata_entry)
        segment_id += 1

        # stride 만큼 이동
        i += stride - context_overlap

# FAISS 인덱스 및 메타데이터 저장 경로 설정
base_dir = os.path.dirname(os.path.realpath(__file__))
metadata_file = os.path.join(base_dir, "../FAISS/Metadata/jhgan_metadata.pkl")
with open(metadata_file, "wb") as f:
    pickle.dump(metadata, f)
print(f"메타데이터를 '{metadata_file}' 파일로 저장했습니다.")

# 4. FAISS 인덱스 생성 및 저장 (내적 방식)
embedding_dim = len(embeddings[0])
embeddings_np = np.array(embeddings).astype('float32')

# 내적 방식
index_dot_product = faiss.IndexFlatIP(embedding_dim)
index_dot_product.add(embeddings_np)
jhgan_index_file_dot = os.path.join(base_dir, "../FAISS/Index/jhgan_dotProduct_index.bin")
faiss.write_index(index_dot_product, jhgan_index_file_dot)
print(f"내적 기반 FAISS 인덱스를 '{jhgan_index_file_dot}' 파일로 저장했습니다.")

# 코사인 유사도 방식 (벡터 정규화 후 내적 방식 활용)
index_cosine = faiss.IndexFlatIP(embedding_dim)
embeddings_cosine = normalize(embeddings_np, norm='l2')  # 정규화하여 코사인 유사도 계산
index_cosine.add(embeddings_cosine)
jhgan_index_file_cosine = os.path.join(base_dir, "../FAISS/Index/jhgan_cosine_index.bin")
faiss.write_index(index_cosine, jhgan_index_file_cosine)
print(f"코사인 유사도 기반 FAISS 인덱스를 '{jhgan_index_file_cosine}' 파일로 저장했습니다.")

# 유클리드 거리 방식
index_euclidean = faiss.IndexFlatL2(embedding_dim)
for _ in tqdm(range(1), desc="Adding embeddings to Euclidean index"):
    index_euclidean.add(embeddings_np)
jhgan_index_file_euclidean = os.path.join(base_dir, "../FAISS/Index/jhgan_euclidean_index.bin")
faiss.write_index(index_euclidean, jhgan_index_file_euclidean)
print(f"유클리드 거리 기반 FAISS 인덱스를 '{jhgan_index_file_euclidean}' 파일로 저장했습니다.")
