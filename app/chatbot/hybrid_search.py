import faiss
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # 진행 표시를 위한 라이브러리
import os

# 1. 메타데이터 및 FAISS 인덱스 파일 경로
metadata_path = "../FAISS/Metadata/jhgan_metadata.pkl"  # 메타데이터 파일 경로
faiss_index_path = "../FAISS/Index/jhgan_cosine_index.bin"  # 기존 FAISS 인덱스 경로

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")

if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {faiss_index_path}")

# 2. 메타데이터 로드
print("메타데이터 로드 중...")
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# text_segment 값 추출
documents = [entry["text_segment"] for entry in metadata]

# 3. 공백 기반 토큰화
print("문서를 공백 기준으로 토큰화 중...")
tokenized_documents = [doc.split(" ") for doc in tqdm(documents, desc="Tokenizing Documents")]

# 4. BM25 인덱스 생성
print("BM25 인덱스 생성 중...")
bm25 = BM25Okapi(tokenized_documents)

# 5. 기존 FAISS 인덱스 로드
print("기존 FAISS 인덱스 로드 중...")
faiss_index = faiss.read_index(faiss_index_path)

# 6. 하이브리드 서치 구현
def hybrid_search(query, k=3, alpha=0.5):
    """
    하이브리드 서치를 수행하는 함수
    :param query: 검색 질의
    :param k: 반환할 결과 개수
    :param alpha: BM25와 FAISS 점수의 가중치 (0 ~ 1)
    :return: 정렬된 검색 결과 리스트
    """
    # (1) BM25 점수 계산
    print("BM25 점수 계산 중...")
    query_tokens = query.split(" ")  # 공백 기반 토큰화
    bm25_scores = []
    for token in tqdm(query_tokens, desc="Calculating BM25 Scores"):
        bm25_scores.append(bm25.get_scores([token]))  # BM25 점수 배열

    bm25_scores = np.max(bm25_scores, axis=0)  # 최대 점수로 통합

    # (2) FAISS 검색 수행
    print("FAISS 검색 수행 중...")
    query_embedding = model.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, len(documents))

    # (3) 점수 결합
    print("BM25와 FAISS 점수 결합 중...")
    faiss_scores = faiss_scores[0]  # FAISS의 유사도 점수 배열
    final_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores

    # (4) 결과 정렬
    print("검색 결과 정렬 중...")
    sorted_indices = np.argsort(-final_scores)[:k]  # 점수 내림차순 정렬
    results = [(documents[i], final_scores[i]) for i in sorted_indices]

    return results

# 7. 테스트: 사용자 질의 처리
query = "부석사에 대해서 알려주세요"
print("하이브리드 서치 수행 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
results = hybrid_search(query, k=3, alpha=0.7)  # BM25에 더 가중치를 준 검색

# 8. 결과 출력
print("Top Results:")
for i, (doc, score) in enumerate(results):
    print(f"{i+1}. 문서: {doc} (점수: {score:.4f})")
