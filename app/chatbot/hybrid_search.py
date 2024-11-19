import faiss
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# 1. 메타데이터 및 FAISS 인덱스 경로
metadata_path = "../FAISS/Metadata/jhgan_metadata.pkl"
faiss_index_path = "../FAISS/Index/jhgan_cosine_index.bin"

if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")

if not os.path.exists(faiss_index_path):
    raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {faiss_index_path}")

# 2. 메타데이터 로드
print("메타데이터 로드 중...")
with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

# '내용' 값만 추출
print("메타데이터의 내용 필드 추출 중...")
documents = [entry["내용"] for entry in metadata]

# 3. 공백 기반 토큰화
print("내용을 공백 기준으로 토큰화 중...")
tokenized_documents = [doc.split(" ") for doc in tqdm(documents, desc="Tokenizing Content")]

# 4. BM25 인덱스 생성
print("BM25 인덱스 생성 중...")
bm25 = BM25Okapi(tokenized_documents)

# 5. 기존 FAISS 인덱스 로드
print("기존 FAISS 인덱스 로드 중...")
faiss_index = faiss.read_index(faiss_index_path)

# 6. 정규화 함수들
def min_max_normalize(scores):
    """
    Min-Max 정규화를 수행합니다.
    :param scores: 정규화할 점수 배열
    :return: Min-Max 정규화된 점수 배열
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)

def z_score_normalize(scores):
    """
    z-score 정규화를 수행합니다.
    :param scores: 정규화할 점수 배열
    :return: z-score 정규화된 점수 배열
    """
    mean = np.mean(scores)
    std = np.std(scores)
    if std == 0:
        return np.zeros_like(scores)
    return (scores - mean) / std

def max_normalize(scores):
    """
    Max 정규화를 수행합니다.
    :param scores: 정규화할 점수 배열
    :return: Max 정규화된 점수 배열
    """
    max_score = np.max(scores)
    if max_score == 0:
        return np.zeros_like(scores)
    return scores / max_score

# 7. 하이브리드 서치 구현
def hybrid_search(query, k=3, alpha=0.5, normalization_method="min_max"):
    """
    다양한 정규화를 선택하여 하이브리드 서치를 수행합니다.
    :param query: 검색 질의
    :param k: 반환할 결과 개수
    :param alpha: BM25와 FAISS 점수의 가중치 (0 ~ 1)
    :param normalization_method: "min_max", "z_score", "max" 중 선택
    :return: 정렬된 검색 결과 리스트
    """
    # (1) BM25 점수 계산
    print("BM25 점수 계산 중...")
    query_tokens = query.split(" ")
    bm25_scores = bm25.get_scores(query_tokens)

    # (2) FAISS 검색 수행
    print("FAISS 검색 수행 중...")
    query_embedding = model.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, len(documents))
    faiss_scores = faiss_scores[0]

    # (3) 점수 정규화
    if normalization_method == "min_max":
        print("Min-Max 정규화 적용 중...")
        bm25_scores = min_max_normalize(bm25_scores)
        faiss_scores = min_max_normalize(faiss_scores)
    elif normalization_method == "z_score":
        print("z-score 정규화 적용 중...")
        bm25_scores = z_score_normalize(bm25_scores)
        faiss_scores = z_score_normalize(faiss_scores)
    elif normalization_method == "max":
        print("Max 정규화 적용 중...")
        bm25_scores = max_normalize(bm25_scores)
        faiss_scores = max_normalize(faiss_scores)
    else:
        raise ValueError("지원하지 않는 정규화 방법입니다. 'min_max', 'z_score', 'max' 중 선택하세요.")

    # (4) 점수 결합
    print("BM25와 FAISS 점수 결합 중...")
    final_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores

    # (5) 결과 정렬
    print("검색 결과 정렬 중...")
    sorted_indices = np.argsort(-final_scores)[:k]
    results = [(documents[i], metadata[i], final_scores[i]) for i in sorted_indices]

    return results

# 8. 테스트: 사용자 질의 처리
query = "한국에서 가장 오래된 목조 건축물"
print("하이브리드 서치 수행 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# Min-Max 정규화
print("\n[Min-Max 정규화 결과]")
results_min_max = hybrid_search(query, k=3, alpha=0.5, normalization_method="min_max")

# z-score 정규화
print("\n[z-score 정규화 결과]")
results_z_score = hybrid_search(query, k=3, alpha=0.5, normalization_method="z_score")

# Max 정규화
print("\n[Max 정규화 결과]")
results_max = hybrid_search(query, k=3, alpha=0.5, normalization_method="max")

# 9. 결과 출력
print("\n[비교 결과]")

print("\nMin-Max 정규화:")
for i, (doc, meta, score) in enumerate(results_min_max):
    print(f"{i+1}. 내용: {doc} | 점수: {score:.4f}")

print("\nz-score 정규화:")
for i, (doc, meta, score) in enumerate(results_z_score):
    print(f"{i+1}. 내용: {doc} | 점수: {score:.4f}")

print("\nMax 정규화:")
for i, (doc, meta, score) in enumerate(results_max):
    print(f"{i+1}. 내용: {doc} | 점수: {score:.4f}")
