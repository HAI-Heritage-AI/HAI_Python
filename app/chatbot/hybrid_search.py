# hybrid_search.py
from eunjeon import Mecab
import faiss
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import pandas as pd

# Mecab 형태소 분석기 초기화
mecab = Mecab()

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

# 3. Mecab 기반 토큰화 (허용된 품사만 남기고 나머지는 제외)
allowed_pos_tags = ['NNP', 'NNG', 'NP', 'VV', 'VA', 'VCP', 'VCN', 'VSV', 'MAG', 'MAJ']

def tokenize_with_mecab(text):
    tokens = mecab.pos(text)  # 형태소 분석 후 품사 태깅
    filtered_tokens = [word for word, pos in tokens if pos in allowed_pos_tags]  # 허용된 품사만 남기기
    return filtered_tokens

print("Mecab으로 문서 토큰화 중...")
tokenized_documents = [tokenize_with_mecab(doc) for doc in tqdm(documents, desc="Tokenizing Content")]

# 4. BM25 인덱스 생성
print("BM25 인덱스 생성 중...")
bm25 = BM25Okapi(tokenized_documents)

# 5. 기존 FAISS 인덱스 로드
print("기존 FAISS 인덱스 로드 중...")
faiss_index = faiss.read_index(faiss_index_path)

# 6. 정규화 함수들
def min_max_normalize(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)

def z_score_normalize(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    if std == 0:
        return np.zeros_like(scores)
    return (scores - mean) / std

def max_normalize(scores):
    max_score = np.max(scores)
    if max_score == 0:
        return np.zeros_like(scores)
    return scores / max_score

# 7. 키워드 서치 구현
def keyword_search(query, k=3):
    """
    Mecab으로 토큰화된 쿼리를 기반으로 BM25를 사용한 키워드 서치.
    """
    query_tokens = tokenize_with_mecab(query)  # Mecab으로 쿼리 토큰화
    bm25_scores = bm25.get_scores(query_tokens)
    sorted_indices = np.argsort(-bm25_scores)[:k]
    results = [(documents[i], metadata[i], bm25_scores[i]) for i in sorted_indices]
    return results

# 8. 시멘틱 서치 구현
def semantic_search(query, k=3):
    """
    FAISS를 사용한 시멘틱 서치.
    """
    query_embedding = model.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, k)
    results = [(documents[i], metadata[i], faiss_scores[0][j]) for j, i in enumerate(faiss_indices[0])]
    return results

# 9. 하이브리드 서치 구현
def hybrid_search(query, k=3, alpha=0.5, normalization_method="min_max"):
    """
    다양한 정규화를 선택하여 하이브리드 서치를 수행합니다.
    """
    query_tokens = tokenize_with_mecab(query)  # Mecab으로 쿼리 토큰화
    bm25_scores = bm25.get_scores(query_tokens)

    query_embedding = model.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, len(documents))
    faiss_scores = faiss_scores[0]

    # 점수 정규화
    if normalization_method == "min_max":
        bm25_scores = min_max_normalize(bm25_scores)
        faiss_scores = min_max_normalize(faiss_scores)
    elif normalization_method == "z_score":
        bm25_scores = z_score_normalize(bm25_scores)
        faiss_scores = z_score_normalize(faiss_scores)
    elif normalization_method == "max":
        bm25_scores = max_normalize(bm25_scores)
        faiss_scores = max_normalize(faiss_scores)
    else:
        raise ValueError("지원하지 않는 정규화 방법입니다. 'min_max', 'z_score', 'max' 중 선택하세요.")

    final_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores
    sorted_indices = np.argsort(-final_scores)[:k]

    results = [(documents[i], metadata[i], final_scores[i]) for i in sorted_indices]
    return results

# 10. 가중치와 정규화 방식별 하이브리드 서치 실행
def run_hybrid_search_by_alpha(query, k=3):
    """
    각 가중치(alpha)와 정규화 방식별로 하이브리드 서치를 실행합니다.
    """
    normalization_methods = ["min_max", "z_score", "max"]
    alphas = [round(i * 0.1, 1) for i in range(1, 10)]  # 0.1 ~ 0.9

    results = []
    for normalization in normalization_methods:
        for alpha in alphas:
            search_results = hybrid_search(query, k=k, alpha=alpha, normalization_method=normalization)
            for rank, (doc, meta, score) in enumerate(search_results, 1):
                results.append({
                    "정규화 방식": normalization,
                    "가중치 (alpha)": alpha,
                    "순위": rank,
                    "점수": score,
                    "내용": doc[:50],  # 문서 내용의 일부만 표시
                    "국가유산명_국문": meta.get("국가유산명_국문", ""),
                    "시대": meta.get("시대", ""),
                    "소재지상세": meta.get("소재지상세", "")
                })

    return pd.DataFrame(results)

# 실행: 사용자 질의와 결과 계산
query = "한국에서 가장 오래된 목조 건축물"
top_k = 3

# 모델 로드
print("SentenceTransformer 모델 로드 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 키워드 서치 결과
print("\n[키워드 서치 결과] (Mecab 기반)")
keyword_results = keyword_search(query, k=top_k)
for rank, (doc, meta, score) in enumerate(keyword_results, 1):
    print(f"{rank}. 내용: {doc[:50]}... | 점수: {score:.4f}")

# 시멘틱 서치 결과
print("\n[시멘틱 서치 결과]")
semantic_results = semantic_search(query, k=top_k)
for rank, (doc, meta, score) in enumerate(semantic_results, 1):
    print(f"{rank}. 내용: {doc[:50]}... | 점수: {score:.4f}")

# 하이브리드 서치 결과 계산
print("\n하이브리드 서치 결과 계산 중...")
df_hybrid_results = run_hybrid_search_by_alpha(query, k=top_k)

# 결과 출력
print("\n하이브리드 서치 결과 DataFrame: (상위 20개)")
print(df_hybrid_results.head(20))  # 상위 20개만 출력

# 결과 저장
output_file = "./hybrid_search_detailed_results.csv"
df_hybrid_results.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n결과가 '{output_file}'에 저장되었습니다.")
