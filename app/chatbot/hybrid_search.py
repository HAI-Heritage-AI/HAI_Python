import matplotlib.pyplot as plt
from matplotlib import rc
import platform

# 한글 폰트 설정
if platform.system() == "Windows":
    rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # macOS
    rc("font", family="AppleGothic")
else:  # Linux (예: Ubuntu)
    rc("font", family="NanumGothic")

# 유니코드 마이너스(-) 처리
plt.rcParams["axes.unicode_minus"] = False

import faiss
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt

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
    query_tokens = query.split(" ")
    bm25_scores = bm25.get_scores(query_tokens)

    # (2) FAISS 검색 수행
    query_embedding = model.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_indices = faiss_index.search(query_embedding, len(documents))
    faiss_scores = faiss_scores[0]

    # (3) 점수 정규화
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

    # (4) 점수 결합
    final_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores

    # (5) 결과 정렬
    sorted_indices = np.argsort(-final_scores)[:k]
    results = [(documents[i], metadata[i], final_scores[i]) for i in sorted_indices]

    return results

# 8. 가중치와 정규화 방식별 검색 수행
def hybrid_search_for_weights(query, k=3):
    """
    각 정규화 방식과 다양한 가중치(alpha)에 대해 하이브리드 서치를 수행.
    :param query: 검색 질의
    :param k: 반환할 결과 개수
    :return: pandas DataFrame
    """
    results = []
    normalization_methods = ["min_max", "z_score", "max"]
    weights = [round(i * 0.1, 1) for i in range(1, 10)]  # alpha: 0.1 ~ 0.9

    for method in normalization_methods:
        for alpha in weights:
            search_results = hybrid_search(query, k=k, alpha=alpha, normalization_method=method)
            for rank, (doc, meta, score) in enumerate(search_results):
                results.append({
                    "정규화 방식": method,
                    "가중치 (alpha)": alpha,
                    "순위": rank + 1,
                    "내용": doc[:50],  # 내용의 일부만 표시
                    "점수": score,
                    "국가유산명_국문": meta.get("국가유산명_국문", ""),
                    "시대": meta.get("시대", ""),
                    "소재지상세": meta.get("소재지상세", "")
                })
    
    return pd.DataFrame(results)

# 9. 결과 시각화
def plot_results(df, k=3):
    """
    검색 결과를 시각화합니다.
    :param df: 검색 결과 DataFrame
    :param k: 상위 몇 개 순위를 시각화할지 설정
    """
    plt.figure(figsize=(12, 6))
    normalization_methods = df["정규화 방식"].unique()
    
    for method in normalization_methods:
        method_df = df[(df["정규화 방식"] == method) & (df["순위"] <= k)]
        plt.plot(method_df["가중치 (alpha)"], method_df["점수"], marker='o', label=f"{method} (Top {k})")
    
    plt.title("가중치(alpha)에 따른 점수 변화")
    plt.xlabel("가중치 (alpha)")
    plt.ylabel("점수")
    plt.legend()
    plt.grid()
    plt.show()

# 실행: 사용자 질의와 가중치 테스트
query = "한국에서 가장 오래된 목조 건축물"
print("가중치(alpha)별 하이브리드 서치 결과 계산 중...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
df_results = hybrid_search_for_weights(query)

# 결과 출력 및 저장
print("\n검색 결과 DataFrame:")
print(df_results.head(20))  # 상위 20개 결과만 표시

output_file = "./hybrid_search_results.csv"
df_results.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n검색 결과가 '{output_file}'에 저장되었습니다.")

# 그래프 생성
plot_results(df_results)

# 1. 하이브리드 서치 수행 (기존 코드에서 생성한 df_results 사용)

# 2. 문서 빈도 분석 코드 (기존 데이터프레임 df_results를 활용)
def analyze_document_frequencies(df):
    """
    각 정규화 방식과 가중치(alpha)별로 검색된 문서의 빈도수를 계산하고 정렬합니다.
    """
    # 문서와 해당 국가유산명을 기준으로 빈도수 계산
    doc_frequencies = df.groupby("내용")["순위"].count().reset_index()
    doc_frequencies.columns = ["문서", "검색 빈도"]
    doc_frequencies = doc_frequencies.sort_values(by="검색 빈도", ascending=False)

    # 상위 문서 확인
    print("\n가장 많이 검색된 문서:")
    print(doc_frequencies.head(10))  # 상위 10개만 출력

    # 정규화 방식과 가중치별 빈도 계산
    grouped = df.groupby(["정규화 방식", "가중치 (alpha)", "내용"])["순위"].count().reset_index()
    grouped.columns = ["정규화 방식", "가중치 (alpha)", "문서", "검색 빈도"]
    grouped = grouped.sort_values(by=["정규화 방식", "가중치 (alpha)", "검색 빈도"], ascending=[True, True, False])

    # 결과를 반환
    return doc_frequencies, grouped

# 문서 빈도 분석 실행
doc_frequencies, grouped_results = analyze_document_frequencies(df_results)

# 1. 가장 많이 검색된 문서 출력
print("\n가장 많이 검색된 문서 (Top 10):")
print(doc_frequencies.head(10))

# 2. 정규화 방식과 가중치별 문서 빈도 출력
print("\n정규화 방식과 가중치별 문서 빈도:")
print(grouped_results.head(20))  # 상위 20개만 출력

# 3. 결과를 CSV로 저장
doc_frequencies.to_csv("overall_document_frequencies.csv", index=False, encoding="utf-8-sig")
grouped_results.to_csv("grouped_document_frequencies.csv", index=False, encoding="utf-8-sig")

print("\n결과가 'overall_document_frequencies.csv'와 'grouped_document_frequencies.csv'에 저장되었습니다.")